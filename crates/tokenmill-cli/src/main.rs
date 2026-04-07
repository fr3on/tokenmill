mod tui;
mod hf;

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};
use indicatif::ProgressBar;
use tokenmill_core::{
    filter, stats, validate, DedupMethod, FilterOptions, Format,
    MinHashDedupOptions, SemanticDedupOptions, exact_dedup,
    minhash_dedup, semantic_dedup,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Show full TUI dashboard instead of inline progress
    #[arg(long, global = true)]
    tui: bool,

    /// Suppress all progress output
    #[arg(long, global = true)]
    quiet: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Check that every line conforms to the specified schema
    Validate {
        /// Schema format to validate against
        #[arg(short, long)]
        schema: Format,

        /// Fail on the first error instead of collecting all errors
        #[arg(long)]
        strict: bool,

        /// Input file; stdin if omitted
        input: Option<PathBuf>,

        /// Write valid lines here (invalid lines are dropped)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Print a statistical summary to stdout as JSON
    Stats {
        /// Tokenizer to use for counting
        #[arg(short, long, default_value = "cl100k_base")]
        tokenizer: String,

        /// Format to use; auto-detected if omitted
        #[arg(short, long)]
        format: Option<Format>,

        /// Include p25/p50/p75/p90/p99 in output
        #[arg(short, long)]
        percentiles: bool,

        /// Input file; stdin if omitted
        input: Option<PathBuf>,
    },
    /// Remove samples that fail quality criteria
    Filter {
        /// Remove samples with fewer tokens
        #[arg(long)]
        min_tokens: Option<u32>,

        /// Remove samples with more tokens
        #[arg(long)]
        max_tokens: Option<u32>,

        /// Character-level lower bound
        #[arg(long)]
        min_chars: Option<u32>,

        /// Character-level upper bound
        #[arg(long)]
        max_chars: Option<u32>,

        /// Strip samples matching built-in refusal patterns
        #[arg(long)]
        remove_refusals: bool,

        /// Tokenizer to use for length filtering
        #[arg(long, default_value = "cl100k_base")]
        tokenizer: String,

        /// Perplexity threshold (items above this are filtered)
        #[arg(long)]
        perplexity_threshold: Option<f32>,

        /// Path to model directory for perplexity scoring
        #[arg(long)]
        model_path: Option<String>,

        /// Write results here; stdout if omitted
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Input file; stdin if omitted
        input: Option<PathBuf>,

        /// Print a filter summary to stderr when done
        #[arg(long)]
        stats: bool,
    },
    /// Convert between dataset formats
    Convert {
        /// Source format
        #[arg(long)]
        from: Format,

        /// Target format
        #[arg(long)]
        to: Format,

        /// Write results here; stdout if omitted
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Skip malformed lines instead of aborting
        #[arg(long)]
        skip_errors: bool,

        /// Input file; stdin if omitted
        input: Option<PathBuf>,
    },
    /// Remove duplicate or near-duplicate samples from a JSONL file
    Dedup {
        /// Deduplication method
        #[arg(short, long, default_value = "exact")]
        method: DedupMethod,

        /// Jaccard similarity cutoff for minhash
        #[arg(short, long, default_value = "0.8")]
        threshold: f64,

        /// Skip malformed lines instead of aborting
        #[arg(long)]
        skip_errors: bool,

        /// Seed for deterministic tie-breaking or MinHash
        #[arg(short, long, default_value = "0")]
        seed: u64,

        /// Path to model directory (required for Semantic)
        #[arg(long)]
        model_path: Option<String>,

        /// Write results here; stdout if omitted
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Input file; stdin if omitted
        input: Option<PathBuf>,
    },
    /// Randomly sample N lines using reservoir sampling
    Sample {
        /// Number of samples to take
        #[arg(short, long)]
        n: usize,

        /// Seed for deterministic sampling
        #[arg(short, long, default_value = "0")]
        seed: u64,

        /// Write results here; stdout if omitted
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Input file; stdin if omitted
        input: Option<PathBuf>,
    },
    /// Hugging Face Hub operations
    Hf {
        #[command(subcommand)]
        command: HfCommand,
    },
}

/// Hugging Face repository type
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum HfRepoType {
    Model,
    Dataset,
    Space,
}

impl From<HfRepoType> for hf_hub::RepoType {
    fn from(t: HfRepoType) -> Self {
        match t {
            HfRepoType::Model   => hf_hub::RepoType::Model,
            HfRepoType::Dataset => hf_hub::RepoType::Dataset,
            HfRepoType::Space   => hf_hub::RepoType::Space,
        }
    }
}

#[derive(Subcommand)]
pub enum HfCommand {
    /// List all files inside a Hugging Face repo
    List {
        /// Repository ID (e.g. "HuggingFaceFW/fineweb")
        repo_id: String,
        /// Repository type
        #[arg(long, default_value = "dataset")]
        r#type: HfRepoType,
    },
    /// Download a file from the Hub
    Download {
        /// Repository ID (e.g. "bert-base-uncased")
        repo_id: String,
        /// Filename to download
        filename: String,
        /// Revision (branch, tag, or commit hash)
        #[arg(long)]
        revision: Option<String>,
        /// Repository type
        #[arg(long, default_value = "dataset")]
        r#type: HfRepoType,
        /// Local output path (default: print cached path)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {:#}", e);
        std::process::exit(1);
    }
}

fn open_reader(path: Option<PathBuf>, format: Option<Format>) -> anyhow::Result<Box<dyn io::BufRead + Send>> {
    match path {
        Some(p) => {
            let p_str = p.to_string_lossy();
            let actual_path = if p_str.starts_with("hf://") {
                hf::resolve_hf_uri(&p_str)?
            } else {
                p
            };

            let f = File::open(&actual_path)
                .with_context(|| format!("cannot open '{}'", actual_path.display()))?;
            
            if actual_path.extension().is_some_and(|ext| ext == "parquet") {
                let fmt = format.unwrap_or(Format::Alpaca);
                Ok(Box::new(tokenmill_core::ParquetLineReader::new(f, fmt)?))
            } else {
                Ok(Box::new(BufReader::new(f)))
            }
        }
        None => Ok(Box::new(BufReader::new(io::stdin()))),
    }
}

fn open_seek_reader(path: Option<PathBuf>) -> anyhow::Result<File> {
    match path {
        Some(p) => {
            let p_str = p.to_string_lossy();
            let actual_path = if p_str.starts_with("hf://") {
                hf::resolve_hf_uri(&p_str)?
            } else {
                p
            };

            let f = File::open(&actual_path)
                .with_context(|| format!("cannot open '{}'", actual_path.display()))?;
            Ok(f)
        }
        None => anyhow::bail!("Parquet input must be a file (stdin not supported for Seek)"),
    }
}

fn open_writer(path: Option<PathBuf>) -> anyhow::Result<Box<dyn Write + Send>> {
    match path {
        Some(p) => {
            let f = File::create(&p)
                .with_context(|| format!("cannot create '{}'", p.display()))?;
            Ok(Box::new(BufWriter::new(f)))
        }
        None => Ok(Box::new(BufWriter::new(io::stdout()))),
    }
}

fn make_progress(path: &Option<PathBuf>, label: &'static str) -> ProgressBar {
    match path {
        Some(p) => {
            let size = std::fs::metadata(p)
                .map(|m| m.len())
                .unwrap_or(0);
            let pb = ProgressBar::new(size);
            pb.set_style(
                indicatif::ProgressStyle::with_template(
                    " {spinner:.cyan} {msg} [{bar:40.purple/dim}] {bytes}/{total_bytes} ({eta})"
                )
                .unwrap()
                .progress_chars("█▓░"),
            );
            pb.set_message(label);
            pb
        }
        None => {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                indicatif::ProgressStyle::with_template(
                    " {spinner:.cyan} {msg} {bytes} processed"
                )
                .unwrap(),
            );
            pb.set_message(label);
            pb.enable_steady_tick(std::time::Duration::from_millis(80));
            pb
        }
    }
}

fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.tui && cli.quiet {
        anyhow::bail!("--tui and --quiet are mutually exclusive");
    }

    // `--tui` with no subcommand → standalone system-monitor dashboard
    let command = match cli.command {
        Some(c) => c,
        None => {
            if cli.tui {
                return tui::run_tui_standalone().map(|_| ());
            }
            // No subcommand and no --tui: let clap print the normal help
            use clap::CommandFactory;
            Cli::command().print_help()?;
            println!();
            return Ok(());
        }
    };

    match command {
        Commands::Validate { schema, strict, input, output } => {
            let reader = open_reader(input.clone(), Some(schema))?;
            let writer = output.clone().map(|p| open_writer(Some(p))).transpose()?;

            if cli.tui {
                let (tx, rx) = std::sync::mpsc::channel();
                let input_clone = input.clone();
                std::thread::spawn(move || {
                    let mut bytes_processed = 0;
                    let mut lines_processed = 0;
                    let mut cb = |n: u64| {
                        bytes_processed += n;
                        lines_processed += 1;
                        if lines_processed % 1000 == 0 {
                            let _ = tx.send(tokenmill_core::ProgressUpdate {
                                lines_processed,
                                bytes_processed,
                                current_format: Some(schema),
                                status: "Validating...".to_string(),
                                ..Default::default()
                            });
                        }
                    };
                    let _ = validate(reader, writer, schema, strict, Some(&mut cb));
                    let _ = tx.send(tokenmill_core::ProgressUpdate {
                        lines_processed,
                        bytes_processed,
                        current_format: Some(schema),
                        status: "Finished".to_string(),
                        ..Default::default()
                    });
                });
                let total_bytes = input_clone.and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len());
                tui::run_tui(rx, total_bytes)?;
            } else if cli.quiet {
                validate(reader, writer, schema, strict, None)?;
            } else {
                let pb = make_progress(&input, "Validating");
                let mut cb = |n| pb.inc(n);
                validate(reader, writer, schema, strict, Some(&mut cb))?;
                pb.finish_with_message("Done!");
            }
        }
        Commands::Stats { tokenizer, format, percentiles, input } => {
            let reader = open_reader(input.clone(), format)?;

            if cli.tui {
                let (tx, rx) = std::sync::mpsc::channel();
                let input_clone = input.clone();
                std::thread::spawn(move || {
                    let mut bytes_processed = 0;
                    let mut lines_processed = 0;
                    let mut cb = |n: u64| {
                        bytes_processed += n;
                        lines_processed += 1;
                        if lines_processed % 1000 == 0 {
                            let _ = tx.send(tokenmill_core::ProgressUpdate {
                                lines_processed,
                                bytes_processed,
                                current_format: format,
                                status: "Collecting stats...".to_string(),
                                ..Default::default()
                            });
                        }
                    };
                    let summary = stats(reader, format, &tokenizer, percentiles, Some(&mut cb));
                    if let Ok(s) = summary {
                        let output = tokenmill_core::serde_json::to_string_pretty(&s).ok();
                        let _ = tx.send(tokenmill_core::ProgressUpdate {
                            lines_processed,
                            bytes_processed,
                            current_format: format,
                            status: "Finished".to_string(),
                            output,
                        });
                    }
                });
                let total_bytes = input_clone.and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len());
                if let Some(out) = tui::run_tui(rx, total_bytes)? {
                    println!("{}", out);
                }
            } else if cli.quiet {
                let summary = stats(reader, format, &tokenizer, percentiles, None)?;
                println!("{}", tokenmill_core::serde_json::to_string_pretty(&summary)?);
            } else {
                let pb = make_progress(&input, "Stats");
                let mut cb = |n| pb.inc(n);
                let summary = stats(reader, format, &tokenizer, percentiles, Some(&mut cb))?;
                pb.finish_with_message("Done!");
                println!("{}", tokenmill_core::serde_json::to_string_pretty(&summary)?);
            }
        }
        Commands::Filter {
            min_tokens,
            max_tokens,
            min_chars,
            max_chars,
            remove_refusals,
            tokenizer,
            perplexity_threshold,
            model_path,
            output,
            input,
            stats: show_stats,
        } => {
            let reader = open_reader(input.clone(), None)?;
            let writer = open_writer(output)?;

            let options = FilterOptions {
                min_tokens,
                max_tokens,
                min_chars,
                max_chars,
                remove_refusals,
                tokenizer_name: Some(tokenizer),
                perplexity_threshold,
                model_path,
            };

            if cli.tui {
                let (tx, rx) = std::sync::mpsc::channel();
                let input_clone = input.clone();
                std::thread::spawn(move || {
                    let mut bytes_processed = 0;
                    let mut lines_processed = 0;
                    let mut cb = |n: u64| {
                        bytes_processed += n;
                        lines_processed += 1;
                        if lines_processed % 1000 == 0 {
                            let _ = tx.send(tokenmill_core::ProgressUpdate {
                                lines_processed,
                                bytes_processed,
                                current_format: None,
                                status: "Filtering...".to_string(),
                                ..Default::default()
                            });
                        }
                    };
                    let _ = filter(reader, writer, options, Some(&mut cb));
                    let _ = tx.send(tokenmill_core::ProgressUpdate {
                        lines_processed,
                        bytes_processed,
                        current_format: None,
                        status: "Finished".to_string(),
                        ..Default::default()
                    });
                });
                let total_bytes = input_clone.and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len());
                tui::run_tui(rx, total_bytes)?;
            } else if cli.quiet {
                filter(reader, writer, options, None)?;
            } else {
                let pb = make_progress(&input, "Filtering");
                let mut cb = |n| pb.inc(n);
                let summary = filter(reader, writer, options, Some(&mut cb))?;
                pb.finish_with_message("Done!");
                if show_stats {
                    eprintln!("[filter] kept: {}  removed: {}  errors: {}", summary.kept, summary.removed, summary.error);
                }
            }
        }
        Commands::Convert {
            from, to, output, skip_errors, input,
        } => {
            let writer = open_writer(output)?;

            if cli.tui {
                let (tx, rx) = std::sync::mpsc::channel();
                let input_clone = input.clone();
                std::thread::spawn(move || {
                    let mut bytes_processed = 0;
                    let mut lines_processed = 0;
                    let mut cb = |n: u64| {
                        bytes_processed += n;
                        lines_processed += 1;
                        if lines_processed % 1000 == 0 {
                            let _ = tx.send(tokenmill_core::ProgressUpdate {
                                lines_processed,
                                bytes_processed,
                                current_format: Some(from),
                                status: "Converting...".to_string(),
                                ..Default::default()
                            });
                        }
                    };

                    if from == Format::Parquet {
                        let reader = open_seek_reader(input_clone).expect("Seek reader needed");
                        let _ = tokenmill_core::convert_from_parquet(reader, writer, from, to, Some(&mut cb));
                    } else {
                        let reader = open_reader(input_clone, Some(from)).expect("Reader needed");
                        let _ = tokenmill_core::convert(reader, writer, from, to, skip_errors, Some(&mut cb));
                    }

                    let _ = tx.send(tokenmill_core::ProgressUpdate {
                        lines_processed,
                        bytes_processed,
                        current_format: Some(from),
                        status: "Finished".to_string(),
                        ..Default::default()
                    });
                });
                let total_bytes = input.clone().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len());
                tui::run_tui(rx, total_bytes)?;
            } else if cli.quiet {
                if from == Format::Parquet {
                    let reader = open_seek_reader(input)?;
                    tokenmill_core::convert_from_parquet(reader, writer, from, to, None)?;
                } else {
                    let reader = open_reader(input, Some(from))?;
                    tokenmill_core::convert(reader, writer, from, to, skip_errors, None)?;
                }
            } else {
                let pb = make_progress(&input, "Converting");
                let mut cb = |n| pb.inc(n);
                if from == Format::Parquet {
                    let reader = open_seek_reader(input)?;
                    tokenmill_core::convert_from_parquet(reader, writer, from, to, Some(&mut cb))?;
                } else {
                    let reader = open_reader(input, Some(from))?;
                    tokenmill_core::convert(reader, writer, from, to, skip_errors, Some(&mut cb))?;
                }
                pb.finish_with_message("Done!");
            }
        }
        Commands::Dedup {
            method, threshold, output, skip_errors, seed, model_path, input,
        } => {
            let reader = open_reader(input.clone(), None)?;
            let writer = open_writer(output)?;

            if cli.tui {
                let (tx, rx) = std::sync::mpsc::channel();
                let input_clone = input.clone();
                let model_path = model_path.clone();
                std::thread::spawn(move || {
                    let mut bytes_processed = 0;
                    let mut lines_processed = 0;
                    let mut cb = |n: u64| {
                        bytes_processed += n;
                        lines_processed += 1;
                        if lines_processed % 1000 == 0 {
                            let _ = tx.send(tokenmill_core::ProgressUpdate {
                                lines_processed,
                                bytes_processed,
                                current_format: None,
                                status: "Deduplicating...".to_string(),
                                ..Default::default()
                            });
                        }
                    };
                    match method {
                        DedupMethod::Exact => { let _ = exact_dedup(reader, writer, skip_errors, Some(&mut cb)); }
                        DedupMethod::Minhash => {
                            let options = MinHashDedupOptions { threshold, seed };
                            let _ = minhash_dedup(reader, writer, options, skip_errors, Some(&mut cb));
                        }
                        DedupMethod::Semantic => {
                            if let Some(path) = model_path {
                                let options = SemanticDedupOptions { threshold, model_path: path };
                                let _ = semantic_dedup(reader, writer, options, Some(&mut cb));
                            }
                        }
                    }
                    let _ = tx.send(tokenmill_core::ProgressUpdate {
                        lines_processed,
                        bytes_processed,
                        current_format: None,
                        status: "Finished".to_string(),
                        ..Default::default()
                    });
                });
                let total_bytes = input_clone.and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len());
                tui::run_tui(rx, total_bytes)?;
            } else if cli.quiet {
                match method {
                    DedupMethod::Exact => { exact_dedup(reader, writer, skip_errors, None)?; }
                    DedupMethod::Minhash => {
                        let options = MinHashDedupOptions { threshold, seed };
                        minhash_dedup(reader, writer, options, skip_errors, None)?;
                    }
                    DedupMethod::Semantic => {
                        let path = model_path.ok_or_else(|| anyhow::anyhow!("--model-path is required for semantic deduplication"))?;
                        let options = SemanticDedupOptions { threshold, model_path: path };
                        semantic_dedup(reader, writer, options, None)?;
                    }
                }
            } else {
                let pb = make_progress(&input, "Deduplicating");
                let mut cb = |n| pb.inc(n);
                let kept = match method {
                    DedupMethod::Exact => exact_dedup(reader, writer, skip_errors, Some(&mut cb))?,
                    DedupMethod::Minhash => {
                        let options = MinHashDedupOptions { threshold, seed };
                        minhash_dedup(reader, writer, options, skip_errors, Some(&mut cb))?
                    }
                    DedupMethod::Semantic => {
                        let path = model_path.ok_or_else(|| anyhow::anyhow!("--model-path is required for semantic deduplication"))?;
                        let options = SemanticDedupOptions { threshold, model_path: path };
                        semantic_dedup(reader, writer, options, Some(&mut cb))?
                    }
                };
                pb.finish_with_message(format!("Done! Kept {} lines.", kept));
            }
        }
        Commands::Sample { n, seed, output, input } => {
            let reader = open_reader(input.clone(), None)?;
            let writer = open_writer(output)?;

            if cli.tui {
                let (tx, rx) = std::sync::mpsc::channel();
                let input_clone = input.clone();
                std::thread::spawn(move || {
                    let mut bytes_processed = 0;
                    let mut lines_processed = 0;
                    let mut cb = |n: u64| {
                        bytes_processed += n;
                        lines_processed += 1;
                        if lines_processed % 1000 == 0 {
                            let _ = tx.send(tokenmill_core::ProgressUpdate {
                                lines_processed,
                                bytes_processed,
                                current_format: None,
                                status: "Sampling...".to_string(),
                                ..Default::default()
                            });
                        }
                    };

                    let _ = tokenmill_core::sample::sample(reader, writer, n, seed, Some(&mut cb));

                    let _ = tx.send(tokenmill_core::ProgressUpdate {
                        lines_processed,
                        bytes_processed,
                        current_format: None,
                        status: "Finished".to_string(),
                        ..Default::default()
                    });
                });
                let total_bytes = input_clone.and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len());
                tui::run_tui(rx, total_bytes)?;
            } else if cli.quiet {
                tokenmill_core::sample::sample(reader, writer, n, seed, None)?;
            } else {
                let pb = make_progress(&input, "Sampling");
                let mut cb = |n| pb.inc(n);
                let total = tokenmill_core::sample::sample(reader, writer, n, seed, Some(&mut cb))?;
                pb.finish_with_message(format!("Done! Sampled {} lines out of {}.", n, total));
            }
        }
        Commands::Hf { command } => {
            match command {
                HfCommand::List { repo_id, r#type } => {
                    let files = hf::list_repo(&repo_id, r#type.into())?;
                    println!("{} files in {}:", files.len(), repo_id);
                    for (name, _size) in &files {
                        println!("  {}", name);
                    }
                }
                HfCommand::Download { repo_id, filename, revision, r#type, output } => {
                    let path = hf::download(&repo_id, &filename, revision.as_deref(), r#type.into())?;
                    if let Some(out_path) = output {
                        std::fs::copy(&path, &out_path)
                            .with_context(|| format!("Failed to copy {} to {}", path.display(), out_path.display()))?;
                        println!("Downloaded and copied to: {}", out_path.display());
                    } else {
                        println!("File cached at: {}", path.display());
                    }
                }
            }
        }
    }

    Ok(())
}
