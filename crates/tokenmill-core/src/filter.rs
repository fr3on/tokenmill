use std::io::{BufRead, Write};
use crate::{Alpaca, ChatML, Format, ShareGPT, inference::Scorer};
use tiktoken_rs::cl100k_base;
use serde::{Deserialize, Serialize};

const REFUSAL_PATTERNS: &[&str] = &[
    "as an ai language model",
    "as a large language model",
    "as an ai assistant",
    "i cannot and will not",
    "i'm not able to provide",
    "i am not able to provide",
    "it is not appropriate for me",
    "i must inform you that",
];

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct FilterOptions {
    pub min_tokens: Option<u32>,
    pub max_tokens: Option<u32>,
    pub min_chars: Option<u32>,
    pub max_chars: Option<u32>,
    pub remove_refusals: bool,
    pub tokenizer_name: Option<String>,
    pub perplexity_threshold: Option<f32>,
    pub model_path: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct FilterSummary {
    pub kept: u64,
    pub removed: u64,
    pub error: u64,
}

pub fn filter<R: BufRead, W: Write>(
    reader: R,
    mut writer: W,
    options: FilterOptions,
    mut progress: Option<&mut dyn FnMut(u64)>,
) -> anyhow::Result<FilterSummary> {
    let bpe = if options.min_tokens.is_some() || options.max_tokens.is_some() {
        let name = options.tokenizer_name.as_deref().unwrap_or("cl100k_base");
        Some(match name {
            "cl100k_base" => cl100k_base()?,
            _ => anyhow::bail!("Unsupported tokenizer: {}", name),
        })
    } else {
        None
    };

    let mut scorer = if let Some(ref model_path) = options.model_path {
        if options.perplexity_threshold.is_some() {
            Some(Scorer::load_qwen2(std::path::Path::new(model_path))?)
        } else {
            None
        }
    } else {
        None
    };

    let mut summary = FilterSummary::default();

    for line_result in reader.lines() {
        let line = line_result?;

        if let Some(cb) = progress.as_mut() {
            cb(line.len() as u64 + 1);
        }

        let format = match crate::detect(&line) {
            Ok(f) => f,
            Err(_) => {
                summary.error += 1;
                continue;
            }
        };

        let mut should_keep = true;

        // Check refusal first as it doesn't need tokenization
        if options.remove_refusals {
            let output_text = match format {
                Format::Alpaca => {
                    let a: Alpaca = serde_json::from_str(&line)?;
                    Some(a.output)
                }
                Format::ShareGPT => {
                    let s: ShareGPT = serde_json::from_str(&line)?;
                    Some(s.conversations.iter()
                        .filter(|c| c.from == "gpt" || c.from == "assistant")
                        .map(|c| c.value.as_str())
                        .collect::<Vec<_>>()
                        .join("\n"))
                }
                Format::ChatML => {
                    let c: ChatML = serde_json::from_str(&line)?;
                    Some(c.messages.iter()
                        .filter(|m| m.role == "assistant" || m.role == "gpt")
                        .map(|m| m.content.as_str())
                        .collect::<Vec<_>>()
                        .join("\n"))
                }
                Format::Parquet | Format::Arrow => None,
            };

            if let Some(text) = output_text {
                let lower_text = text.to_lowercase();
                for pattern in REFUSAL_PATTERNS {
                    if lower_text.contains(pattern) {
                        should_keep = false;
                        break;
                    }
                }
            }
        }

        if !should_keep {
            summary.removed += 1;
            continue;
        }

        // Canonical text for other filters
        let canonical_text = match format {
            Format::Alpaca => serde_json::from_str::<Alpaca>(&line)?.canonical_text(),
            Format::ShareGPT => serde_json::from_str::<ShareGPT>(&line)?.canonical_text(),
            Format::ChatML => serde_json::from_str::<ChatML>(&line)?.canonical_text(),
            Format::Parquet | Format::Arrow => anyhow::bail!("Parquet/Arrow not supported in filter"),
        };

        // Char filters
        let char_count = canonical_text.chars().count() as u32;
        if let Some(min) = options.min_chars {
            if char_count < min {
                should_keep = false;
            }
        }
        if let Some(max) = options.max_chars {
            if char_count > max {
                should_keep = false;
            }
        }

        if !should_keep {
            summary.removed += 1;
            continue;
        }

        // Token filters
        if let Some(ref bpe_engine) = bpe {
            let token_count = bpe_engine.encode_with_special_tokens(&canonical_text).len() as u32;
            if let Some(min) = options.min_tokens {
                if token_count < min {
                    should_keep = false;
                }
            }
            if let Some(max) = options.max_tokens {
                if token_count > max {
                    should_keep = false;
                }
            }
        }

        // Perplexity filter
        if should_keep {
            if let Some(ref mut s) = scorer {
                if let Some(threshold) = options.perplexity_threshold {
                    let ppl = s.compute_perplexity(&canonical_text).unwrap_or(f32::MAX);
                    if ppl > threshold {
                        should_keep = false;
                    }
                }
            }
        }

        if should_keep {
            summary.kept += 1;
            writeln!(writer, "{}", line)?;
        } else {
            summary.removed += 1;
        }
    }

    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_filter_chars() {
        let input = r#"{"instruction": "short", "output": "short"}"#;
        let reader = Cursor::new(input);
        let mut output = Vec::new();
        let options = FilterOptions {
            min_chars: Some(100),
            ..FilterOptions::default()
        };
        let summary = filter(reader, &mut output, options, None).unwrap();
        assert_eq!(summary.kept, 0);
        assert_eq!(summary.removed, 1);
    }

    #[test]
    fn test_filter_refusals() {
        let input = r#"{"instruction": "i", "output": "as an ai language model"}"#;
        let reader = Cursor::new(input);
        let mut output = Vec::new();
        let options = FilterOptions {
            remove_refusals: true,
            ..FilterOptions::default()
        };
        let summary = filter(reader, &mut output, options, None).unwrap();
        assert_eq!(summary.kept, 0);
        assert_eq!(summary.removed, 1);
    }
}
