use std::io::BufRead;
use crate::{Alpaca, ChatML, Format, ShareGPT};
use tiktoken_rs::cl100k_base;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct StatsSummary {
    pub total_lines: u64,
    pub valid_lines: u64,
    pub invalid_lines: u64,
    pub total_tokens: u64,
    pub mean_tokens: f64,
    pub min_tokens: u64,
    pub max_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub percentiles: Option<Percentiles>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Percentiles {
    pub p25: u64,
    pub p50: u64,
    pub p75: u64,
    pub p90: u64,
    pub p99: u64,
}

pub fn stats<R: BufRead>(
    reader: R,
    format: Option<Format>,
    tokenizer_name: &str,
    calculate_percentiles: bool,
    mut progress: Option<&mut dyn FnMut(u64)>,
) -> anyhow::Result<StatsSummary> {
    let bpe = match tokenizer_name {
        "cl100k_base" => cl100k_base()?,
        _ => anyhow::bail!("Unsupported tokenizer: {}", tokenizer_name),
    };

    let mut total_lines = 0;
    let mut valid_lines = 0;
    let mut invalid_lines = 0;
    let mut total_tokens = 0;
    let mut min_tokens = u64::MAX;
    let mut max_tokens = 0;

    // TODO: Add P2 algorithm for percentiles
    let mut token_counts = if calculate_percentiles { Some(Vec::new()) } else { None };

    for line_result in reader.lines() {
        let line = line_result?;
        total_lines += 1;

        if let Some(cb) = progress.as_mut() {
            cb(line.len() as u64 + 1);
        }

        let fmt = match format {
            Some(f) => f,
            None => {
                if let Ok(f) = crate::detect(&line) {
                    f
                } else {
                    invalid_lines += 1;
                    continue;
                }
            }
        };

        let canonical_text = match fmt {
            Format::Alpaca => serde_json::from_str::<Alpaca>(&line).ok().map(|a| a.canonical_text()),
            Format::ShareGPT => serde_json::from_str::<ShareGPT>(&line).ok().and_then(|s| {
                if s.validate().is_ok() { Some(s.canonical_text()) } else { None }
            }),
            Format::ChatML => serde_json::from_str::<ChatML>(&line).ok().and_then(|c| {
                if c.validate().is_ok() { Some(c.canonical_text()) } else { None }
            }),
            Format::Parquet | Format::Arrow => None,
        };

        if let Some(text) = canonical_text {
            valid_lines += 1;
            let tokens = bpe.encode_with_special_tokens(&text).len() as u64;
            total_tokens += tokens;
            min_tokens = min_tokens.min(tokens);
            max_tokens = max_tokens.max(tokens);
            if let Some(ref mut counts) = token_counts {
                counts.push(tokens);
            }
        } else {
            invalid_lines += 1;
        }
    }

    let mean_tokens = if valid_lines > 0 {
        total_tokens as f64 / valid_lines as f64
    } else {
        0.0
    };

    let percentiles = if let Some(mut counts) = token_counts {
        if valid_lines > 0 {
            counts.sort_unstable();
            let p = |pct: f64| counts[( (pct * (valid_lines as f64 - 1.0)) / 100.0 ).round() as usize];
            Some(Percentiles {
                p25: p(25.0),
                p50: p(50.0),
                p75: p(75.0),
                p90: p(90.0),
                p99: p(99.0),
            })
        } else {
            None
        }
    } else {
        None
    };

    Ok(StatsSummary {
        total_lines,
        valid_lines,
        invalid_lines,
        total_tokens,
        mean_tokens,
        min_tokens: if valid_lines > 0 { min_tokens } else { 0 },
        max_tokens,
        percentiles,
    })
}
