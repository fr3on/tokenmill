use serde::{Deserialize, Serialize};
use serde_json::Value;
pub use serde_json;
pub use rand;
use xxhash_rust::xxh3::xxh3_128;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Alpaca {
    pub instruction: String,
    #[serde(default)]
    pub input: String,
    pub output: String,
}

impl Alpaca {
    pub fn canonical_text(&self) -> String {
        format!("{}\n{}\n{}", self.instruction, self.input, self.output)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShareGPTConversation {
    pub from: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShareGPT {
    pub conversations: Vec<ShareGPTConversation>,
}

impl ShareGPT {
    pub fn canonical_text(&self) -> String {
        self.conversations
            .iter()
            .map(|c| c.value.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn validate(&self) -> Result<(), String> {
        for conv in &self.conversations {
            match conv.from.as_str() {
                "human" | "gpt" | "system" => {}
                _ => return Err(format!("Invalid ShareGPT `from` value: {}", conv.from)),
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMLMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatML {
    pub messages: Vec<ChatMLMessage>,
}

impl ChatML {
    pub fn canonical_text(&self) -> String {
        self.messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn validate(&self) -> Result<(), String> {
        for message in &self.messages {
            match message.role.as_str() {
                "user" | "assistant" | "system" => {}
                _ => return Err(format!("Invalid ChatML `role` value: {}", message.role)),
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Alpaca,
    ShareGPT,
    ChatML,
    Parquet,
    Arrow,
}

#[derive(Debug, Clone, Copy, PartialEq, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
pub enum DedupMethod {
    Exact,
    Minhash,
    Semantic,
}

impl std::str::FromStr for Format {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "alpaca" => Ok(Format::Alpaca),
            "sharegpt" => Ok(Format::ShareGPT),
            "chatml" => Ok(Format::ChatML),
            "parquet" => Ok(Format::Parquet),
            "arrow" => Ok(Format::Arrow),
            _ => anyhow::bail!("Invalid format: {}", s),
        }
    }
}

pub fn detect(line: &str) -> anyhow::Result<Format> {
    let v: Value = serde_json::from_str(line)?;
    let obj = v.as_object().ok_or_else(|| anyhow::anyhow!("Expected JSON object"))?;

    if obj.contains_key("conversations") {
        Ok(Format::ShareGPT)
    } else if obj.contains_key("messages") {
        Ok(Format::ChatML)
    } else if obj.contains_key("instruction") {
        Ok(Format::Alpaca)
    } else {
        anyhow::bail!("Could not auto-detect format from line: {}", line)
    }
}

pub mod dedup;
pub use dedup::{exact_dedup, minhash_dedup, semantic_dedup, MinHashDedupOptions, SemanticDedupOptions};

pub mod validate;
pub mod stats;
pub mod filter;
pub mod convert;
pub mod sample;
pub mod parquet;
pub mod inference;
pub use parquet::{ParquetLineReader, read_parquet, write_parquet};

pub use validate::validate;
pub use stats::stats;
pub use filter::filter;
pub use filter::FilterOptions;
pub use convert::{convert, convert_from_parquet};
pub use sample::sample;

pub use anyhow::Result;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Parse error at line {line}: {source}")]
    ParseError {
        line: u64,
        content: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Format error: {0}")]
    FormatError(String),
    #[error("Unsupported conversion: {0}")]
    UnsupportedConversion(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProgressUpdate {
    pub lines_processed: u64,
    pub bytes_processed: u64,
    pub current_format: Option<Format>,
    pub status: String,
    /// Optional payload (e.g. serialised stats JSON) sent with the "Finished" event.
    #[serde(default)]
    pub output: Option<String>,
}
