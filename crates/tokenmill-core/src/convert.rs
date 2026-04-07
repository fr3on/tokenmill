use std::io::{BufRead, Write};
use crate::{Alpaca, ChatML, ChatMLMessage, Format, ShareGPT, ShareGPTConversation};
use serde_json::{self, Value};

pub fn convert<R: BufRead, W: Write + Send + 'static>(
    reader: R,
    mut writer: W,
    from: Format,
    to: Format,
    _skip_errors: bool,
    mut progress: Option<&mut dyn FnMut(u64)>,
) -> anyhow::Result<()> {
    if from == to {
        anyhow::bail!("Source and target formats are the same: {:?}", from);
    }

    if to == Format::Parquet {
        // Source is JSONL (reader), target is Parquet
        let iter = reader.lines().map(|line_result| {
            let line = line_result?;
            serde_json::from_str::<Value>(&line).map_err(Into::into)
        });
        return crate::parquet::write_parquet(writer, from, iter, progress);
    }

    for (i, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let _line_num = (i + 1) as u64;

        if let Some(cb) = progress.as_mut() {
            cb(line.len() as u64 + 1);
        }

        let converted_line = match (from, to) {
            (Format::Alpaca, Format::ShareGPT) => {
                let a: Alpaca = serde_json::from_str(&line)?;
                let s = ShareGPT {
                    conversations: vec![
                        ShareGPTConversation {
                            from: "human".to_string(),
                            value: if a.input.is_empty() { a.instruction } else { format!("{}\n{}", a.instruction, a.input) },
                        },
                        ShareGPTConversation {
                            from: "gpt".to_string(),
                            value: a.output,
                        },
                    ],
                };
                Some(serde_json::to_string(&s)?)
            }
            (Format::Alpaca, Format::ChatML) => {
                let a: Alpaca = serde_json::from_str(&line)?;
                let c = ChatML {
                    messages: vec![
                        ChatMLMessage {
                            role: "user".to_string(),
                            content: if a.input.is_empty() { a.instruction } else { format!("{}\n{}", a.instruction, a.input) },
                        },
                        ChatMLMessage {
                            role: "assistant".to_string(),
                            content: a.output,
                        },
                    ],
                };
                Some(serde_json::to_string(&c)?)
            }
            (Format::ShareGPT, Format::Alpaca) => {
                let s: ShareGPT = serde_json::from_str(&line)?;
                let instruction = s.conversations.iter()
                    .find(|c| c.from == "human")
                    .map(|c| c.value.clone())
                    .unwrap_or_default();
                let output = s.conversations.iter()
                    .find(|c| c.from == "gpt")
                    .map(|c| c.value.clone())
                    .unwrap_or_default();
                let a = Alpaca { instruction, input: String::new(), output };
                Some(serde_json::to_string(&a)?)
            }
            (Format::ShareGPT, Format::ChatML) => {
                let s: ShareGPT = serde_json::from_str(&line)?;
                let c = ChatML {
                    messages: s.conversations.into_iter().map(|conv| {
                        ChatMLMessage {
                            role: match conv.from.as_str() {
                                "human" => "user".to_string(),
                                "gpt" => "assistant".to_string(),
                                other => other.to_string(),
                            },
                            content: conv.value,
                        }
                    }).collect(),
                };
                Some(serde_json::to_string(&c)?)
            }
            (Format::ChatML, Format::Alpaca) => {
                let c: ChatML = serde_json::from_str(&line)?;
                let instruction = c.messages.iter()
                    .find(|m| m.role == "user")
                    .map(|m| m.content.clone())
                    .unwrap_or_default();
                let output = c.messages.iter()
                    .find(|m| m.role == "assistant")
                    .map(|m| m.content.clone())
                    .unwrap_or_default();
                let a = Alpaca { instruction, input: String::new(), output };
                Some(serde_json::to_string(&a)?)
            }
            (Format::ChatML, Format::ShareGPT) => {
                let c: ChatML = serde_json::from_str(&line)?;
                let s = ShareGPT {
                    conversations: c.messages.into_iter().map(|msg| {
                        ShareGPTConversation {
                            from: match msg.role.as_str() {
                                "user" => "human".to_string(),
                                "assistant" => "gpt".to_string(),
                                other => other.to_string(),
                            },
                            value: msg.content,
                        }
                    }).collect(),
                };
                Some(serde_json::to_string(&s)?)
            }
            _ => None,
        };

        if let Some(out) = converted_line {
            writeln!(writer, "{}", out)?;
        }
    }

    Ok(())
}

pub fn convert_from_parquet<R: std::io::Read + std::io::Seek + Send + 'static + parquet::file::reader::ChunkReader, W: Write + Send + 'static>(
    reader: R,
    mut writer: W,
    _from: Format,
    to: Format,
    mut progress: Option<&mut dyn FnMut(u64)>,
) -> anyhow::Result<()> {
    let iter = crate::parquet::read_parquet(reader, to)?;

    for item_result in iter {
        let item: Value = item_result?;
        
        if let Some(cb) = progress.as_mut() {
            // Report progress (rough estimate)
            cb(serde_json::to_string(&item).map(|s| s.len()).unwrap_or(100) as u64);
        }

        let line = serde_json::to_string(&item)?;
        writeln!(writer, "{}", line)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_alpaca_to_sharegpt() {
        let input = r#"{"instruction": "i", "output": "o"}"#;
        let mut output = Vec::new();
        convert(Cursor::new(input), &mut output, Format::Alpaca, Format::ShareGPT, false, None).unwrap();
        let expected = r#"{"conversations":[{"from":"human","value":"i"},{"from":"gpt","value":"o"}]}"#;
        assert_eq!(String::from_utf8(output).unwrap().trim(), expected);
    }
}
