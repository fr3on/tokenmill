use std::io::{BufRead, Write};
use crate::{Alpaca, ChatML, Format, ShareGPT};
use serde_json;

#[derive(Debug, Default)]
pub struct ValidationSummary {
    pub valid_lines: u64,
    pub invalid_lines: u64,
}

pub fn validate<R: BufRead, W: Write>(
    reader: R,
    mut writer: Option<W>,
    format: Format,
    strict: bool,
    mut progress: Option<&mut dyn FnMut(u64)>,
) -> anyhow::Result<ValidationSummary> {
    let mut summary = ValidationSummary::default();

    for (i, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line_num = (i + 1) as u64;

        let is_valid = match format {
            Format::Alpaca => serde_json::from_str::<Alpaca>(&line).is_ok(),
            Format::ShareGPT => {
                if let Ok(sgpt) = serde_json::from_str::<ShareGPT>(&line) {
                    sgpt.validate().is_ok()
                } else {
                    false
                }
            }
            Format::ChatML => {
                if let Ok(chatml) = serde_json::from_str::<ChatML>(&line) {
                    chatml.validate().is_ok()
                } else {
                    false
                }
            }
            Format::Parquet | Format::Arrow => false,
        };

        if is_valid {
            summary.valid_lines += 1;
            if let Some(ref mut w) = writer {
                writeln!(w, "{}", line)?;
            }
        } else {
            summary.invalid_lines += 1;
            if strict {
                anyhow::bail!("Invalid line at {}: {}", line_num, line);
            }
        }

        if let Some(cb) = progress.as_mut() {
            cb(line.len() as u64 + 1);
        }
    }

    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_validate_alpaca() {
        let input = r#"{"instruction": "i1", "output": "o1"}
{"instruction": "i2"}
{"instruction": "i3", "output": "o3"}"#;
        let reader = Cursor::new(input);
        let summary = validate(reader, None::<Cursor<Vec<u8>>>, Format::Alpaca, false, None).unwrap();
        assert_eq!(summary.valid_lines, 2);
        assert_eq!(summary.invalid_lines, 1);
    }
}
