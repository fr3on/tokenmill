use std::io::{Read, Write, Seek, BufRead};
use std::sync::Arc;
use anyhow::{Context, Result};
use arrow::array::{Array, ArrayRef, StringArray, ListArray, StructArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use serde_json::{json, Value};
use crate::Format;

/// A reader that implements BufRead by yielding JSONL lines from a Parquet file.
pub struct ParquetLineReader {
    iter: Box<dyn Iterator<Item = Result<Value>> + Send>,
    current_line: Option<Vec<u8>>,
    pos: usize,
}

impl ParquetLineReader {
    pub fn new<R: Read + Seek + Send + 'static + parquet::file::reader::ChunkReader>(reader: R, format: Format) -> Result<Self> {
        let iter = read_parquet(reader, format)?;
        Ok(Self {
            iter,
            current_line: None,
            pos: 0,
        })
    }
}

impl Read for ParquetLineReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut n = 0;
        while n < buf.len() {
            if self.current_line.is_none() {
                match self.iter.next() {
                    Some(Ok(val)) => {
                        let mut line = serde_json::to_vec(&val).map_err(std::io::Error::other)?;
                        line.push(b'\n');
                        self.current_line = Some(line);
                        self.pos = 0;
                    }
                    Some(Err(e)) => return Err(std::io::Error::other(e)),
                    None => break,
                }
            }

            if let Some(line) = &self.current_line {
                let remaining = line.len() - self.pos;
                let to_copy = std::cmp::min(remaining, buf.len() - n);
                buf[n..n+to_copy].copy_from_slice(&line[self.pos..self.pos+to_copy]);
                n += to_copy;
                self.pos += to_copy;
                if self.pos >= line.len() {
                    self.current_line = None;
                }
            }
        }
        Ok(n)
    }
}

impl BufRead for ParquetLineReader {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        if self.current_line.is_none() {
            match self.iter.next() {
                Some(Ok(val)) => {
                    let mut line = serde_json::to_vec(&val).map_err(std::io::Error::other)?;
                    line.push(b'\n');
                    self.current_line = Some(line);
                    self.pos = 0;
                }
                Some(Err(e)) => return Err(std::io::Error::other(e)),
                None => return Ok(&[]),
            }
        }
        Ok(&self.current_line.as_ref().unwrap()[self.pos..])
    }

    fn consume(&mut self, amt: usize) {
        self.pos += amt;
        if let Some(line) = &self.current_line {
            if self.pos >= line.len() {
                self.current_line = None;
            }
        }
    }
}

unsafe impl Send for ParquetLineReader {}

/// Read a Parquet file and convert each row to a JSON Value.
pub fn read_parquet<R: Read + Seek + Send + 'static + parquet::file::reader::ChunkReader>(
    reader: R,
    format: Format,
) -> Result<Box<dyn Iterator<Item = Result<Value>> + Send>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(reader)
        .context("Failed to create Parquet reader")?;
    
    let arrow_reader = builder.build().context("Failed to build Parquet reader")?;
    
    let iter = arrow_reader.flat_map(move |batch_result| {
        let batch = match batch_result {
            Ok(b) => b,
            Err(e) => return vec![Err(anyhow::anyhow!("Parquet read error: {}", e))].into_iter(),
        };
        
        match record_batch_to_json(&batch, format) {
            Ok(values) => values.into_iter().map(Ok).collect::<Vec<_>>().into_iter(),
            Err(e) => vec![Err(e)].into_iter(),
        }
    });

    Ok(Box::new(iter))
}

fn record_batch_to_json(batch: &RecordBatch, format: Format) -> Result<Vec<Value>> {
    let num_rows = batch.num_rows();
    let mut results = Vec::with_capacity(num_rows);

    match format {
        Format::Alpaca => {
            let instruction_arr = batch.column(0).as_any().downcast_ref::<StringArray>()
                .context("Alpaca Parquet: Column 0 ('instruction') is not a StringArray")?;
            let input_arr = batch.column(1).as_any().downcast_ref::<StringArray>()
                .context("Alpaca Parquet: Column 1 ('input') is not a StringArray")?;
            let output_arr = batch.column(2).as_any().downcast_ref::<StringArray>()
                .context("Alpaca Parquet: Column 2 ('output') is not a StringArray")?;

            for i in 0..num_rows {
                results.push(json!({
                    "instruction": instruction_arr.value(i),
                    "input": input_arr.value(i),
                    "output": output_arr.value(i),
                }));
            }
        }
        Format::ShareGPT => {
            let schema = batch.schema();
            let conv_field = schema.field(0);
            let conv_list = batch.column(0).as_any().downcast_ref::<ListArray>()
                .context(format!("ShareGPT Parquet: Column 0 ({}) is not a ListArray", conv_field.name()))?;
            let conv_structs = conv_list.values().as_any().downcast_ref::<StructArray>()
                .context("ShareGPT Parquet: List elements are not structs")?;
            
            let from_arr = conv_structs.column(0).as_any().downcast_ref::<StringArray>()
                .context("ShareGPT Parquet: 'from' is not a StringArray")?;
            let value_arr = conv_structs.column(1).as_any().downcast_ref::<StringArray>()
                .context("ShareGPT Parquet: 'value' is not a StringArray")?;

            for i in 0..num_rows {
                let start = conv_list.value_offsets()[i] as usize;
                let end = conv_list.value_offsets()[i+1] as usize;
                let mut conversations = Vec::new();
                for j in start..end {
                    conversations.push(json!({
                        "from": from_arr.value(j),
                        "value": value_arr.value(j),
                    }));
                }
                results.push(json!({ "conversations": conversations }));
            }
        }
        Format::ChatML => {
            let _schema = batch.schema();
            let msg_list = batch.column(0).as_any().downcast_ref::<ListArray>()
                .context("ChatML Parquet: Column 0 ('messages') is not a ListArray")?;
            let msg_structs = msg_list.values().as_any().downcast_ref::<StructArray>()
                .context("ChatML Parquet: List elements are not structs")?;
            
            let role_arr = msg_structs.column(0).as_any().downcast_ref::<StringArray>()
                .context("ChatML Parquet: 'role' is not a StringArray")?;
            let content_arr = msg_structs.column(1).as_any().downcast_ref::<StringArray>()
                .context("ChatML Parquet: 'content' is not a StringArray")?;

            for i in 0..num_rows {
                let start = msg_list.value_offsets()[i] as usize;
                let end = msg_list.value_offsets()[i+1] as usize;
                let mut messages = Vec::new();
                for j in start..end {
                    messages.push(json!({
                        "role": role_arr.value(j),
                        "content": content_arr.value(j),
                    }));
                }
                results.push(json!({ "messages": messages }));
            }
        }
        _ => anyhow::bail!("Unsupported Parquet format: {:?}", format),
    }

    Ok(results)
}

/// Write a stream of JSON values to a Parquet file.
pub fn write_parquet<W: Write + Send + 'static>(
    writer: W,
    format: Format,
    items: impl Iterator<Item = Result<Value>>,
    mut progress: Option<&mut dyn FnMut(u64)>,
) -> Result<()> {
    let schema = get_arrow_schema(format)?;
    let mut arrow_writer = ArrowWriter::try_new(writer, schema.clone(), None)?;

    // Buffer items into RecordBatches
    let batch_size = 1000;
    let mut current_items = Vec::with_capacity(batch_size);

    for item_result in items {
        let item = item_result?;
        
        if let Some(cb) = progress.as_mut() {
            // Estimate size for progress (roughly JSON size)
            cb(serde_json::to_string(&item).map(|s| s.len()).unwrap_or(100) as u64);
        }

        current_items.push(item);

        if current_items.len() >= batch_size {
            let batch = json_to_record_batch(&current_items, format, &schema)?;
            arrow_writer.write(&batch)?;
            current_items.clear();
        }
    }

    if !current_items.is_empty() {
        let batch = json_to_record_batch(&current_items, format, &schema)?;
        arrow_writer.write(&batch)?;
    }

    arrow_writer.close()?;
    Ok(())
}

fn get_arrow_schema(format: Format) -> Result<SchemaRef> {
    let schema = match format {
        Format::Alpaca => Schema::new(vec![
            Field::new("instruction", DataType::Utf8, false),
            Field::new("input", DataType::Utf8, false),
            Field::new("output", DataType::Utf8, false),
        ]),
        Format::ShareGPT => {
            let conv_fields = vec![
                Field::new("from", DataType::Utf8, false),
                Field::new("value", DataType::Utf8, false),
            ];
            Schema::new(vec![
                Field::new(
                    "conversations",
                    DataType::List(Arc::new(Field::new("item", DataType::Struct(conv_fields.into()), false))),
                    false,
                ),
            ])
        }
        Format::ChatML => {
            let msg_fields = vec![
                Field::new("role", DataType::Utf8, false),
                Field::new("content", DataType::Utf8, false),
            ];
            Schema::new(vec![
                Field::new(
                    "messages",
                    DataType::List(Arc::new(Field::new("item", DataType::Struct(msg_fields.into()), false))),
                    false,
                ),
            ])
        }
        _ => anyhow::bail!("Unsupported Parquet schema for format: {:?}", format),
    };
    Ok(Arc::new(schema))
}

fn json_to_record_batch(items: &[Value], format: Format, schema: &SchemaRef) -> Result<RecordBatch> {
    let mut columns: Vec<ArrayRef> = Vec::new();

    match format {
        Format::Alpaca => {
            let mut instructions = Vec::new();
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            for item in items {
                instructions.push(item["instruction"].as_str().unwrap_or("").to_string());
                inputs.push(item["input"].as_str().unwrap_or("").to_string());
                outputs.push(item["output"].as_str().unwrap_or("").to_string());
            }
            columns.push(Arc::new(StringArray::from(instructions)));
            columns.push(Arc::new(StringArray::from(inputs)));
            columns.push(Arc::new(StringArray::from(outputs)));
        }
        Format::ShareGPT => {
            let mut froms = Vec::new();
            let mut values = Vec::new();
            let mut offsets = Vec::with_capacity(items.len() + 1);
            offsets.push(0);
            
            for item in items {
                if let Some(convs) = item["conversations"].as_array() {
                    for conv in convs {
                        froms.push(conv["from"].as_str().unwrap_or("").to_string());
                        values.push(conv["value"].as_str().unwrap_or("").to_string());
                    }
                }
                offsets.push(froms.len() as i32);
            }

            let from_arr = Arc::new(StringArray::from(froms)) as ArrayRef;
            let value_arr = Arc::new(StringArray::from(values)) as ArrayRef;
            
            let struct_fields = match &schema.field(0).data_type() {
                DataType::List(field) => match field.data_type() {
                    DataType::Struct(fields) => fields.clone(),
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            
            let struct_arr = StructArray::try_new(
                struct_fields,
                vec![from_arr, value_arr],
                None,
            )?;

            let list_arr = ListArray::try_new(
                Arc::new(Field::new("item", struct_arr.data_type().clone(), false)),
                arrow::buffer::OffsetBuffer::new(arrow::buffer::ScalarBuffer::from(offsets)),
                Arc::new(struct_arr),
                None,
            )?;
            columns.push(Arc::new(list_arr));
        }
        Format::ChatML => {
            let mut roles = Vec::new();
            let mut contents = Vec::new();
            let mut offsets = Vec::with_capacity(items.len() + 1);
            offsets.push(0);
            
            for item in items {
                if let Some(msgs) = item["messages"].as_array() {
                    for msg in msgs {
                        roles.push(msg["role"].as_str().unwrap_or("").to_string());
                        contents.push(msg["content"].as_str().unwrap_or("").to_string());
                    }
                }
                offsets.push(roles.len() as i32);
            }

            let role_arr = Arc::new(StringArray::from(roles)) as ArrayRef;
            let content_arr = Arc::new(StringArray::from(contents)) as ArrayRef;
            
            let struct_fields = match &schema.field(0).data_type() {
                DataType::List(field) => match field.data_type() {
                    DataType::Struct(fields) => fields.clone(),
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            
            let struct_arr = StructArray::try_new(
                struct_fields,
                vec![role_arr, content_arr],
                None,
            )?;

            let list_arr = ListArray::try_new(
                Arc::new(Field::new("item", struct_arr.data_type().clone(), false)),
                arrow::buffer::OffsetBuffer::new(arrow::buffer::ScalarBuffer::from(offsets)),
                Arc::new(struct_arr),
                None,
            )?;
            columns.push(Arc::new(list_arr));
        }
        _ => unreachable!(),
    }

    RecordBatch::try_new(schema.clone(), columns).map_err(Into::into)
}
