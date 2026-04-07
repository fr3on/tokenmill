# tokenmill

> Stream-process LLM fine-tuning datasets at near-disk speed.

`tokenmill` is a single-binary Rust CLI for working with JSONL datasets in
**Alpaca**, **ShareGPT**, and **ChatML** formats. No Python, no GPU, no LLM
calls — pure data transformation with a live TUI dashboard.

```
 CPU   2.3%  │  RAM 4,821/16,384 MB  20%  │  tokenmill  Streaming Pipeline
┌[1] MONITOR──────────────────────────────────────────┐┌[2] ACTIVITY──┐
│ TASK METRICS          │ THROUGHPUT (lines/tick)      ││ KEYBINDINGS  │
│  LINES    1,240,000   │ ▁▂▃▄▅▆▇█▇▆▅▄▃▄▅▆▇           ││              │
│  LINES/s  82,400      │                              ││ ...          │
│  BYTES    1.2 GB      │                              │└──────────────┘
│  BYTES/s  98.1 MB/s   │                              │
│  ETA      3s          │                              │
│  ELAPSED  12s         │                              │
│  ERRORS   0           │                              │
├───────────────────────────────────────────────────── ┤
│ RECENT ACTIVITY                                      │
│ [12.1s] 1,240,000 lines · 1.2 GB                     │
│ [8.3s]  820,000 lines · 798.4 MB                     │
├──────────────────────────────────────────────────────┤
│ PROGRESS  ████████████████████░░░░░░░░   74%         │
└──────────────────────────────────────────────────────┘
  RUNNING   │  Collecting stats...  q Quit  Tab Tabs  ↑↓ Scroll  │  12s elapsed
```

---

## Install

```bash
cargo install --path crates/tokenmill-cli
```

Or build from source:

```bash
git clone https://github.com/fr3on/tokenmill
cd tokenmill
cargo build --release
# binary at: target/release/tokenmill-cli
```

---

## Commands

### `validate` — check schema conformance

```bash
# Validate and print error lines to stderr
tokenmill-cli validate --schema alpaca data.jsonl

# Stop on first error
tokenmill-cli validate --schema alpaca --strict data.jsonl

# Write only valid lines to a new file
tokenmill-cli validate --schema chatml -o valid.jsonl data.jsonl
```

Supported schemas: `alpaca` `sharegpt` `chatml`

---

### `stats` — token & length distribution

```bash
# Auto-detect format, print JSON summary
tokenmill-cli stats data.jsonl

# Include percentiles (p25/p50/p75/p90/p99)
tokenmill-cli stats --percentiles data.jsonl

# Use a different tokeniser
tokenmill-cli stats --tokenizer cl100k_base data.jsonl

# Live TUI dashboard — summary printed to stdout after you press q
tokenmill-cli stats --tui data.jsonl
```

Example output:

```json
{
  "total_lines": 100000,
  "valid_lines": 99843,
  "invalid_lines": 157,
  "total_tokens": 48200341,
  "mean_tokens": 482.7,
  "min_tokens": 12,
  "max_tokens": 4096
}
```

---

### `filter` — remove low-quality samples

```bash
# Keep samples between 50 and 2048 tokens
tokenmill-cli filter --min-tokens 50 --max-tokens 2048 data.jsonl -o filtered.jsonl

# Remove refusal responses
tokenmill-cli filter --remove-refusals data.jsonl -o filtered.jsonl

# Character bounds
tokenmill-cli filter --min-chars 100 --max-chars 8000 data.jsonl

# Perplexity filter (requires a local Qwen2 model directory)
tokenmill-cli filter --perplexity-threshold 50.0 --model-path ./qwen2 data.jsonl

# Print kept/removed/error counts to stderr
tokenmill-cli filter --min-tokens 10 --stats data.jsonl -o filtered.jsonl
```

---

### `convert` — format conversion

```bash
# Alpaca → ShareGPT
tokenmill-cli convert --from alpaca --to sharegpt data.jsonl -o out.jsonl

# ShareGPT → ChatML
tokenmill-cli convert --from sharegpt --to chatml data.jsonl

# Skip malformed lines instead of aborting
tokenmill-cli convert --from alpaca --to chatml --skip-errors data.jsonl

# Convert from Parquet
tokenmill-cli convert --from parquet --to alpaca data.parquet -o out.jsonl
```

Supported conversion pairs: `alpaca↔sharegpt` `alpaca↔chatml` `sharegpt↔chatml` `parquet→any`

---

### `dedup` — remove duplicates

```bash
# Exact dedup using xxHash-128
tokenmill-cli dedup data.jsonl -o deduped.jsonl

# Near-dedup with MinHash (Jaccard ≥ 0.8)
tokenmill-cli dedup --method minhash --threshold 0.8 data.jsonl -o deduped.jsonl

# Semantic dedup with BERT embeddings (cosine similarity)
tokenmill-cli dedup --method semantic --threshold 0.95 --model-path ./bert data.jsonl

# Deterministic with a fixed seed
tokenmill-cli dedup --method minhash --seed 42 data.jsonl
```

---

### `sample` — reservoir sampling

```bash
# Draw 10 000 samples reproducibly
tokenmill-cli sample --n 10000 --seed 42 data.jsonl -o sample.jsonl

# Pipe to stats
tokenmill-cli sample --n 5000 data.jsonl | tokenmill-cli stats
```

---

### `hf` — Hugging Face Hub

```bash
# List all files in a dataset repo
tokenmill-cli hf list HuggingFaceFW/fineweb
tokenmill-cli hf list bert-base-uncased --type model

# Download a file (cached in ~/.cache/huggingface/hub)
tokenmill-cli hf download HuggingFaceFW/fineweb sample-10BT/000_00000.parquet

# Save to a specific path
tokenmill-cli hf download HuggingFaceFW/fineweb sample-10BT/000_00000.parquet -o ./data.parquet

# Use hf:// URIs directly in other commands
tokenmill-cli stats hf://HuggingFaceFW/fineweb/sample-10BT/000_00000.parquet
```

---

## Global flags

| Flag | Description |
|---|---|
| `--tui` | Live ratatui dashboard instead of inline progress bar |
| `--quiet` | No progress output — stdout only (for scripts and pipes) |

Both flags work with any command:

```bash
tokenmill-cli filter --min-tokens 50 --tui big.jsonl -o out.jsonl
tokenmill-cli dedup --method minhash --quiet huge.jsonl -o deduped.jsonl
```

`--tui` alone opens a standalone system-monitor dashboard:

```bash
tokenmill-cli --tui
```

---

## TUI keybindings

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `Enter` | Exit after task finishes |
| `1` `2` `3` | Switch to Monitor / Activity / Help tab |
| `Tab` / `←` `→` | Cycle tabs |
| `↑` `↓` / `PgUp` `PgDn` | Scroll activity log |
| `Ctrl-C` | Force quit |

---

## Performance

Benchmarked on a 2024 MacBook Pro M3 (single core) against a 1 M line
synthetic Alpaca file (~1.2 GB):

| Command | Throughput |
|---|---|
| `validate` | ~480 MB/s |
| `stats` (no percentiles) | ~380 MB/s |
| `filter` (char bounds only) | ~310 MB/s |
| `filter` (token count) | ~45 MB/s *(tiktoken-rs bound)* |
| `convert` | ~390 MB/s |
| `dedup --method exact` | ~290 MB/s |
| `dedup --method minhash` | ~28 MB/s |
| `sample` | ~410 MB/s |

All commands stream line-by-line and run comfortably on 200 GB files with 2 GB RAM.

---

## Supported formats

| Format | Key field | Canonical text |
|---|---|---|
| **Alpaca** | `instruction` | `instruction + \n + input + \n + output` |
| **ShareGPT** | `conversations` | all `value` fields joined with `\n` |
| **ChatML** | `messages` | all `content` fields joined with `\n` |
| **Parquet** | *(auto)* | schema-mapped to any JSONL format |

Format is auto-detected from the first valid line when not specified.

---

## Building from source

```bash
# Debug build
cargo build

# Release (optimised, ~15 MB binary)
cargo build --release

# Run tests
cargo test

# Lint
cargo clippy -- -D warnings
```
