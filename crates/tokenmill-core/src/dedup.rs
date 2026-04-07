use std::io::{BufRead, Write};
use std::collections::{HashMap, HashSet};
use crate::{Alpaca, ChatML, Format, ShareGPT, detect, xxh3_128, inference::Embedder};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use rand::Rng;

pub struct MinHashDedupOptions {
    pub threshold: f64,
    pub seed: u64,
}

pub struct SemanticDedupOptions {
    pub threshold: f64,
    pub model_path: String,
}

pub fn exact_dedup<R: BufRead, W: Write>(
    reader: R,
    mut writer: W,
    _skip_errors: bool,
    mut progress: Option<&mut dyn FnMut(u64)>,
) -> anyhow::Result<u64> {
    let mut seen = HashSet::new();
    let mut kept = 0;

    for line_result in reader.lines() {
        let line = line_result?;

        if let Some(cb) = progress.as_mut() {
            cb(line.len() as u64 + 1);
        }
        
        let format = match detect(&line) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let canonical_text = match format {
            Format::Alpaca => serde_json::from_str::<Alpaca>(&line)?.canonical_text(),
            Format::ShareGPT => serde_json::from_str::<ShareGPT>(&line)?.canonical_text(),
            Format::ChatML => serde_json::from_str::<ChatML>(&line)?.canonical_text(),
            Format::Parquet | Format::Arrow => anyhow::bail!("Parquet/Arrow not supported directly in dedup"),
        };

        let hash = xxh3_128(canonical_text.as_bytes());

        if seen.insert(hash) {
            kept += 1;
            writeln!(writer, "{}", line)?;
        }
    }

    Ok(kept)
}

pub fn minhash_dedup<R: BufRead, W: Write>(
    reader: R,
    mut writer: W,
    options: MinHashDedupOptions,
    _skip_errors: bool,
    mut progress: Option<&mut dyn FnMut(u64)>,
) -> anyhow::Result<u64> {
    let mut signatures = Vec::new();
    let mut lines = Vec::new();
    
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(options.seed);
    let mut a = [0u64; NUM_HASHES];
    let mut b = [0u64; NUM_HASHES];
    for i in 0..NUM_HASHES {
        a[i] = rng.gen_range(1..PRIME);
        b[i] = rng.gen_range(0..PRIME);
    }

    for line_result in reader.lines() {
        let line = line_result?;
        if let Some(cb) = progress.as_mut() {
            cb(line.len() as u64 + 1);
        }
        let format = match detect(&line) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let text = match format {
            Format::Alpaca => serde_json::from_str::<Alpaca>(&line)?.canonical_text(),
            Format::ShareGPT => serde_json::from_str::<ShareGPT>(&line)?.canonical_text(),
            Format::ChatML => serde_json::from_str::<ChatML>(&line)?.canonical_text(),
            Format::Parquet | Format::Arrow => anyhow::bail!("Parquet/Arrow not supported directly in dedup"),
        };

        let sig = compute_signature(&text, &a, &b);
        signatures.push(sig);
        lines.push(line);
    }

    if lines.is_empty() {
        return Ok(0);
    }

    let num_bands = 8;
    let rows_per_band = NUM_HASHES / num_bands;
    let mut uf = UnionFind::new(lines.len());

    for band_idx in 0..num_bands {
        let mut bands: HashMap<Vec<u32>, Vec<usize>> = HashMap::new();
        for (idx, sig) in signatures.iter().enumerate() {
            let start = band_idx * rows_per_band;
            let end = start + rows_per_band;
            let band_key = sig[start..end].to_vec();
            bands.entry(band_key).or_default().push(idx);
        }

        for candidates in bands.values() {
            if candidates.len() > 1 {
                for i in 0..candidates.len() - 1 {
                    for j in i + 1..candidates.len() {
                        let idx1 = candidates[i];
                        let idx2 = candidates[j];
                        
                        let intersection = signatures[idx1]
                            .iter()
                            .zip(signatures[idx2].iter())
                            .filter(|(a, b)| a == b)
                            .count();
                        let similarity = intersection as f64 / NUM_HASHES as f64;
                        if similarity >= options.threshold {
                            uf.union(idx1, idx2);
                        }
                    }
                }
            }
        }
    }

    let mut kept = 0;
    let mut seen_roots = HashSet::new();
    for (idx, line) in lines.iter().enumerate() {
        let root = uf.find(idx);
        if seen_roots.insert(root) {
            kept += 1;
            writeln!(writer, "{}", line)?;
        }
    }

    Ok(kept)
}

pub fn semantic_dedup<R: BufRead, W: Write>(
    reader: R,
    mut writer: W,
    options: SemanticDedupOptions,
    mut progress: Option<&mut dyn FnMut(u64)>,
) -> anyhow::Result<u64> {
    let embedder = Embedder::load_bert(std::path::Path::new(&options.model_path))?;
    
    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    
    for line_result in reader.lines() {
        let line = line_result?;
        if let Some(cb) = progress.as_mut() {
            cb(line.len() as u64 + 1);
        }
        
        let format = match detect(&line) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let text = match format {
            Format::Alpaca => serde_json::from_str::<Alpaca>(&line)?.canonical_text(),
            Format::ShareGPT => serde_json::from_str::<ShareGPT>(&line)?.canonical_text(),
            Format::ChatML => serde_json::from_str::<ChatML>(&line)?.canonical_text(),
            Format::Parquet | Format::Arrow => anyhow::bail!("Parquet/Arrow not supported directly in dedup"),
        };

        let emb = embedder.embed(&text)?;
        
        let mut is_duplicate = false;
        // Optimization: only compare against the last 5000 items to keep it semi-fast and low-memory
        let search_start = embeddings.len().saturating_sub(5000);
        for existing in &embeddings[search_start..] {
            let sim = cosine_similarity(existing, &emb);
            if sim >= options.threshold as f32 {
                is_duplicate = true;
                break;
            }
        }
        
        if !is_duplicate {
            embeddings.push(emb);
            writeln!(writer, "{}", line)?;
            lines.push(line);
        }
    }
    
    Ok(lines.len() as u64)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

const NUM_HASHES: usize = 128;
const PRIME: u64 = 2147483647;

fn compute_signature(text: &str, a: &[u64], b: &[u64]) -> [u32; NUM_HASHES] {
    let mut sig = [u32::MAX; NUM_HASHES];
    let shingle_size = 5;
    
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < shingle_size {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let h = hasher.finish();
        for i in 0..NUM_HASHES {
            sig[i] = (((a[i] as u128 * h as u128) + b[i] as u128) % PRIME as u128) as u32;
        }
        return sig;
    }

    for window in chars.windows(shingle_size) {
        let shingle: String = window.iter().collect();
        let mut hasher = DefaultHasher::new();
        shingle.hash(&mut hasher);
        let h = hasher.finish();

        for i in 0..NUM_HASHES {
            let permuted = (((a[i] as u128 * h as u128) + b[i] as u128) % PRIME as u128) as u32;
            if permuted < sig[i] {
                sig[i] = permuted;
            }
        }
    }
    sig
}

pub struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
        }
    }

    pub fn find(&mut self, i: usize) -> usize {
        if self.parent[i] == i {
            i
        } else {
            self.parent[i] = self.find(self.parent[i]);
            self.parent[i]
        }
    }

    pub fn union(&mut self, i: usize, j: usize) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            if root_i < root_j {
                self.parent[root_j] = root_i;
            } else {
                self.parent[root_i] = root_j;
            }
        }
    }
}
