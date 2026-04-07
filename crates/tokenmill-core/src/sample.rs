use std::io::{BufRead, Write};
use rand::Rng;
use rand::seq::SliceRandom;
use crate::Result;

pub fn sample<R: BufRead, W: Write>(
    reader: R,
    mut writer: W,
    n: usize,
    seed: u64,
    mut progress: Option<&mut dyn FnMut(u64)>,
) -> Result<u64> {
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut reservoir = Vec::with_capacity(n);
    let mut total_count: u64 = 0;

    for line_result in reader.lines() {
        let line = line_result?;
        total_count += 1;

        if let Some(cb) = progress.as_mut() {
            cb(line.len() as u64 + 1);
        }

        if reservoir.len() < n {
            reservoir.push(line);
        } else {
            let j = rng.gen_range(0u64..total_count);
            if j < n as u64 {
                reservoir[j as usize] = line;
            }
        }
    }

    // Shuffle once more to ensure random order
    reservoir.shuffle(&mut rng);

    for line in reservoir {
        writeln!(writer, "{}", line)?;
    }

    Ok(total_count)
}
