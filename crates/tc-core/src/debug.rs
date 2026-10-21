//! Opt-in numeric diagnostics (`TC_DEBUG_STATS=1`), printed to stderr. They copy tensors to
//! the host, so they are slow and only meant for tracking down overflow or garbage on a device.

use nnc::Var;

pub fn stats_enabled() -> bool {
    std::env::var("TC_DEBUG_STATS").map(|v| v == "1").unwrap_or(false)
}

/// Print max |x|, mean |x|, the number of non-finite values and of values at the f16 limit.
pub fn print_stats(label: &str, x: &Var) {
    let v = x.to_f32_vec();
    let (mut max, mut sum, mut nonfinite, mut saturated) = (0f32, 0f64, 0usize, 0usize);
    for a in v.iter() {
        if !a.is_finite() {
            nonfinite += 1;
            continue;
        }
        let a = a.abs();
        max = max.max(a);
        sum += a as f64;
        if a >= 65000.0 {
            saturated += 1;
        }
    }
    let mean = sum / v.len().max(1) as f64;
    eprintln!(
        "[stats] {label:<18} {:?} {:?} max|x| {max:<11.4} mean|x| {mean:<10.5} non-finite {nonfinite} at-f16-limit {saturated}",
        x.shape(),
        x.dtype()
    );
}
