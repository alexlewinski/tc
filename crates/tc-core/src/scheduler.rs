//! Flow-matching (rectified flow) Euler schedule.

use std::path::Path;

use serde::Deserialize;

/// The subset of a diffusers `FlowMatchEulerDiscreteScheduler` config we honour.
#[derive(Clone, Debug, Deserialize)]
pub struct FlowMatchConfig {
    #[serde(default = "d_1000")]
    pub num_train_timesteps: f64,
    #[serde(default = "d_one")]
    pub shift: f64,
    #[serde(default)]
    pub use_dynamic_shifting: bool,
    #[serde(default = "d_exp")]
    pub time_shift_type: String,
    #[serde(default)]
    pub shift_terminal: Option<f64>,
}

fn d_1000() -> f64 {
    1000.0
}
fn d_one() -> f64 {
    1.0
}
fn d_exp() -> String {
    "exponential".into()
}

impl Default for FlowMatchConfig {
    fn default() -> Self {
        FlowMatchConfig {
            num_train_timesteps: 1000.0,
            shift: 1.0,
            use_dynamic_shifting: true,
            time_shift_type: "exponential".into(),
            shift_terminal: None,
        }
    }
}

impl FlowMatchConfig {
    pub fn load(path: &Path) -> FlowMatchConfig {
        std::fs::read(path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }
}

/// FLUX.2's empirical resolution- and step-dependent shift (`mu`).
pub fn flux2_empirical_mu(image_seq_len: usize, num_steps: usize) -> f64 {
    let (a1, b1) = (8.73809524e-05, 1.89833333);
    let (a2, b2) = (0.00016927, 0.45666666);
    let n = image_seq_len as f64;
    if image_seq_len > 4300 {
        return a2 * n + b2;
    }
    let m200 = a2 * n + b2;
    let m10 = a1 * n + b1;
    let a = (m200 - m10) / 190.0;
    let b = m200 - 200.0 * a;
    a * num_steps as f64 + b
}

/// How to shift the base `linspace(1, 1/n, n)` sigmas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shift {
    /// Resolution dependent shift: `exp(mu) / (exp(mu) + (1/s - 1))`.
    Mu(f64),
    /// Constant shift: `k*s / (1 + (k-1)*s)`.
    Constant(f64),
}

/// Sigmas (length `steps + 1`, ending with 0) for a flow-matching Euler sampler.
pub fn sigmas(steps: usize, shift: Shift, config: &FlowMatchConfig) -> Vec<f64> {
    let steps = steps.max(1);
    let mut s: Vec<f64> = (0..steps)
        .map(|i| {
            // np.linspace(1.0, 1/steps, steps) in float32
            let v = if steps == 1 { 1.0 } else { 1.0 + (1.0 / steps as f64 - 1.0) * i as f64 / (steps - 1) as f64 };
            v as f32 as f64
        })
        .collect();
    for v in s.iter_mut() {
        *v = match shift {
            Shift::Mu(mu) => {
                if config.time_shift_type == "linear" {
                    mu / (mu + (1.0 / *v - 1.0))
                } else {
                    mu.exp() / (mu.exp() + (1.0 / *v - 1.0))
                }
            }
            Shift::Constant(k) => k * *v / (1.0 + (k - 1.0) * *v),
        };
    }
    if let Some(terminal) = config.shift_terminal {
        let last = *s.last().unwrap();
        let one_minus_z: Vec<f64> = s.iter().map(|v| 1.0 - v).collect();
        let scale = (1.0 - last) / (1.0 - terminal);
        for (v, o) in s.iter_mut().zip(one_minus_z) {
            *v = 1.0 - o / scale;
        }
    }
    let mut s: Vec<f64> = s.into_iter().map(|v| v as f32 as f64).collect();
    s.push(0.0);
    s
}

/// For img2img with `strength` < 1: index of the first step to run (diffusers semantics).
pub fn img2img_start(steps: usize, strength: f32) -> usize {
    let init = (steps as f64 * strength.clamp(0.0, 1.0) as f64).min(steps as f64);
    (steps as f64 - init).max(0.0) as usize
}
