//! Deterministic noise.
//!
//! [`TorchCpuRng`] reproduces the values of `torch.randn(..., generator=torch.Generator().manual_seed(seed))`
//! on CPU for float32 tensors (MT19937 + PyTorch's vectorized Box-Muller fill), so a seed gives
//! the same initial noise as reference Python pipelines.

/// MT19937 32-bit Mersenne Twister.
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Mt19937 {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1812433253u32.wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { state, index: 624 }
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
            let mut v = self.state[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                v ^= 0x9908_b0df;
            }
            self.state[i] = v;
        }
        self.index = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }
}

pub struct TorchCpuRng {
    mt: Mt19937,
    cached_normal: Option<f64>,
}

impl TorchCpuRng {
    pub fn new(seed: u64) -> TorchCpuRng {
        TorchCpuRng { mt: Mt19937::new(seed as u32), cached_normal: None }
    }

    fn uniform_f32(&mut self) -> f32 {
        let x = self.mt.next_u32();
        ((x & ((1 << 24) - 1)) as f32) * (1.0 / (1u32 << 24) as f32)
    }

    fn normal_f64(&mut self) -> f64 {
        // Small tensors use a scalar Box-Muller on two 53-bit uniforms; the second value of
        // each pair is cached for the next call.
        if let Some(v) = self.cached_normal.take() {
            return v;
        }
        let u1 = self.uniform_f64();
        let u2 = self.uniform_f64();
        let r = (-2.0 * (1.0 - u2).ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u1;
        self.cached_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    fn uniform_f64(&mut self) -> f64 {
        let hi = self.mt.next_u32() as u64;
        let lo = self.mt.next_u32() as u64;
        let x = (hi << 32) | lo;
        ((x & ((1u64 << 53) - 1)) as f64) * (1.0 / (1u64 << 53) as f64)
    }

    fn fill16(data: &mut [f32]) {
        for j in 0..8 {
            let u1 = 1.0 - data[j];
            let u2 = data[j + 8];
            let radius = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            data[j] = radius * theta.cos();
            data[j + 8] = radius * theta.sin();
        }
    }

    /// Standard normal samples, in the same order PyTorch fills a contiguous tensor.
    pub fn randn(&mut self, n: usize) -> Vec<f32> {
        let mut data = vec![0f32; n];
        if n >= 16 {
            for v in data.iter_mut() {
                *v = self.uniform_f32();
            }
            let mut i = 0;
            while i + 16 <= n {
                Self::fill16(&mut data[i..i + 16]);
                i += 16;
            }
            if n % 16 != 0 {
                // Recompute the last 16 values.
                let start = n - 16;
                for v in data[start..].iter_mut() {
                    *v = self.uniform_f32();
                }
                Self::fill16(&mut data[start..]);
            }
        } else {
            for v in data.iter_mut() {
                *v = self.normal_f64() as f32;
            }
        }
        data
    }
}
