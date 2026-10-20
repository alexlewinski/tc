//! Palettized ("q8p"-style) weight quantization: per block of `block` elements, a palette of
//! `2^bits` centroids (k-means) plus `bits`-bit indices. On Metal, GEMM consumes such weights
//! directly; on other devices they are expanded back to floats.

use crate::graph::{Graph, Var};
use crate::host::HostTensor;
use crate::sys;
use crate::types::{params, DataType, Device, Format};

/// Default palette block size.
pub const DEFAULT_BLOCK: usize = 16384;

fn palettized_size(numel: usize, dtype: DataType, bits: u32, block: usize) -> usize {
    let blocks = numel.div_ceil(block);
    (1usize << bits) * dtype.size() * blocks + (numel * bits as usize).div_ceil(8)
}

/// Palettize a host tensor (f16, bf16, f32 or f64).
pub fn palettize(t: &HostTensor, bits: u32, block: usize) -> Vec<u8> {
    assert!((4..=8).contains(&bits));
    let n = t.numel();
    let size = palettized_size(n, t.dtype(), bits, block);
    let mut out = vec![0u8; size];
    let written = unsafe {
        sys::ccv_nnc_palettize(
            t.bytes().as_ptr() as *const _,
            t.dtype().to_raw(),
            sys::CCV_TENSOR_CPU_MEMORY as i32,
            n,
            bits as i32,
            block as i32,
            out.as_mut_ptr() as *mut _,
            size,
        )
    };
    debug_assert!(written <= size);
    let _ = written;
    out
}

/// Fast 8-bit palettization (same layout as [`palettize`]): per block, 256 centroids from
/// value quantiles refined by a few Lloyd iterations. Much faster than optimal 1-D k-means
/// with nearly the same error, so it can run at model load time.
pub fn palettize8_fast(t: &HostTensor, block: usize) -> Vec<u8> {
    use rayon::prelude::*;
    let values = t.to_f32_vec();
    let n = values.len();
    let esize = t.dtype().size();
    let dtype = t.dtype();
    let blocks = n.div_ceil(block);
    let mut out = vec![0u8; palettized_size(n, dtype, 8, block)];
    out.par_chunks_mut(256 * esize + block).enumerate().for_each(|(bi, dst)| {
        let start = bi * block;
        let v = &values[start..(start + block).min(n)];
        let (palette, idx) = palette8(v);
        let pal_bytes = HostTensor::from_f32(&palette, &[256]).to_dtype(dtype);
        dst[..256 * esize].copy_from_slice(pal_bytes.bytes());
        dst[256 * esize..256 * esize + idx.len()].copy_from_slice(&idx);
    });
    debug_assert_eq!(out.len(), 256 * esize * blocks + n);
    out
}

fn palette8(v: &[f32]) -> (Vec<f32>, Vec<u8>) {
    let m = v.len();
    let mut sorted: Vec<f32> = v.iter().map(|x| if x.is_finite() { *x } else { 0.0 }).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = 256;
    let mut c: Vec<f32> = (0..k).map(|i| sorted[(((i as f64 + 0.5) * m as f64 / k as f64) as usize).min(m - 1)]).collect();
    for _ in 0..4 {
        // Assign sorted values to centroids by sweeping the midpoints.
        let mut sum = vec![0f64; k];
        let mut cnt = vec![0u32; k];
        let mut j = 0;
        for x in &sorted {
            while j + 1 < k && *x > 0.5 * (c[j] + c[j + 1]) {
                j += 1;
            }
            sum[j] += *x as f64;
            cnt[j] += 1;
        }
        for i in 0..k {
            if cnt[i] > 0 {
                c[i] = (sum[i] / cnt[i] as f64) as f32;
            }
        }
        // Keep centroids sorted (they stay sorted in exact arithmetic).
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    let mids: Vec<f32> = (0..k - 1).map(|i| 0.5 * (c[i] + c[i + 1])).collect();
    let idx = v
        .iter()
        .map(|x| {
            let x = if x.is_finite() { *x } else { 0.0 };
            mids.partition_point(|m| *m < x) as u8
        })
        .collect();
    (c, idx)
}

/// Expand palettized data back to a host tensor of `dtype` with `shape`.
pub fn depalettize(data: &[u8], dtype: DataType, shape: &[usize], bits: u32, block: usize) -> HostTensor {
    let n: usize = shape.iter().product();
    let mut out = vec![0u8; n * dtype.size()];
    unsafe {
        sys::ccv_nnc_depalettize(
            data.as_ptr() as *const _,
            dtype.to_raw(),
            sys::CCV_TENSOR_CPU_MEMORY as i32,
            data.len(),
            bits as i32,
            block as i32,
            out.as_mut_ptr() as *mut _,
            n,
        )
    };
    HostTensor::from_bytes(out, dtype, shape)
}

/// nnc datatype code of a palettized tensor whose values are `dtype`.
fn qx_datatype(dtype: DataType, bits: u32) -> i32 {
    (sys::CCV_QX as i32) | ((bits as i32) << 8) | (dtype.to_raw() >> 12)
}

impl Graph {
    /// Upload palettized weights as a quantized tensor on `device` (Metal GEMM consumes these
    /// directly).
    pub fn from_palettized(&self, data: &[u8], dtype: DataType, shape: &[usize], bits: u32, block: usize, device: Device) -> Var {
        let mut p = params(Device::Cpu, Format::Nhwc, dtype, shape);
        p.datatype = qx_datatype(dtype, bits);
        p.reserved = block as i32;
        let cpu = self.new_with_params(p);
        unsafe {
            let t = sys::ccv_nnc_tensor_from_variable_impl(self.inner.raw, cpu.raw(), std::ptr::null_mut());
            assert!(!t.is_null());
            let len = sys_data_size(&p).min(data.len());
            std::ptr::copy_nonoverlapping(data.as_ptr(), (*t).data.ptr as *mut u8, len);
        }
        if device == Device::Cpu {
            return cpu;
        }
        let mut gp = p;
        gp.type_ = device.to_raw_type();
        let gpu = self.new_with_params(gp);
        crate::ops::data_transfer(&cpu, &gpu);
        // Wait for the copy and let the host staging copy go: keeping it alive with the weight
        // would hold every palettized weight twice in (unified) memory.
        self.synchronize();
        gpu
    }
}

fn sys_data_size(p: &sys::ccv_nnc_tensor_param_t) -> usize {
    let n: usize = p.dim.iter().take_while(|d| **d > 0).map(|d| *d as usize).product();
    let bits = ((p.datatype & 0xf00) >> 8) as u32;
    let dtype = DataType::from_raw((p.datatype & 0xff) << 12).unwrap();
    palettized_size(n, dtype, bits, p.reserved as usize)
}
