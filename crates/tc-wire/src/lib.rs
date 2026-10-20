//! Tensor wire format (images, masks and hints in the gRPC API).
//!
//! Layout (little endian):
//!
//! ```text
//! u32  codec identifier: 0 = raw, 0x217 = raw deflate, 0xf7217 = fpzip
//! 64B  tensor parameters: i32 type, i32 format, i32 datatype, i32 reserved, i32 dim[12]
//! ...  payload
//! ```
//!
//! fpzip payloads hold floats (f16 tensors are widened to f32 with 19 bits of precision).

use anyhow::{anyhow, bail, Result};
use half::f16;
use std::io::{Read, Write};

pub const CODEC_RAW: u32 = 0;
pub const CODEC_ZIP: u32 = 0x217;
pub const CODEC_FPZIP: u32 = 0xf7217;

pub const TYPE_CPU: i32 = 1;
pub const FORMAT_NCHW: i32 = 1;
pub const FORMAT_NHWC: i32 = 2;

/// Element types (nnc datatype codes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireType {
    U8,
    I32,
    F32,
    I64,
    F64,
    F16,
    BF16,
}

impl WireType {
    pub fn code(self) -> i32 {
        match self {
            WireType::U8 => 0x01000,
            WireType::I32 => 0x02000,
            WireType::F32 => 0x04000,
            WireType::I64 => 0x08000,
            WireType::F64 => 0x10000,
            WireType::F16 => 0x20000,
            WireType::BF16 => 0x80000,
        }
    }
    pub fn from_code(c: i32) -> Option<WireType> {
        Some(match c & 0xFF000 {
            0x01000 => WireType::U8,
            0x02000 => WireType::I32,
            0x04000 => WireType::F32,
            0x08000 => WireType::I64,
            0x10000 => WireType::F64,
            0x20000 => WireType::F16,
            0x80000 => WireType::BF16,
            _ => return None,
        })
    }
    pub fn size(self) -> usize {
        match self {
            WireType::U8 => 1,
            WireType::F16 | WireType::BF16 => 2,
            WireType::I32 | WireType::F32 => 4,
            WireType::I64 | WireType::F64 => 8,
        }
    }
}

/// A decoded CPU tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct WireTensor {
    pub dtype: WireType,
    pub format: i32,
    pub shape: Vec<usize>,
    /// Contiguous little-endian elements.
    pub data: Vec<u8>,
}

impl WireTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Elements as f32 (floating point types only, U8 as 0..255).
    pub fn to_f32(&self) -> Vec<f32> {
        match self.dtype {
            WireType::F32 => bytemuck::pod_collect_to_vec(&self.data),
            WireType::F16 => self.data.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            WireType::BF16 => self.data.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            WireType::F64 => self.data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            WireType::U8 => self.data.iter().map(|v| *v as f32).collect(),
            WireType::I32 => self.data.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            WireType::I64 => self.data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        }
    }

    pub fn from_f32(values: &[f32], shape: &[usize], dtype: WireType) -> WireTensor {
        let data: Vec<u8> = match dtype {
            WireType::F32 => bytemuck::cast_slice(values).to_vec(),
            WireType::F16 => values.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect(),
            WireType::BF16 => values.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            WireType::F64 => values.iter().flat_map(|v| (*v as f64).to_le_bytes()).collect(),
            WireType::U8 => values.iter().map(|v| v.clamp(0.0, 255.0) as u8).collect(),
            WireType::I32 => values.iter().flat_map(|v| (*v as i32).to_le_bytes()).collect(),
            WireType::I64 => values.iter().flat_map(|v| (*v as i64).to_le_bytes()).collect(),
        };
        WireTensor { dtype, format: FORMAT_NHWC, shape: shape.to_vec(), data }
    }
}

const HEADER: usize = 4 + 64;

fn header(t: &WireTensor, codec: u32) -> Vec<u8> {
    let mut h = Vec::with_capacity(HEADER);
    h.extend_from_slice(&codec.to_le_bytes());
    for v in [TYPE_CPU, t.format, t.dtype.code(), 0] {
        h.extend_from_slice(&v.to_le_bytes());
    }
    for i in 0..12 {
        let d = t.shape.get(i).copied().unwrap_or(0) as i32;
        h.extend_from_slice(&d.to_le_bytes());
    }
    h
}

/// Which compression to use when encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compression {
    None,
    /// fpzip for floating point tensors, raw deflate otherwise.
    FpzipOrZip,
    Zip,
}

pub fn encode(t: &WireTensor, compression: Compression) -> Vec<u8> {
    let compressed = match compression {
        Compression::None => None,
        Compression::Zip => zip(&t.data).map(|d| (CODEC_ZIP, d)),
        Compression::FpzipOrZip => match t.dtype {
            WireType::F16 | WireType::F32 | WireType::F64 => fpzip_encode(t).map(|d| (CODEC_FPZIP, d)),
            _ => zip(&t.data).map(|d| (CODEC_ZIP, d)),
        },
    };
    match compressed {
        Some((codec, payload)) if payload.len() <= t.data.len() => {
            let mut out = header(t, codec);
            out.extend_from_slice(&payload);
            out
        }
        _ => {
            let mut out = header(t, CODEC_RAW);
            out.extend_from_slice(&t.data);
            out
        }
    }
}

pub fn decode(bytes: &[u8]) -> Result<WireTensor> {
    if bytes.len() < HEADER {
        bail!("tensor data too short ({} bytes)", bytes.len());
    }
    let rd = |o: usize| i32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let codec = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let format = rd(8);
    let dtype = WireType::from_code(rd(12)).ok_or_else(|| anyhow!("unsupported tensor datatype {:#x}", rd(12)))?;
    let shape: Vec<usize> = (0..12).map(|i| rd(20 + 4 * i)).take_while(|d| *d > 0).map(|d| d as usize).collect();
    if shape.is_empty() {
        bail!("tensor has no dimensions");
    }
    let data = decode_payload(codec, &bytes[HEADER..], dtype, &shape)?;
    Ok(WireTensor { dtype, format, shape, data })
}

/// Decode a codec payload (raw, raw deflate or fpzip) into `shape` elements of `dtype`. Also
/// used for tensors stored in nnc SQLite checkpoints, which use the same codec identifiers.
pub fn decode_payload(codec: u32, payload: &[u8], dtype: WireType, shape: &[usize]) -> Result<Vec<u8>> {
    let n: usize = shape.iter().product();
    let size = n.checked_mul(dtype.size()).ok_or_else(|| anyhow!("tensor too large"))?;
    if size > 1 << 34 {
        bail!("tensor too large");
    }
    Ok(match codec {
        CODEC_RAW => {
            if payload.len() < size {
                bail!("tensor payload truncated");
            }
            payload[..size].to_vec()
        }
        CODEC_ZIP => {
            let mut out = Vec::with_capacity(size);
            flate2::read::DeflateDecoder::new(payload).take(size as u64).read_to_end(&mut out)?;
            if out.len() != size {
                bail!("deflate payload has {} bytes, expected {size}", out.len());
            }
            out
        }
        CODEC_FPZIP => fpzip_decode(payload, dtype, shape)?,
        other => bail!("unsupported tensor codec {other:#x}"),
    })
}

fn zip(data: &[u8]) -> Option<Vec<u8>> {
    let mut e = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
    e.write_all(data).ok()?;
    e.finish().ok()
}

#[repr(C)]
struct Fpz {
    type_: i32,
    prec: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    nf: i32,
}

extern "C" {
    fn fpzip_write_to_buffer(buffer: *mut std::ffi::c_void, size: usize) -> *mut Fpz;
    fn fpzip_write_header(fpz: *mut Fpz) -> i32;
    fn fpzip_write(fpz: *mut Fpz, data: *const std::ffi::c_void) -> usize;
    fn fpzip_write_close(fpz: *mut Fpz);
    fn fpzip_read_from_buffer(buffer: *const std::ffi::c_void) -> *mut Fpz;
    fn fpzip_read_header(fpz: *mut Fpz) -> i32;
    fn fpzip_read(fpz: *mut Fpz, data: *mut std::ffi::c_void) -> usize;
    fn fpzip_read_close(fpz: *mut Fpz);
}

fn fpzip_dims(shape: &[usize]) -> (i32, i32, i32, i32) {
    let n = shape.len();
    let nx = shape[n - 1] as i32;
    let ny = if n >= 2 { shape[n - 2] as i32 } else { 1 };
    let nz = if n >= 3 { shape[n - 3] as i32 } else { 1 };
    let nf = if n >= 4 { shape[..n - 3].iter().product::<usize>() as i32 } else { 1 };
    (nx, ny, nz, nf)
}

fn fpzip_encode(t: &WireTensor) -> Option<Vec<u8>> {
    let (type_, prec, input): (i32, i32, Vec<u8>) = match t.dtype {
        WireType::F64 => (1, 64, t.data.clone()),
        WireType::F32 => (0, 32, t.data.clone()),
        WireType::F16 => (0, 19, bytemuck::cast_slice(&t.to_f32()).to_vec()),
        _ => return None,
    };
    let cap = t.data.len() + 1024;
    let mut buf = vec![0u8; cap];
    unsafe {
        let fpz = fpzip_write_to_buffer(buf.as_mut_ptr() as *mut _, cap);
        if fpz.is_null() {
            return None;
        }
        let (nx, ny, nz, nf) = fpzip_dims(&t.shape);
        (*fpz).type_ = type_;
        (*fpz).prec = prec;
        (*fpz).nx = nx;
        (*fpz).ny = ny;
        (*fpz).nz = nz;
        (*fpz).nf = nf;
        let ok = fpzip_write_header(fpz) != 0;
        let n = if ok { fpzip_write(fpz, input.as_ptr() as *const _) } else { 0 };
        fpzip_write_close(fpz);
        if n == 0 {
            return None;
        }
        buf.truncate(n);
    }
    Some(buf)
}

fn fpzip_decode(payload: &[u8], dtype: WireType, shape: &[usize]) -> Result<Vec<u8>> {
    let n: usize = shape.iter().product();
    unsafe {
        let fpz = fpzip_read_from_buffer(payload.as_ptr() as *const _);
        if fpz.is_null() {
            bail!("invalid fpzip stream");
        }
        if fpzip_read_header(fpz) == 0 {
            fpzip_read_close(fpz);
            bail!("invalid fpzip header");
        }
        let f = &*fpz;
        let len = f.nx as usize * f.ny as usize * f.nz as usize * f.nf as usize;
        let is_double = f.type_ == 1;
        let elem = if is_double { 8 } else { 4 };
        if len < n || len > n.max(1).saturating_mul(64) {
            fpzip_read_close(fpz);
            bail!("fpzip stream does not match tensor shape");
        }
        let mut out = vec![0u8; len * elem];
        let r = fpzip_read(fpz, out.as_mut_ptr() as *mut _);
        fpzip_read_close(fpz);
        if r == 0 {
            bail!("fpzip decode failed");
        }
        out.truncate(n * elem);
        Ok(match (dtype, is_double) {
            (WireType::F64, true) | (WireType::F32, false) => out,
            (WireType::F16, false) => {
                let f: Vec<f32> = bytemuck::pod_collect_to_vec(&out);
                f.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect()
            }
            _ => bail!("fpzip payload type does not match tensor datatype"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips() {
        let vals: Vec<f32> = (0..2 * 16 * 24 * 3).map(|i| ((i as f32) * 0.37).sin()).collect();
        for dtype in [WireType::F32, WireType::F16] {
            let t = WireTensor::from_f32(&vals, &[2, 16, 24, 3], dtype);
            for c in [Compression::None, Compression::Zip, Compression::FpzipOrZip] {
                let enc = encode(&t, c);
                let dec = decode(&enc).unwrap();
                assert_eq!(dec.shape, t.shape);
                assert_eq!(dec.dtype, t.dtype);
                assert_eq!(dec.data, t.data, "{dtype:?} {c:?} lossless");
            }
        }
        let mask = WireTensor { dtype: WireType::U8, format: FORMAT_NHWC, shape: vec![8, 8], data: vec![3; 64] };
        let enc = encode(&mask, Compression::FpzipOrZip);
        assert_eq!(u32::from_le_bytes(enc[0..4].try_into().unwrap()), CODEC_ZIP);
        assert_eq!(decode(&enc).unwrap(), mask);
        assert!(decode(&[0; 10]).is_err());
    }
}
