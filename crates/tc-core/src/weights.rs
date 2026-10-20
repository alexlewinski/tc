//! Memory-mapped safetensors reading and device parameter stores.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use nnc::{DataType, Device, Graph, HostTensor, Var};
use rayon::prelude::*;

/// Element types that can appear in safetensors files we read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StDtype {
    F64,
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    I64,
    I32,
    U8,
    Bool,
}

impl StDtype {
    fn parse(s: &str) -> Result<StDtype> {
        Ok(match s {
            "F64" => StDtype::F64,
            "F32" => StDtype::F32,
            "F16" => StDtype::F16,
            "BF16" => StDtype::BF16,
            "F8_E4M3" => StDtype::F8E4M3,
            "F8_E5M2" => StDtype::F8E5M2,
            "I64" => StDtype::I64,
            "I32" => StDtype::I32,
            "U8" => StDtype::U8,
            "BOOL" => StDtype::Bool,
            other => bail!("unsupported safetensors dtype {other}"),
        })
    }

    pub fn size(self) -> usize {
        match self {
            StDtype::F64 | StDtype::I64 => 8,
            StDtype::F32 | StDtype::I32 => 4,
            StDtype::F16 | StDtype::BF16 => 2,
            StDtype::F8E4M3 | StDtype::F8E5M2 | StDtype::U8 | StDtype::Bool => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: StDtype,
    pub shape: Vec<usize>,
    file: usize,
    start: usize,
    end: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A set of (possibly sharded) safetensors files, memory mapped.
pub struct SafeTensorsFiles {
    paths: Vec<PathBuf>,
    maps: Vec<Mmap>,
    index: BTreeMap<String, TensorInfo>,
    /// `__metadata__` of every file merged.
    pub metadata: HashMap<String, String>,
}

impl SafeTensorsFiles {
    pub fn open<P: AsRef<Path>>(paths: &[P]) -> Result<SafeTensorsFiles> {
        let mut out = SafeTensorsFiles {
            paths: vec![],
            maps: vec![],
            index: BTreeMap::new(),
            metadata: HashMap::new(),
        };
        for p in paths {
            out.add(p.as_ref())?;
        }
        Ok(out)
    }

    /// Open every `*.safetensors` file in a directory.
    pub fn open_dir(dir: &Path) -> Result<SafeTensorsFiles> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("no .safetensors files in {}", dir.display());
        }
        SafeTensorsFiles::open(&files)
    }

    fn add(&mut self, path: &Path) -> Result<()> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let map = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
        if map.len() < 8 {
            bail!("{} is not a safetensors file", path.display());
        }
        let header_len = u64::from_le_bytes(map[0..8].try_into().unwrap()) as usize;
        if 8 + header_len > map.len() {
            bail!("{}: corrupt safetensors header", path.display());
        }
        let header: serde_json::Value = serde_json::from_slice(&map[8..8 + header_len])
            .with_context(|| format!("{}: parsing safetensors header", path.display()))?;
        let obj = header.as_object().ok_or_else(|| anyhow!("header is not an object"))?;
        let base = 8 + header_len;
        let file_idx = self.maps.len();
        for (name, v) in obj {
            if name == "__metadata__" {
                if let Some(m) = v.as_object() {
                    for (k, v) in m {
                        if let Some(s) = v.as_str() {
                            self.metadata.insert(k.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = StDtype::parse(v["dtype"].as_str().unwrap_or(""))?;
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .ok_or_else(|| anyhow!("{name}: missing shape"))?
                .iter()
                .map(|d| d.as_u64().unwrap_or(0) as usize)
                .collect();
            let offs = v["data_offsets"].as_array().ok_or_else(|| anyhow!("{name}: missing offsets"))?;
            let start = base + offs[0].as_u64().unwrap() as usize;
            let end = base + offs[1].as_u64().unwrap() as usize;
            if end > map.len() || end - start != shape.iter().product::<usize>() * dtype.size() {
                bail!("{}: tensor {name} has inconsistent offsets", path.display());
            }
            self.index.insert(name.clone(), TensorInfo { dtype, shape, file: file_idx, start, end });
        }
        self.paths.push(path.to_path_buf());
        self.maps.push(map);
        Ok(())
    }

    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name)
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn raw(&self, name: &str) -> Result<(&TensorInfo, &[u8])> {
        let info = self.index.get(name).ok_or_else(|| anyhow!("tensor {name} not found"))?;
        Ok((info, &self.maps[info.file][info.start..info.end]))
    }

    /// Read a tensor converted to f32.
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let (info, bytes) = self.raw(name)?;
        Ok(convert_to_f32(info.dtype, bytes))
    }

    /// Read a tensor as a host tensor of the requested floating point type.
    pub fn read(&self, name: &str, dtype: DataType) -> Result<HostTensor> {
        let (info, bytes) = self.raw(name)?;
        Ok(convert_bytes(info.dtype, bytes, dtype, &info.shape))
    }
}

/// Named tensors from any supported container (safetensors files, or a Draw Things checkpoint
/// viewed under canonical names).
pub trait TensorSource: Send + Sync {
    fn names(&self) -> Vec<String>;
    fn shape(&self, name: &str) -> Option<Vec<usize>>;
    /// Read a tensor converted to `dtype`.
    fn read(&self, name: &str, dtype: DataType) -> Result<HostTensor>;
    /// Read a tensor in its stored form (palettized tensors are kept palettized).
    fn read_raw(&self, name: &str) -> Result<crate::ckpt::CkptTensor> {
        Ok(crate::ckpt::CkptTensor::Dense(self.read(name, DataType::F32)?))
    }
    fn contains(&self, name: &str) -> bool {
        self.shape(name).is_some()
    }
    fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self.read(name, DataType::F32)?.to_f32_vec())
    }
}

impl TensorSource for SafeTensorsFiles {
    fn names(&self) -> Vec<String> {
        self.keys().cloned().collect()
    }
    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.info(name).map(|i| i.shape.clone())
    }
    fn read(&self, name: &str, dtype: DataType) -> Result<HostTensor> {
        SafeTensorsFiles::read(self, name, dtype)
    }
    fn read_raw(&self, name: &str) -> Result<crate::ckpt::CkptTensor> {
        let info = self.info(name).ok_or_else(|| anyhow!("tensor {name} not found"))?;
        let dtype = match info.dtype {
            StDtype::F16 => DataType::F16,
            StDtype::BF16 => DataType::BF16,
            _ => DataType::F32,
        };
        Ok(crate::ckpt::CkptTensor::Dense(SafeTensorsFiles::read(self, name, dtype)?))
    }
}

fn f8e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 3) & 0xF) as i32;
    let man = (b & 0x7) as f32;
    if exp == 0xF && (b & 0x7) == 0x7 {
        return f32::NAN;
    }
    if exp == 0 {
        sign * man / 8.0 * 2f32.powi(-6)
    } else {
        sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

fn f8e5m2_to_f32(b: u8) -> f32 {
    f16::from_bits((b as u16) << 8).to_f32()
}

pub(crate) fn convert_to_f32(src: StDtype, bytes: &[u8]) -> Vec<f32> {
    const CHUNK: usize = 1 << 16;
    let n = bytes.len() / src.size();
    let mut out = vec![0f32; n];
    out.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, dst)| {
        let start = ci * CHUNK;
        for (i, d) in dst.iter_mut().enumerate() {
            let j = start + i;
            *d = match src {
                StDtype::F32 => f32::from_le_bytes(bytes[j * 4..j * 4 + 4].try_into().unwrap()),
                StDtype::F16 => f16::from_le_bytes([bytes[j * 2], bytes[j * 2 + 1]]).to_f32(),
                StDtype::BF16 => bf16::from_le_bytes([bytes[j * 2], bytes[j * 2 + 1]]).to_f32(),
                StDtype::F64 => f64::from_le_bytes(bytes[j * 8..j * 8 + 8].try_into().unwrap()) as f32,
                StDtype::F8E4M3 => f8e4m3_to_f32(bytes[j]),
                StDtype::F8E5M2 => f8e5m2_to_f32(bytes[j]),
                StDtype::I64 => i64::from_le_bytes(bytes[j * 8..j * 8 + 8].try_into().unwrap()) as f32,
                StDtype::I32 => i32::from_le_bytes(bytes[j * 4..j * 4 + 4].try_into().unwrap()) as f32,
                StDtype::U8 | StDtype::Bool => bytes[j] as f32,
            };
        }
    });
    out
}

/// Convert raw safetensors bytes into a host tensor of `dst` type, in parallel.
pub(crate) fn convert_bytes(src: StDtype, bytes: &[u8], dst: DataType, shape: &[usize]) -> HostTensor {
    match (src, dst) {
        (StDtype::F32, DataType::F32) | (StDtype::F16, DataType::F16) | (StDtype::BF16, DataType::BF16) => {
            return HostTensor::from_bytes(bytes.to_vec(), dst, shape);
        }
        (StDtype::BF16, DataType::F16) => {
            let n = bytes.len() / 2;
            let mut out = vec![0u8; n * 2];
            out.par_chunks_mut(1 << 17).zip(bytes.par_chunks(1 << 17)).for_each(|(o, i)| {
                for k in 0..i.len() / 2 {
                    let v = bf16::from_le_bytes([i[2 * k], i[2 * k + 1]]).to_f32();
                    let h = f16::from_f32(v).to_le_bytes();
                    o[2 * k] = h[0];
                    o[2 * k + 1] = h[1];
                }
            });
            return HostTensor::from_bytes(out, dst, shape);
        }
        _ => {}
    }
    let f = convert_to_f32(src, bytes);
    HostTensor::from_f32(&f, shape).to_dtype(dst)
}

/// Named parameters resident on a device.
#[derive(Default)]
pub struct Params {
    map: HashMap<String, Var>,
}

impl Params {
    pub fn new() -> Params {
        Params { map: HashMap::new() }
    }

    pub fn insert(&mut self, name: impl Into<String>, v: Var) {
        self.map.insert(name.into(), v);
    }

    #[track_caller]
    pub fn get(&self, name: &str) -> &Var {
        self.map.get(name).unwrap_or_else(|| panic!("missing parameter {name}"))
    }

    pub fn try_get(&self, name: &str) -> Option<&Var> {
        self.map.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }

    pub fn remove(&mut self, name: &str) -> Option<Var> {
        self.map.remove(name)
    }
}

/// Upload a host tensor, converting to the compute type.
pub fn upload(graph: &Graph, host: &HostTensor, device: Device, dtype: DataType) -> Var {
    let host = if host.dtype() == dtype { host.clone() } else { host.to_dtype(dtype) };
    graph.from_host(&host, device)
}

/// Permute a PyTorch conv weight `[O, I, kh, kw]` to nnc NHWC layout `[O, kh, kw, I]`.
pub fn conv_weight_to_nhwc(w: &[f32], shape: &[usize]) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(shape.len(), 4);
    let (o, i, kh, kw) = (shape[0], shape[1], shape[2], shape[3]);
    let mut out = vec![0f32; w.len()];
    out.par_chunks_mut(kh * kw * i).enumerate().for_each(|(oo, dst)| {
        for ii in 0..i {
            for y in 0..kh {
                for x in 0..kw {
                    dst[(y * kw + x) * i + ii] = w[((oo * i + ii) * kh + y) * kw + x];
                }
            }
        }
    });
    (out, vec![o, kh, kw, i])
}
