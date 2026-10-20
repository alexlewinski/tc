use half::{bf16, f16};

use crate::types::DataType;

/// A dense, contiguous tensor in host memory.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    bytes: Vec<u8>,
    dtype: DataType,
    shape: Vec<usize>,
}

impl HostTensor {
    pub fn from_bytes(bytes: Vec<u8>, dtype: DataType, shape: &[usize]) -> HostTensor {
        let n: usize = shape.iter().product();
        assert_eq!(bytes.len(), n * dtype.size(), "byte length does not match shape {shape:?} {dtype:?}");
        HostTensor { bytes, dtype, shape: shape.to_vec() }
    }

    pub fn from_f32(data: &[f32], shape: &[usize]) -> HostTensor {
        HostTensor::from_bytes(bytemuck::cast_slice(data).to_vec(), DataType::F32, shape)
    }

    pub fn from_f16(data: &[f16], shape: &[usize]) -> HostTensor {
        HostTensor::from_bytes(bytemuck::cast_slice(data).to_vec(), DataType::F16, shape)
    }

    pub fn from_i32(data: &[i32], shape: &[usize]) -> HostTensor {
        HostTensor::from_bytes(bytemuck::cast_slice(data).to_vec(), DataType::I32, shape)
    }

    pub fn zeros(dtype: DataType, shape: &[usize]) -> HostTensor {
        let n: usize = shape.iter().product();
        HostTensor { bytes: vec![0; n * dtype.size()], dtype, shape: shape.to_vec() }
    }

    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn reshape(mut self, shape: &[usize]) -> HostTensor {
        assert_eq!(shape.iter().product::<usize>(), self.numel());
        self.shape = shape.to_vec();
        self
    }

    /// Convert every element to f32.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self.dtype {
            DataType::F32 => bytemuck::pod_collect_to_vec::<u8, f32>(&self.bytes),
            DataType::F16 => bytemuck::pod_collect_to_vec::<u8, f16>(&self.bytes).iter().map(|v| v.to_f32()).collect(),
            DataType::BF16 => bytemuck::pod_collect_to_vec::<u8, bf16>(&self.bytes).iter().map(|v| v.to_f32()).collect(),
            DataType::F64 => bytemuck::pod_collect_to_vec::<u8, f64>(&self.bytes).iter().map(|v| *v as f32).collect(),
            DataType::I32 => bytemuck::pod_collect_to_vec::<u8, i32>(&self.bytes).iter().map(|v| *v as f32).collect(),
            DataType::I64 => bytemuck::pod_collect_to_vec::<u8, i64>(&self.bytes).iter().map(|v| *v as f32).collect(),
            DataType::U8 => self.bytes.iter().map(|v| *v as f32).collect(),
        }
    }

    /// Convert to another element type on the host.
    pub fn to_dtype(&self, dtype: DataType) -> HostTensor {
        if dtype == self.dtype {
            return self.clone();
        }
        let f = self.to_f32_vec();
        let bytes: Vec<u8> = match dtype {
            DataType::F32 => bytemuck::cast_slice(&f).to_vec(),
            DataType::F16 => {
                let v: Vec<f16> = f.iter().map(|x| f16::from_f32(*x)).collect();
                bytemuck::cast_slice(&v).to_vec()
            }
            DataType::BF16 => {
                let v: Vec<bf16> = f.iter().map(|x| bf16::from_f32(*x)).collect();
                bytemuck::cast_slice(&v).to_vec()
            }
            DataType::F64 => {
                let v: Vec<f64> = f.iter().map(|x| *x as f64).collect();
                bytemuck::cast_slice(&v).to_vec()
            }
            DataType::I32 => {
                let v: Vec<i32> = f.iter().map(|x| *x as i32).collect();
                bytemuck::cast_slice(&v).to_vec()
            }
            DataType::I64 => {
                let v: Vec<i64> = f.iter().map(|x| *x as i64).collect();
                bytemuck::cast_slice(&v).to_vec()
            }
            DataType::U8 => f.iter().map(|x| x.clamp(0.0, 255.0) as u8).collect(),
        };
        HostTensor { bytes, dtype, shape: self.shape.clone() }
    }
}
