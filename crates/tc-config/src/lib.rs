//! `GenerationConfiguration`: the generation settings exchanged by the gRPC API (as a
//! FlatBuffers table) and by the CLI (as JSON).
//!
//! The FlatBuffers layout (field order, types and defaults) is part of the wire protocol and
//! is described in `docs/INTERFACE.md`.

mod json;

pub use json::{apply_json_overrides, to_json};

use anyhow::{bail, Result};
use flatbuffers::{FlatBufferBuilder, ForwardsUOffset, Table, Vector, WIPOffset};

/// Sampler identifiers (wire values).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i8)]
pub enum SamplerType {
    DPMPP2MKarras = 0,
    EulerA = 1,
    DDIM = 2,
    PLMS = 3,
    DPMPPSDEKarras = 4,
    UniPC = 5,
    LCM = 6,
    EulerASubstep = 7,
    DPMPPSDESubstep = 8,
    TCD = 9,
    EulerATrailing = 10,
    DPMPPSDETrailing = 11,
    DPMPP2MAYS = 12,
    EulerAAYS = 13,
    DPMPPSDEAYS = 14,
    DPMPP2MTrailing = 15,
    DDIMTrailing = 16,
    UniPCTrailing = 17,
    UniPCAYS = 18,
    TCDTrailing = 19,
}

impl SamplerType {
    pub fn from_raw(v: i8) -> SamplerType {
        use SamplerType::*;
        const ALL: [SamplerType; 20] = [
            DPMPP2MKarras, EulerA, DDIM, PLMS, DPMPPSDEKarras, UniPC, LCM, EulerASubstep, DPMPPSDESubstep, TCD,
            EulerATrailing, DPMPPSDETrailing, DPMPP2MAYS, EulerAAYS, DPMPPSDEAYS, DPMPP2MTrailing, DDIMTrailing,
            UniPCTrailing, UniPCAYS, TCDTrailing,
        ];
        ALL.get(v.max(0) as usize).copied().unwrap_or(DPMPP2MKarras)
    }

    /// Whether this is an ancestral (stochastic) sampler.
    pub fn is_ancestral(self) -> bool {
        use SamplerType::*;
        matches!(self, EulerA | EulerASubstep | EulerATrailing | EulerAAYS | DPMPPSDEKarras | DPMPPSDESubstep | DPMPPSDETrailing | DPMPPSDEAYS | LCM | TCD | TCDTrailing)
    }
}

/// LoRA reference inside a configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct LoRA {
    pub file: Option<String>,
    pub weight: f32,
    /// 0 = all, 1 = base, 2 = refiner.
    pub mode: i8,
}

impl Default for LoRA {
    fn default() -> Self {
        LoRA { file: None, weight: 0.6, mode: 0 }
    }
}

/// Control (ControlNet etc.) reference inside a configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct Control {
    pub file: Option<String>,
    pub weight: f32,
    pub guidance_start: f32,
    pub guidance_end: f32,
    pub no_prompt: bool,
    pub global_average_pooling: bool,
    pub down_sampling_rate: f32,
    pub control_mode: i8,
    pub target_blocks: Vec<String>,
    pub input_override: i8,
}

impl Default for Control {
    fn default() -> Self {
        Control {
            file: None,
            weight: 1.0,
            guidance_start: 0.0,
            guidance_end: 1.0,
            no_prompt: false,
            global_average_pooling: true,
            down_sampling_rate: 1.0,
            control_mode: 0,
            target_blocks: vec![],
            input_override: 0,
        }
    }
}

/// Helper arms shared by `config_table!` (types, defaults, readers, writers).
macro_rules! config {
    (@ty u16) => { u16 };
    (@ty u32) => { u32 };
    (@ty i32) => { i32 };
    (@ty i64) => { i64 };
    (@ty u8) => { u8 };
    (@ty i8) => { i8 };
    (@ty f32) => { f32 };
    (@ty bool) => { bool };
    (@ty str) => { Option<String> };
    (@ty loras) => { Vec<LoRA> };
    (@ty controls) => { Vec<Control> };

    (@conv str, $d:expr) => { $d };
    (@conv loras, $d:expr) => { $d };
    (@conv controls, $d:expr) => { $d };
    (@conv $k:ident, $d:expr) => { $d as config!(@ty $k) };

    (@read str, $t:ident, $slot:literal, $d:expr) => {
        unsafe { $t.get::<ForwardsUOffset<&str>>(4 + 2 * $slot, None) }.map(|s| s.to_string())
    };
    (@read loras, $t:ident, $slot:literal, $d:expr) => { read_loras(&$t, 4 + 2 * $slot) };
    (@read controls, $t:ident, $slot:literal, $d:expr) => { read_controls(&$t, 4 + 2 * $slot) };
    (@read $k:ident, $t:ident, $slot:literal, $d:expr) => {
        unsafe { $t.get::<config!(@ty $k)>(4 + 2 * $slot, Some($d)) }.unwrap_or($d)
    };

    (@prewrite str, $b:ident, $v:expr) => { $v.as_ref().map(|s| $b.create_string(s)) };
    (@prewrite loras, $b:ident, $v:expr) => { Some(write_loras(&mut $b, &$v)) };
    (@prewrite controls, $b:ident, $v:expr) => { Some(write_controls(&mut $b, &$v)) };
    (@prewrite $k:ident, $b:ident, $v:expr) => { () };

}

macro_rules! config_table {
    ($( $slot:literal $name:ident : $kind:ident = $default:expr ; )*) => {
        /// All generation settings. Sizes are in 64-pixel units where the field name says
        /// `start_*`, `hires_fix_start_*` and `decoding_tile_*` / `diffusion_tile_*`.
        #[derive(Clone, Debug, PartialEq)]
        pub struct GenerationConfiguration {
            $( pub $name: config!(@ty $kind), )*
        }

        impl Default for GenerationConfiguration {
            fn default() -> Self {
                GenerationConfiguration { $( $name: config!(@conv $kind, $default), )* }
            }
        }

        impl GenerationConfiguration {
            /// Decode from FlatBuffers bytes (root table). Empty input yields defaults.
            pub fn from_flatbuffer(buf: &[u8]) -> Result<GenerationConfiguration> {
                if buf.is_empty() {
                    return Ok(GenerationConfiguration::default());
                }
                let t = root_table(buf)?;
                let d = GenerationConfiguration::default();
                Ok(GenerationConfiguration {
                    $( $name: config!(@read $kind, t, $slot, d.$name), )*
                })
            }

            /// Encode as FlatBuffers bytes. Fields equal to their default are omitted.
            pub fn to_flatbuffer(&self) -> Vec<u8> {
                let d = GenerationConfiguration::default();
                let mut b = FlatBufferBuilder::new();
                $( #[allow(unused_variables)] let $name = config!(@prewrite $kind, b, self.$name); )*
                let start = b.start_table();
                $( config_table!(@write $kind, b, $slot, $name, self.$name, d.$name); )*
                let root = b.end_table(start);
                b.finish_minimal(root);
                b.finished_data().to_vec()
            }
        }
    };
    (@write str, $b:ident, $slot:literal, $pre:ident, $v:expr, $d:expr) => {
        if let Some(o) = $pre { $b.push_slot_always(4 + 2 * $slot, o); }
    };
    (@write loras, $b:ident, $slot:literal, $pre:ident, $v:expr, $d:expr) => {
        if let Some(o) = $pre { $b.push_slot_always(4 + 2 * $slot, o); }
    };
    (@write controls, $b:ident, $slot:literal, $pre:ident, $v:expr, $d:expr) => {
        if let Some(o) = $pre { $b.push_slot_always(4 + 2 * $slot, o); }
    };
    (@write $k:ident, $b:ident, $slot:literal, $pre:ident, $v:expr, $d:expr) => {
        $b.push_slot::<config!(@ty $k)>(4 + 2 * $slot, $v, $d);
    };
}

fn root_table(buf: &[u8]) -> Result<Table<'_>> {
    if buf.len() < 8 {
        bail!("configuration buffer too small");
    }
    let root = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    if root + 4 > buf.len() {
        bail!("invalid configuration root offset");
    }
    // Validate the vtable reference before handing out a table.
    let soffset = i32::from_le_bytes(buf[root..root + 4].try_into().unwrap()) as isize;
    let vt = root as isize - soffset;
    if vt < 0 || vt as usize + 4 > buf.len() {
        bail!("invalid configuration vtable");
    }
    let vt_len = u16::from_le_bytes(buf[vt as usize..vt as usize + 2].try_into().unwrap()) as usize;
    if vt as usize + vt_len > buf.len() {
        bail!("invalid configuration vtable length");
    }
    Ok(unsafe { Table::new(buf, root) })
}

fn read_loras(t: &Table, slot: u16) -> Vec<LoRA> {
    let Some(v) = (unsafe { t.get::<ForwardsUOffset<Vector<ForwardsUOffset<Table>>>>(slot, None) }) else {
        return vec![];
    };
    v.iter()
        .map(|l| unsafe {
            LoRA {
                file: l.get::<ForwardsUOffset<&str>>(4, None).map(|s| s.to_string()),
                weight: l.get::<f32>(6, Some(0.6)).unwrap_or(0.6),
                mode: l.get::<i8>(8, Some(0)).unwrap_or(0),
            }
        })
        .collect()
}

fn write_loras<'a>(b: &mut FlatBufferBuilder<'a>, loras: &[LoRA]) -> WIPOffset<flatbuffers::Vector<'a, ForwardsUOffset<flatbuffers::TableFinishedWIPOffset>>> {
    let offs: Vec<_> = loras
        .iter()
        .map(|l| {
            let file = l.file.as_ref().map(|f| b.create_string(f));
            let s = b.start_table();
            if let Some(f) = file {
                b.push_slot_always(4, f);
            }
            b.push_slot::<f32>(6, l.weight, 0.6);
            b.push_slot::<i8>(8, l.mode, 0);
            b.end_table(s)
        })
        .collect();
    b.create_vector(&offs)
}

fn read_controls(t: &Table, slot: u16) -> Vec<Control> {
    let Some(v) = (unsafe { t.get::<ForwardsUOffset<Vector<ForwardsUOffset<Table>>>>(slot, None) }) else {
        return vec![];
    };
    let d = Control::default();
    v.iter()
        .map(|c| unsafe {
            Control {
                file: c.get::<ForwardsUOffset<&str>>(4, None).map(|s| s.to_string()),
                weight: c.get::<f32>(6, Some(d.weight)).unwrap_or(d.weight),
                guidance_start: c.get::<f32>(8, Some(d.guidance_start)).unwrap_or(d.guidance_start),
                guidance_end: c.get::<f32>(10, Some(d.guidance_end)).unwrap_or(d.guidance_end),
                no_prompt: c.get::<bool>(12, Some(false)).unwrap_or(false),
                global_average_pooling: c.get::<bool>(14, Some(true)).unwrap_or(true),
                down_sampling_rate: c.get::<f32>(16, Some(1.0)).unwrap_or(1.0),
                control_mode: c.get::<i8>(18, Some(0)).unwrap_or(0),
                target_blocks: c
                    .get::<ForwardsUOffset<Vector<ForwardsUOffset<&str>>>>(20, None)
                    .map(|v| v.iter().map(|s| s.to_string()).collect())
                    .unwrap_or_default(),
                input_override: c.get::<i8>(22, Some(0)).unwrap_or(0),
            }
        })
        .collect()
}

fn write_controls<'a>(b: &mut FlatBufferBuilder<'a>, controls: &[Control]) -> WIPOffset<flatbuffers::Vector<'a, ForwardsUOffset<flatbuffers::TableFinishedWIPOffset>>> {
    let d = Control::default();
    let offs: Vec<_> = controls
        .iter()
        .map(|c| {
            let file = c.file.as_ref().map(|f| b.create_string(f));
            let blocks = if c.target_blocks.is_empty() {
                None
            } else {
                let s: Vec<_> = c.target_blocks.iter().map(|t| b.create_string(t)).collect();
                Some(b.create_vector(&s))
            };
            let s = b.start_table();
            if let Some(f) = file {
                b.push_slot_always(4, f);
            }
            b.push_slot::<f32>(6, c.weight, d.weight);
            b.push_slot::<f32>(8, c.guidance_start, d.guidance_start);
            b.push_slot::<f32>(10, c.guidance_end, d.guidance_end);
            b.push_slot::<bool>(12, c.no_prompt, false);
            b.push_slot::<bool>(14, c.global_average_pooling, true);
            b.push_slot::<f32>(16, c.down_sampling_rate, 1.0);
            b.push_slot::<i8>(18, c.control_mode, 0);
            if let Some(bl) = blocks {
                b.push_slot_always(20, bl);
            }
            b.push_slot::<i8>(22, c.input_override, 0);
            b.end_table(s)
        })
        .collect();
    b.create_vector(&offs)
}

config_table! {
    0 id: i64 = 0;
    1 start_width: u16 = 0;
    2 start_height: u16 = 0;
    3 seed: u32 = 0;
    4 steps: u32 = 0;
    5 guidance_scale: f32 = 0.0;
    6 strength: f32 = 0.0;
    7 model: str = None;
    8 sampler: i8 = 0;
    9 batch_count: u32 = 1;
    10 batch_size: u32 = 1;
    11 hires_fix: bool = false;
    12 hires_fix_start_width: u16 = 0;
    13 hires_fix_start_height: u16 = 0;
    14 hires_fix_strength: f32 = 0.7;
    15 upscaler: str = None;
    16 image_guidance_scale: f32 = 1.5;
    17 seed_mode: i8 = 0;
    18 clip_skip: u32 = 1;
    19 controls: controls = vec![];
    20 loras: loras = vec![];
    21 mask_blur: f32 = 0.0;
    22 face_restoration: str = None;
    23 decode_with_attention: bool = true;
    24 hires_fix_decode_with_attention: bool = true;
    25 clip_weight: f32 = 1.0;
    26 negative_prompt_for_image_prior: bool = true;
    27 image_prior_steps: u32 = 5;
    28 refiner_model: str = None;
    29 original_image_height: u32 = 0;
    30 original_image_width: u32 = 0;
    31 crop_top: i32 = 0;
    32 crop_left: i32 = 0;
    33 target_image_height: u32 = 0;
    34 target_image_width: u32 = 0;
    35 aesthetic_score: f32 = 6.0;
    36 negative_aesthetic_score: f32 = 2.5;
    37 zero_negative_prompt: bool = false;
    38 refiner_start: f32 = 0.7;
    39 negative_original_image_height: u32 = 0;
    40 negative_original_image_width: u32 = 0;
    41 name: str = None;
    42 fps_id: u32 = 5;
    43 motion_bucket_id: u32 = 127;
    44 cond_aug: f32 = 0.02;
    45 start_frame_cfg: f32 = 1.0;
    46 num_frames: u32 = 14;
    47 mask_blur_outset: i32 = 0;
    48 sharpness: f32 = 0.0;
    49 shift: f32 = 1.0;
    50 stage_2_steps: u32 = 10;
    51 stage_2_cfg: f32 = 1.0;
    52 stage_2_shift: f32 = 1.0;
    53 tiled_decoding: bool = false;
    54 decoding_tile_width: u16 = 10;
    55 decoding_tile_height: u16 = 10;
    56 decoding_tile_overlap: u16 = 2;
    57 stochastic_sampling_gamma: f32 = 0.3;
    58 preserve_original_after_inpaint: bool = true;
    59 tiled_diffusion: bool = false;
    60 diffusion_tile_width: u16 = 16;
    61 diffusion_tile_height: u16 = 16;
    62 diffusion_tile_overlap: u16 = 2;
    63 upscaler_scale_factor: u8 = 0;
    64 t5_text_encoder: bool = true;
    65 separate_clip_l: bool = false;
    66 clip_l_text: str = None;
    67 separate_open_clip_g: bool = false;
    68 open_clip_g_text: str = None;
    69 speed_up_with_guidance_embed: bool = true;
    70 guidance_embed: f32 = 3.5;
    71 resolution_dependent_shift: bool = true;
    72 tea_cache_start: i32 = 5;
    73 tea_cache_end: i32 = -1;
    74 tea_cache_threshold: f32 = 0.06;
    75 tea_cache: bool = false;
    76 separate_t5: bool = false;
    77 t5_text: str = None;
    78 tea_cache_max_skip_steps: i32 = 3;
    79 causal_inference_enabled: bool = false;
    80 causal_inference: i32 = 3;
    81 causal_inference_pad: i32 = 0;
    82 cfg_zero_star: bool = false;
    83 cfg_zero_init_steps: i32 = 0;
    84 compression_artifacts: i8 = 0;
    85 compression_artifacts_quality: f32 = 43.1;
    86 color_calibration: i8 = 0;
    87 expand_prompt_to_json: bool = false;
    88 shift_for_audio: f32 = 3.0;
    89 uses_sol_attention: bool = false;
    90 sol_attention_start: i32 = 2;
    91 sol_attention_tau: f32 = 0.5;
}

impl GenerationConfiguration {
    pub fn width_px(&self) -> u32 {
        self.start_width as u32 * 64
    }
    pub fn height_px(&self) -> u32 {
        self.start_height as u32 * 64
    }
    pub fn sampler_type(&self) -> SamplerType {
        SamplerType::from_raw(self.sampler)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatbuffer_roundtrip() {
        let mut c = GenerationConfiguration::default();
        c.start_width = 16;
        c.start_height = 12;
        c.seed = 1234;
        c.steps = 4;
        c.guidance_scale = 1.0;
        c.strength = 1.0;
        c.model = Some("flux_2_klein_9b".into());
        c.sampler = SamplerType::EulerATrailing as i8;
        c.loras = vec![
            LoRA { file: Some("a.safetensors".into()), weight: 0.8, mode: 0 },
            LoRA { file: Some("b.safetensors".into()), weight: 0.5, mode: 1 },
        ];
        c.controls = vec![Control { file: Some("c".into()), target_blocks: vec!["x".into()], ..Default::default() }];
        c.tea_cache_end = -1;
        c.shift = 2.5;
        c.sol_attention_tau = 0.75;
        let bytes = c.to_flatbuffer();
        let d = GenerationConfiguration::from_flatbuffer(&bytes).unwrap();
        assert_eq!(c, d);
        assert_eq!(GenerationConfiguration::from_flatbuffer(&[]).unwrap(), GenerationConfiguration::default());
        assert!(GenerationConfiguration::from_flatbuffer(&[1, 2, 3]).is_err());
    }
}
