//! JSON form of the configuration (camelCase keys, sizes in pixels), as accepted by
//! `draw-things-cli generate --config-json`. snake_case keys are accepted too.

use anyhow::{bail, Result};
use serde_json::{Map, Value};

use crate::{Control, GenerationConfiguration, LoRA};

fn snake_to_camel(s: &str) -> String {
    let mut out = String::new();
    let mut up = false;
    for c in s.chars() {
        if c == '_' {
            up = true;
        } else if up {
            out.extend(c.to_uppercase());
            up = false;
        } else {
            out.push(c);
        }
    }
    out
}

fn lora_mode_str(m: i8) -> &'static str {
    match m {
        1 => "base",
        2 => "refiner",
        _ => "all",
    }
}

const COMPRESSION: [&str; 4] = ["disabled", "h264", "h265", "jpeg"];
const CONTROL_MODE: [&str; 3] = ["balanced", "prompt", "control"];
const INPUT_OVERRIDE: [&str; 19] = [
    "unspecified", "custom", "depth", "canny", "scribble", "pose", "normalbae", "color", "lineart", "softedge",
    "seg", "inpaint", "ip2p", "shuffle", "mlsd", "tile", "blur", "lowquality", "gray",
];

/// Serialize to the JSON form.
pub fn to_json(c: &GenerationConfiguration) -> Value {
    let mut m = Map::new();
    let mut put = |k: &str, v: Value| {
        m.insert(k.to_string(), v);
    };
    put("id", c.id.into());
    put("width", (c.start_width as u32 * 64).into());
    put("height", (c.start_height as u32 * 64).into());
    put("seed", (c.seed as i64).into());
    put("steps", c.steps.into());
    put("guidanceScale", c.guidance_scale.into());
    put("strength", c.strength.into());
    put("model", c.model.clone().into());
    put("sampler", c.sampler.into());
    put("hiresFix", c.hires_fix.into());
    put("hiresFixWidth", (c.hires_fix_start_width as u32 * 64).into());
    put("hiresFixHeight", (c.hires_fix_start_height as u32 * 64).into());
    put("hiresFixStrength", c.hires_fix_strength.into());
    put("tiledDecoding", c.tiled_decoding.into());
    put("decodingTileWidth", (c.decoding_tile_width as u32 * 64).into());
    put("decodingTileHeight", (c.decoding_tile_height as u32 * 64).into());
    put("decodingTileOverlap", (c.decoding_tile_overlap as u32 * 64).into());
    put("tiledDiffusion", c.tiled_diffusion.into());
    put("diffusionTileWidth", (c.diffusion_tile_width as u32 * 64).into());
    put("diffusionTileHeight", (c.diffusion_tile_height as u32 * 64).into());
    put("diffusionTileOverlap", (c.diffusion_tile_overlap as u32 * 64).into());
    put("upscaler", c.upscaler.clone().into());
    put("upscalerScaleFactor", c.upscaler_scale_factor.into());
    put("imageGuidanceScale", c.image_guidance_scale.into());
    put("seedMode", c.seed_mode.into());
    put("clipSkip", c.clip_skip.into());
    put(
        "controls",
        Value::Array(
            c.controls
                .iter()
                .map(|x| {
                    serde_json::json!({
                        "file": x.file, "weight": x.weight, "guidanceStart": x.guidance_start,
                        "guidanceEnd": x.guidance_end, "noPrompt": x.no_prompt,
                        "globalAveragePooling": x.global_average_pooling,
                        "downSamplingRate": x.down_sampling_rate,
                        "controlImportance": CONTROL_MODE.get(x.control_mode as usize).unwrap_or(&"balanced"),
                        "inputOverride": INPUT_OVERRIDE.get(x.input_override as usize).unwrap_or(&"unspecified"),
                        "targetBlocks": x.target_blocks,
                    })
                })
                .collect(),
        ),
    );
    put(
        "loras",
        Value::Array(
            c.loras
                .iter()
                .map(|l| serde_json::json!({"file": l.file, "weight": l.weight, "mode": lora_mode_str(l.mode)}))
                .collect(),
        ),
    );
    put("maskBlur", c.mask_blur.into());
    put("maskBlurOutset", c.mask_blur_outset.into());
    put("sharpness", c.sharpness.into());
    put("faceRestoration", c.face_restoration.clone().into());
    put("clipWeight", c.clip_weight.into());
    put("negativePromptForImagePrior", c.negative_prompt_for_image_prior.into());
    put("imagePriorSteps", c.image_prior_steps.into());
    put("refinerModel", c.refiner_model.clone().into());
    put("originalImageHeight", c.original_image_height.into());
    put("originalImageWidth", c.original_image_width.into());
    put("cropTop", c.crop_top.into());
    put("cropLeft", c.crop_left.into());
    put("targetImageHeight", c.target_image_height.into());
    put("targetImageWidth", c.target_image_width.into());
    put("aestheticScore", c.aesthetic_score.into());
    put("negativeAestheticScore", c.negative_aesthetic_score.into());
    put("zeroNegativePrompt", c.zero_negative_prompt.into());
    put("refinerStart", c.refiner_start.into());
    put("negativeOriginalImageHeight", c.negative_original_image_height.into());
    put("negativeOriginalImageWidth", c.negative_original_image_width.into());
    put("batchCount", c.batch_count.into());
    put("batchSize", c.batch_size.into());
    put("numFrames", c.num_frames.into());
    put("fps", c.fps_id.into());
    put("motionScale", c.motion_bucket_id.into());
    put("guidingFrameNoise", c.cond_aug.into());
    put("startFrameGuidance", c.start_frame_cfg.into());
    put("shift", c.shift.into());
    put("shiftForAudio", c.shift_for_audio.into());
    put("usesSolAttention", c.uses_sol_attention.into());
    put("solAttentionStart", c.sol_attention_start.into());
    put("solAttentionTau", c.sol_attention_tau.into());
    put("stage2Steps", c.stage_2_steps.into());
    put("stage2Guidance", c.stage_2_cfg.into());
    put("stage2Shift", c.stage_2_shift.into());
    put("stochasticSamplingGamma", c.stochastic_sampling_gamma.into());
    put("preserveOriginalAfterInpaint", c.preserve_original_after_inpaint.into());
    put("t5TextEncoder", c.t5_text_encoder.into());
    put("separateClipL", c.separate_clip_l.into());
    put("clipLText", c.clip_l_text.clone().into());
    put("separateOpenClipG", c.separate_open_clip_g.into());
    put("openClipGText", c.open_clip_g_text.clone().into());
    put("speedUpWithGuidanceEmbed", c.speed_up_with_guidance_embed.into());
    put("guidanceEmbed", c.guidance_embed.into());
    put("resolutionDependentShift", c.resolution_dependent_shift.into());
    put("teaCache", c.tea_cache.into());
    put("teaCacheStart", c.tea_cache_start.into());
    put("teaCacheEnd", c.tea_cache_end.into());
    put("teaCacheThreshold", c.tea_cache_threshold.into());
    put("teaCacheMaxSkipSteps", c.tea_cache_max_skip_steps.into());
    put("separateT5", c.separate_t5.into());
    put("t5Text", c.t5_text.clone().into());
    put("causalInference", (if c.causal_inference_enabled { c.causal_inference } else { 0 }).into());
    put("causalInferencePad", c.causal_inference_pad.into());
    put("cfgZeroStar", c.cfg_zero_star.into());
    put("cfgZeroInitSteps", c.cfg_zero_init_steps.into());
    put("compressionArtifacts", COMPRESSION.get(c.compression_artifacts as usize).copied().unwrap_or("disabled").into());
    put("compressionArtifactsQuality", c.compression_artifacts_quality.into());
    put("colorCalibration", (if c.color_calibration == 1 { "lab" } else { "disabled" }).into());
    put("expandPromptToJson", c.expand_prompt_to_json.into());
    Value::Object(m)
}

fn num(v: &Value, key: &str) -> Result<f64> {
    match v {
        Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0)),
        Value::Bool(b) => Ok(*b as i32 as f64),
        Value::String(s) => s.parse().map_err(|_| anyhow::anyhow!("{key}: expected a number, got {s:?}")),
        _ => bail!("{key}: expected a number"),
    }
}

fn boolean(v: &Value, key: &str) -> Result<bool> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0) != 0.0),
        _ => bail!("{key}: expected a boolean"),
    }
}

fn string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn px_to_units(px: f64) -> u16 {
    ((px / 64.0).round().max(0.0)) as u16
}

/// Apply a (partial) JSON object onto `c`. Unknown keys are returned.
pub fn apply_json_overrides(c: &mut GenerationConfiguration, json: &Value) -> Result<Vec<String>> {
    let Value::Object(obj) = json else { bail!("configuration JSON must be an object") };
    let mut unknown = vec![];
    for (raw_key, v) in obj {
        let key = if raw_key.contains('_') { snake_to_camel(raw_key) } else { raw_key.clone() };
        let k = key.as_str();
        match k {
            "id" => c.id = num(v, k)? as i64,
            "width" => c.start_width = px_to_units(num(v, k)?),
            "height" => c.start_height = px_to_units(num(v, k)?),
            "seed" => c.seed = num(v, k)? as i64 as u32,
            "steps" => c.steps = num(v, k)? as u32,
            "guidanceScale" | "cfg" => c.guidance_scale = num(v, k)? as f32,
            "strength" => c.strength = num(v, k)? as f32,
            "model" => c.model = string(v),
            "sampler" => {
                c.sampler = match v {
                    Value::String(s) => sampler_from_name(s).ok_or_else(|| anyhow::anyhow!("unknown sampler {s:?}"))?,
                    _ => num(v, k)? as i8,
                }
            }
            "hiresFix" => c.hires_fix = boolean(v, k)?,
            "hiresFixWidth" => c.hires_fix_start_width = px_to_units(num(v, k)?),
            "hiresFixHeight" => c.hires_fix_start_height = px_to_units(num(v, k)?),
            "hiresFixStrength" => c.hires_fix_strength = num(v, k)? as f32,
            "tiledDecoding" => c.tiled_decoding = boolean(v, k)?,
            "decodingTileWidth" => c.decoding_tile_width = px_to_units(num(v, k)?),
            "decodingTileHeight" => c.decoding_tile_height = px_to_units(num(v, k)?),
            "decodingTileOverlap" => c.decoding_tile_overlap = px_to_units(num(v, k)?),
            "tiledDiffusion" => c.tiled_diffusion = boolean(v, k)?,
            "diffusionTileWidth" => c.diffusion_tile_width = px_to_units(num(v, k)?),
            "diffusionTileHeight" => c.diffusion_tile_height = px_to_units(num(v, k)?),
            "diffusionTileOverlap" => c.diffusion_tile_overlap = px_to_units(num(v, k)?),
            "upscaler" => c.upscaler = string(v),
            "upscalerScaleFactor" => c.upscaler_scale_factor = num(v, k)? as u8,
            "imageGuidanceScale" => c.image_guidance_scale = num(v, k)? as f32,
            "seedMode" => c.seed_mode = num(v, k)? as i8,
            "clipSkip" => c.clip_skip = num(v, k)? as u32,
            "loras" => {
                let Value::Array(a) = v else { bail!("loras must be an array") };
                c.loras = a
                    .iter()
                    .filter_map(|l| l.as_object())
                    .map(|l| LoRA {
                        file: l.get("file").and_then(string),
                        weight: l.get("weight").and_then(|w| w.as_f64()).unwrap_or(0.6) as f32,
                        mode: match l.get("mode").and_then(|m| m.as_str()) {
                            Some("base") => 1,
                            Some("refiner") => 2,
                            _ => 0,
                        },
                    })
                    .collect();
            }
            "controls" => {
                let Value::Array(a) = v else { bail!("controls must be an array") };
                c.controls = a
                    .iter()
                    .filter_map(|x| x.as_object())
                    .map(|x| {
                        let d = Control::default();
                        let f = |n: &str, d: f32| x.get(n).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(d);
                        let b = |n: &str, d: bool| x.get(n).and_then(|v| v.as_bool()).unwrap_or(d);
                        Control {
                            file: x.get("file").and_then(string),
                            weight: f("weight", d.weight),
                            guidance_start: f("guidanceStart", d.guidance_start),
                            guidance_end: f("guidanceEnd", d.guidance_end),
                            no_prompt: b("noPrompt", d.no_prompt),
                            global_average_pooling: b("globalAveragePooling", d.global_average_pooling),
                            down_sampling_rate: f("downSamplingRate", d.down_sampling_rate),
                            control_mode: x
                                .get("controlImportance")
                                .and_then(|v| v.as_str())
                                .and_then(|s| CONTROL_MODE.iter().position(|m| m.eq_ignore_ascii_case(s)))
                                .unwrap_or(0) as i8,
                            target_blocks: x
                                .get("targetBlocks")
                                .and_then(|v| v.as_array())
                                .map(|a| a.iter().filter_map(string).collect())
                                .unwrap_or_default(),
                            input_override: x
                                .get("inputOverride")
                                .and_then(|v| v.as_str())
                                .and_then(|s| INPUT_OVERRIDE.iter().position(|m| m.eq_ignore_ascii_case(s)))
                                .unwrap_or(0) as i8,
                        }
                    })
                    .collect();
            }
            "maskBlur" => c.mask_blur = num(v, k)? as f32,
            "maskBlurOutset" => c.mask_blur_outset = num(v, k)? as i32,
            "sharpness" => c.sharpness = num(v, k)? as f32,
            "faceRestoration" => c.face_restoration = string(v),
            "clipWeight" => c.clip_weight = num(v, k)? as f32,
            "negativePromptForImagePrior" => c.negative_prompt_for_image_prior = boolean(v, k)?,
            "imagePriorSteps" => c.image_prior_steps = num(v, k)? as u32,
            "refinerModel" => c.refiner_model = string(v),
            "originalImageHeight" => c.original_image_height = num(v, k)? as u32,
            "originalImageWidth" => c.original_image_width = num(v, k)? as u32,
            "cropTop" => c.crop_top = num(v, k)? as i32,
            "cropLeft" => c.crop_left = num(v, k)? as i32,
            "targetImageHeight" => c.target_image_height = num(v, k)? as u32,
            "targetImageWidth" => c.target_image_width = num(v, k)? as u32,
            "aestheticScore" => c.aesthetic_score = num(v, k)? as f32,
            "negativeAestheticScore" => c.negative_aesthetic_score = num(v, k)? as f32,
            "zeroNegativePrompt" => c.zero_negative_prompt = boolean(v, k)?,
            "refinerStart" => c.refiner_start = num(v, k)? as f32,
            "negativeOriginalImageHeight" => c.negative_original_image_height = num(v, k)? as u32,
            "negativeOriginalImageWidth" => c.negative_original_image_width = num(v, k)? as u32,
            "batchCount" => c.batch_count = num(v, k)? as u32,
            "batchSize" => c.batch_size = num(v, k)? as u32,
            "numFrames" => c.num_frames = num(v, k)? as u32,
            "fps" => c.fps_id = num(v, k)? as u32,
            "motionScale" => c.motion_bucket_id = num(v, k)? as u32,
            "guidingFrameNoise" => c.cond_aug = num(v, k)? as f32,
            "startFrameGuidance" => c.start_frame_cfg = num(v, k)? as f32,
            "shift" => c.shift = num(v, k)? as f32,
            "shiftForAudio" => c.shift_for_audio = num(v, k)? as f32,
            "usesSolAttention" => c.uses_sol_attention = boolean(v, k)?,
            "solAttentionStart" => c.sol_attention_start = num(v, k)? as i32,
            "solAttentionTau" => c.sol_attention_tau = num(v, k)? as f32,
            "stage2Steps" => c.stage_2_steps = num(v, k)? as u32,
            "stage2Guidance" | "stage2Cfg" => c.stage_2_cfg = num(v, k)? as f32,
            "stage2Shift" => c.stage_2_shift = num(v, k)? as f32,
            "stochasticSamplingGamma" => c.stochastic_sampling_gamma = num(v, k)? as f32,
            "preserveOriginalAfterInpaint" => c.preserve_original_after_inpaint = boolean(v, k)?,
            "t5TextEncoder" => c.t5_text_encoder = boolean(v, k)?,
            "separateClipL" => c.separate_clip_l = boolean(v, k)?,
            "clipLText" => c.clip_l_text = string(v),
            "separateOpenClipG" => c.separate_open_clip_g = boolean(v, k)?,
            "openClipGText" => c.open_clip_g_text = string(v),
            "speedUpWithGuidanceEmbed" => c.speed_up_with_guidance_embed = boolean(v, k)?,
            "guidanceEmbed" => c.guidance_embed = num(v, k)? as f32,
            "resolutionDependentShift" => c.resolution_dependent_shift = boolean(v, k)?,
            "teaCache" => c.tea_cache = boolean(v, k)?,
            "teaCacheStart" => c.tea_cache_start = num(v, k)? as i32,
            "teaCacheEnd" => c.tea_cache_end = num(v, k)? as i32,
            "teaCacheThreshold" => c.tea_cache_threshold = num(v, k)? as f32,
            "teaCacheMaxSkipSteps" => c.tea_cache_max_skip_steps = num(v, k)? as i32,
            "separateT5" => c.separate_t5 = boolean(v, k)?,
            "t5Text" => c.t5_text = string(v),
            "causalInference" => {
                let n = num(v, k)? as i32;
                c.causal_inference_enabled = n > 0;
                if n > 0 {
                    c.causal_inference = n;
                }
            }
            "causalInferencePad" => c.causal_inference_pad = num(v, k)? as i32,
            "cfgZeroStar" => c.cfg_zero_star = boolean(v, k)?,
            "cfgZeroInitSteps" => c.cfg_zero_init_steps = num(v, k)? as i32,
            "compressionArtifacts" => {
                c.compression_artifacts = string(v)
                    .and_then(|s| COMPRESSION.iter().position(|m| m.eq_ignore_ascii_case(&s)))
                    .unwrap_or(0) as i8
            }
            "compressionArtifactsQuality" => c.compression_artifacts_quality = num(v, k)? as f32,
            "colorCalibration" => c.color_calibration = (string(v).map(|s| s.eq_ignore_ascii_case("lab")).unwrap_or(false)) as i8,
            "expandPromptToJson" => c.expand_prompt_to_json = boolean(v, k)?,
            _ => unknown.push(raw_key.clone()),
        }
    }
    Ok(unknown)
}

/// Parse a sampler name (case-insensitive, punctuation ignored), e.g. "Euler A Trailing".
pub fn sampler_from_name(name: &str) -> Option<i8> {
    let norm: String = name.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
    const NAMES: [&str; 20] = [
        "dpmpp2mkarras", "eulera", "ddim", "plms", "dpmppsdekarras", "unipc", "lcm", "euleraSubstep",
        "dpmppsdesubstep", "tcd", "euleratrailing", "dpmppsdetrailing", "dpmpp2mays", "euleraays",
        "dpmppsdeays", "dpmpp2mtrailing", "ddimtrailing", "unipctrailing", "unipcays", "tcdtrailing",
    ];
    NAMES.iter().position(|n| n.to_lowercase() == norm).map(|i| i as i8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip_and_overrides() {
        let mut c = GenerationConfiguration::default();
        c.start_width = 16;
        c.start_height = 16;
        c.loras = vec![LoRA { file: Some("x.safetensors".into()), weight: 0.7, mode: 0 }];
        let j = to_json(&c);
        let mut d = GenerationConfiguration::default();
        assert!(apply_json_overrides(&mut d, &j).unwrap().is_empty());
        assert_eq!(c, d);
        let unknown = apply_json_overrides(
            &mut d,
            &serde_json::json!({"steps": 4, "guidance_scale": 1.5, "width": 768, "sampler": "Euler A Trailing", "bogus": 1}),
        )
        .unwrap();
        assert_eq!(unknown, vec!["bogus".to_string()]);
        assert_eq!(d.steps, 4);
        assert_eq!(d.guidance_scale, 1.5);
        assert_eq!(d.start_width, 12);
        assert_eq!(d.sampler, 10);
    }
}
