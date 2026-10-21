//! Image <-> tensor conversion helpers. Tensors are NHWC float in `[-1, 1]`.

use std::path::Path;

use anyhow::{Context, Result};
use image::{imageops::FilterType, RgbImage};
use nnc::HostTensor;

pub fn load_rgb(path: &Path) -> Result<RgbImage> {
    Ok(flatten_on_white(image::open(path).with_context(|| format!("reading image {}", path.display()))?))
}

pub fn decode_rgb(bytes: &[u8]) -> Result<RgbImage> {
    Ok(flatten_on_white(image::load_from_memory(bytes).context("decoding image")?))
}

/// Drop alpha by compositing over white: transparent areas (e.g. a sketch on a transparent
/// background) become white instead of whatever colour the transparent pixels happen to store.
fn flatten_on_white(img: image::DynamicImage) -> RgbImage {
    if !img.color().has_alpha() {
        return img.to_rgb8();
    }
    let rgba = img.to_rgba8();
    RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let p = rgba.get_pixel(x, y).0;
        let a = p[3] as u32;
        let blend = |c: u8| ((c as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
        image::Rgb([blend(p[0]), blend(p[1]), blend(p[2])])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transparent_pixels_become_white() {
        let img = image::RgbaImage::from_fn(3, 1, |x, _| match x {
            0 => image::Rgba([0, 0, 0, 0]),       // transparent, black underneath
            1 => image::Rgba([10, 20, 30, 255]),  // opaque
            _ => image::Rgba([0, 0, 0, 128]),     // half transparent black
        });
        let out = flatten_on_white(image::DynamicImage::ImageRgba8(img));
        assert_eq!(out.get_pixel(0, 0).0, [255, 255, 255]);
        assert_eq!(out.get_pixel(1, 0).0, [10, 20, 30]);
        assert_eq!(out.get_pixel(2, 0).0, [127, 127, 127]);
    }
}

/// Resize to exactly `w x h` (Lanczos).
pub fn resize(img: &RgbImage, w: u32, h: u32) -> RgbImage {
    if img.width() == w && img.height() == h {
        return img.clone();
    }
    image::imageops::resize(img, w, h, FilterType::Lanczos3)
}

/// Scale to cover `w x h`, then center crop.
pub fn resize_cover_crop(img: &RgbImage, w: u32, h: u32) -> RgbImage {
    let scale = (w as f64 / img.width() as f64).max(h as f64 / img.height() as f64);
    let sw = ((img.width() as f64 * scale).round() as u32).max(w);
    let sh = ((img.height() as f64 * scale).round() as u32).max(h);
    let r = resize(img, sw, sh);
    image::imageops::crop_imm(&r, (sw - w) / 2, (sh - h) / 2, w, h).to_image()
}

/// `[1, H, W, 3]` f32 tensor in `[-1, 1]`.
pub fn to_tensor(img: &RgbImage) -> HostTensor {
    let data: Vec<f32> = img.as_raw().iter().map(|v| *v as f32 / 127.5 - 1.0).collect();
    HostTensor::from_f32(&data, &[1, img.height() as usize, img.width() as usize, 3])
}

/// Convert an `[N, H, W, 3]` (or `[H, W, 3]`) tensor in `[-1, 1]` into images.
pub fn from_tensor(t: &HostTensor) -> Vec<RgbImage> {
    let s = t.shape();
    let (n, h, w) = if s.len() == 4 { (s[0], s[1], s[2]) } else { (1, s[0], s[1]) };
    let c = *s.last().unwrap();
    let f = t.to_f32_vec();
    (0..n)
        .map(|i| {
            let mut out = Vec::with_capacity(h * w * 3);
            for p in 0..h * w {
                for ch in 0..3 {
                    let v = f[(i * h * w + p) * c + ch.min(c - 1)];
                    out.push(((v.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8);
                }
            }
            RgbImage::from_raw(w as u32, h as u32, out).unwrap()
        })
        .collect()
}

pub fn encode_png(img: &RgbImage) -> Result<Vec<u8>> {
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png)?;
    Ok(out.into_inner())
}
