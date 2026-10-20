//! Inline image display for iTerm2 and kitty-graphics terminals (kitty, Ghostty, WezTerm).

use std::io::Write;

use anyhow::Result;
use base64::Engine;

use crate::TerminalProtocol;

fn detect() -> Option<TerminalProtocol> {
    let term_program = std::env::var("TERM_PROGRAM").unwrap_or_default();
    if term_program == "iTerm.app" || term_program == "WezTerm" {
        return Some(TerminalProtocol::Iterm2);
    }
    let term = std::env::var("TERM").unwrap_or_default();
    if std::env::var_os("KITTY_WINDOW_ID").is_some() || term.contains("kitty") || term_program.eq_ignore_ascii_case("ghostty") || term.contains("ghostty") {
        return Some(TerminalProtocol::Kitty);
    }
    None
}

pub fn supported(p: TerminalProtocol) -> bool {
    match p {
        TerminalProtocol::Auto => detect().is_some(),
        _ => true,
    }
}

pub fn show(png: &[u8], p: TerminalProtocol) -> Result<()> {
    let p = match p {
        TerminalProtocol::Auto => detect().unwrap_or(TerminalProtocol::Iterm2),
        other => other,
    };
    let b64 = base64::engine::general_purpose::STANDARD.encode(png);
    let mut out = std::io::stdout().lock();
    match p {
        TerminalProtocol::Iterm2 | TerminalProtocol::Auto => {
            write!(out, "\x1b]1337;File=inline=1;size={};preserveAspectRatio=1:{}\x07\n", png.len(), b64)?;
        }
        TerminalProtocol::Kitty => {
            let chunks: Vec<&[u8]> = b64.as_bytes().chunks(4096).collect();
            for (i, c) in chunks.iter().enumerate() {
                let more = if i + 1 < chunks.len() { 1 } else { 0 };
                if i == 0 {
                    write!(out, "\x1b_Ga=T,f=100,m={more};")?;
                } else {
                    write!(out, "\x1b_Gm={more};")?;
                }
                out.write_all(c)?;
                write!(out, "\x1b\\")?;
            }
            writeln!(out)?;
        }
    }
    out.flush()?;
    Ok(())
}
