//! Minimal CPU brand / cache-size introspection for the startup banner.
//!
//! On x86/x86_64 we read the processor brand string directly via the
//! CPUID instruction (EAX = 0x8000_0002/3/4). This works identically on
//! Windows and Linux, with no extra dependency and no process spawn.
//!
//! On every other architecture we return a conservative fallback so the
//! startup banner still works.

/// Returns the processor brand string (e.g. `"Intel(R) Core(TM) i5-9300H
/// CPU @ 2.40GHz"`), trimmed of repeated spaces and trailing NUL bytes.
///
/// Falls back to `"Unknown CPU"` if the CPUID brand leaves are not
/// supported (which is vanishingly rare on modern x86_64 parts).
pub fn cpu_brand() -> String {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        // Probe extended CPUID support first: EAX=0x80000000 returns the
        // highest extended leaf in EAX. Rust promotes `__cpuid` to a safe
        // intrinsic on edition 2024 when the target actually has CPUID
        // (which every x86/x86_64 host since ~1998 has).
        let max_ext = core::arch::x86_64::__cpuid(0x8000_0000).eax;
        if max_ext < 0x8000_0004 {
            return "Unknown CPU".to_string();
        }

        let mut bytes = [0u8; 48];
        for (i, leaf) in [0x8000_0002u32, 0x8000_0003, 0x8000_0004].iter().enumerate() {
            let r = core::arch::x86_64::__cpuid(*leaf);
            let base = i * 16;
            bytes[base..base + 4].copy_from_slice(&r.eax.to_le_bytes());
            bytes[base + 4..base + 8].copy_from_slice(&r.ebx.to_le_bytes());
            bytes[base + 8..base + 12].copy_from_slice(&r.ecx.to_le_bytes());
            bytes[base + 12..base + 16].copy_from_slice(&r.edx.to_le_bytes());
        }
        let raw = String::from_utf8_lossy(&bytes)
            .trim_end_matches('\0')
            .trim()
            .to_string();
        // Collapse runs of spaces so the banner stays one line.
        raw.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        "Unknown CPU".to_string()
    }
}

/// Returns a compact CPU label derived from [`cpu_brand`], suitable for
/// the column header of a summary table. Strips the common Intel / AMD
/// marketing prefixes and the trailing frequency suffix.
///
/// Examples:
/// - `"Intel(R) Core(TM) i5-9300HF CPU @ 2.40GHz"` → `"i5-9300HF"`
/// - `"Intel(R) Core(TM) i5-13450HX"` → `"i5-13450HX"`
/// - `"AMD Ryzen 9 5950X 16-Core Processor"` → `"Ryzen 9 5950X"`
///
/// Falls back to the full brand string when no well-known prefix matches.
pub fn cpu_short() -> String {
    let b = cpu_brand();
    let mut s = b.as_str();
    for prefix in ["Intel(R) Core(TM) ", "Intel(R) Xeon(R) ", "AMD "] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest;
            break;
        }
    }
    for suffix_sep in [" CPU @", " @"] {
        if let Some(i) = s.find(suffix_sep) {
            s = &s[..i];
        }
    }
    // Drop a trailing "NN-Core Processor" tail (AMD).
    if let Some(i) = s.find(" Processor") {
        s = &s[..i];
    }
    if let Some(i) = s.rfind(" ") {
        // If the last token matches "<N>-Core", drop it (AMD decorations).
        let (head, tail) = s.split_at(i);
        if tail.trim_start().contains("-Core") {
            s = head;
        }
    }
    s.trim().to_string()
}

/// Snapshot of the hardware fields shown in the startup banner.
#[derive(Clone, Debug)]
pub struct CpuInfo {
    pub brand: String,
    /// Compact model label — see [`cpu_short`].
    pub short: String,
    pub cores: usize,
    pub threads: usize,
    pub l3_mb: usize,
}

/// Detects the CPU brand, physical/logical core counts and L3 size (MiB).
pub fn detect() -> CpuInfo {
    CpuInfo {
        brand: cpu_brand(),
        short: cpu_short(),
        cores: num_cpus::get_physical(),
        threads: num_cpus::get(),
        l3_mb: cache_size::l3_cache_size().unwrap_or(8 << 20) >> 20,
    }
}

#[cfg(test)]
mod tests {
    use super::cpu_brand;

    #[test]
    fn cpu_brand_is_non_empty_on_x86() {
        let b = cpu_brand();
        assert!(!b.is_empty(), "cpu_brand() returned an empty string");
        // The x86 CPUID brand string is always ≤ 48 bytes.
        assert!(b.len() <= 48, "brand string suspiciously long: {b:?}");
    }
}
