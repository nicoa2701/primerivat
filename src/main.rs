use rivat3::baseline;
use rivat3::baseline::s2::s2;
use rivat3::baseline::s3::s3;
use rivat3::dr;
use rivat3::math::{icbrt, isqrt};
use rivat3::sieve::{extract_primes, lucy_hedgehog_with_phi, lucy_phi_early_stop_profiled, pi_at};
use std::env;
use std::io::Write;
use std::time::Instant;

/// Formats a u128 with a space as thousands separator (e.g. 50_847_534 → "50 847 534").
fn fmt_thousands(n: u128) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let rem = bytes.len() % 3;
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (i % 3 == rem) {
            out.push(' ');
        }
        out.push(b as char);
    }
    out
}

/// Formats a Duration with appropriate unit.
fn fmt_elapsed(d: std::time::Duration) -> String {
    if d.as_secs() >= 60 {
        format!("{:.2} min", d.as_secs_f64() / 60.0)
    } else if d.as_secs() >= 1 {
        format!("{:.3} s", d.as_secs_f64())
    } else if d.as_millis() >= 1 {
        format!("{} ms", d.as_millis())
    } else {
        format!("{} µs", d.as_micros())
    }
}

fn fmt_elapsed_seconds(d: std::time::Duration) -> String {
    format!("{:.3} s", d.as_secs_f64())
}

/// Parses a string as u128, accepting:
/// - Plain integers:       "1000000", "1_000_000"
/// - Scientific notation:  "1e6", "1e12", "1.5e10"  (rounded to nearest integer)
fn parse_x(s: &str) -> Result<u128, String> {
    let cleaned = s.trim().replace('_', "");

    // Try plain integer first
    if let Ok(v) = cleaned.parse::<u128>() {
        return Ok(v);
    }

    // Try scientific notation via f64
    match cleaned.parse::<f64>() {
        Ok(f) if f >= 0.0 && f <= u128::MAX as f64 => Ok(f.round() as u128),
        Ok(f) => Err(format!("{} is out of range for u128", f)),
        Err(_) => Err(format!(
            "'{}' is not a valid integer or scientific notation",
            s
        )),
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn parse_threads(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(v) if v >= 1 => Ok(v),
        Ok(_) => Err("thread count must be >= 1".to_string()),
        Err(_) => Err(format!("'{}' is not a valid thread count", s)),
    }
}

fn parse_nt_batch_job(s: &str) -> Result<NtBatchJob, String> {
    let (x_raw, threads_raw) = s
        .split_once(',')
        .ok_or_else(|| format!("'{s}' must use the form <x,threads> after -nt"))?;
    let label = x_raw.trim();
    if label.is_empty() {
        return Err("missing x before comma in -nt <x,threads>".to_string());
    }
    let x = parse_x(label)?;
    let threads = parse_threads(threads_raw.trim())?;
    Ok(NtBatchJob {
        label: label.to_string(),
        x,
        threads,
    })
}

fn parse_hard_leaf_term_max(s: &str) -> Result<u128, String> {
    match s.parse::<u128>() {
        Ok(v) if v >= 2 => Ok(v),
        Ok(_) => Err("hard-leaf-term-max must be >= 2".to_string()),
        Err(_) => Err(format!("'{}' is not a valid hard-leaf-term-max", s)),
    }
}

fn parse_easy_leaf_term_max(s: &str) -> Result<u128, String> {
    match s.parse::<u128>() {
        Ok(v) if v >= 1 => Ok(v),
        Ok(_) => Err("easy-leaf-term-max must be >= 1".to_string()),
        Err(_) => Err(format!("'{}' is not a valid easy-leaf-term-max", s)),
    }
}

fn parse_relative_width(flag: &str, s: &str) -> Result<u128, String> {
    match s.parse::<u128>() {
        Ok(v) if v >= 1 => Ok(v),
        Ok(_) => Err(format!("{flag} must be >= 1")),
        Err(_) => Err(format!("'{}' is not a valid value for {}", s, flag)),
    }
}

fn parse_alpha(s: &str) -> Result<f64, String> {
    match s.parse::<f64>() {
        Ok(v) if v.is_finite() && (1.0..=2.0).contains(&v) => Ok(v),
        Ok(_) => Err("alpha must be between 1 and 2 (inclusive)".to_string()),
        Err(_) => Err(format!("'{}' is not a valid alpha value", s)),
    }
}

const ALPHA_INTERMEDIATE_MAX_X: u128 = 1_000_000_000_000_000; // 1e15

fn alpha_is_canonical(alpha: f64) -> bool {
    (alpha - 1.0).abs() < 1e-9 || (alpha - 2.0).abs() < 1e-9
}

fn mode_reference_x(mode: &Mode) -> Option<u128> {
    match mode {
        Mode::Normal { x, .. }
        | Mode::Profile { x }
        | Mode::DrProfile { x }
        | Mode::DrMeisselProfile { x }
        | Mode::DrMeissel2Profile { x }
        | Mode::DrMeissel3Profile { x }
        | Mode::DrMeissel4Profile { x }
        | Mode::DrV3Profile { x }
        | Mode::DrV4Profile { x }
        | Mode::LucyProfile { x }
        | Mode::PhiBackendProfile { x }
        | Mode::DrPhiBackendProfile { x } => Some(*x),
        Mode::Sweep { x_max } => Some(*x_max),
        Mode::NtBatch { jobs } => jobs.iter().map(|j| j.x).max(),
        _ => None,
    }
}

fn validate_alpha_override(alpha: f64, mode: &Mode) -> Result<(), String> {
    if alpha_is_canonical(alpha) {
        return Ok(());
    }
    if let Some(x) = mode_reference_x(mode) {
        if x > ALPHA_INTERMEDIATE_MAX_X {
            return Err(format!(
                "alpha strictement entre 1 et 2 n'est autorisé que pour x ≤ 1e15 (ici x = {}); au-delà, utilisez exactement 1 ou 2",
                x
            ));
        }
    }
    Ok(())
}

fn fmt_delta_usize(current: usize, experimental: usize) -> String {
    match experimental.cmp(&current) {
        std::cmp::Ordering::Greater => format!("+{}", experimental - current),
        std::cmp::Ordering::Less => format!("-{}", current - experimental),
        std::cmp::Ordering::Equal => "0".to_string(),
    }
}

fn fmt_delta_u128(current: u128, experimental: u128) -> String {
    match experimental.cmp(&current) {
        std::cmp::Ordering::Greater => format!("+{}", fmt_thousands(experimental - current)),
        std::cmp::Ordering::Less => format!("-{}", fmt_thousands(current - experimental)),
        std::cmp::Ordering::Equal => "0".to_string(),
    }
}

fn pct(part: std::time::Duration, total: std::time::Duration) -> f64 {
    if total.is_zero() {
        0.0
    } else {
        100.0 * part.as_secs_f64() / total.as_secs_f64()
    }
}

fn estimate_baseline_memory_mib(x: u128) -> u128 {
    let z = isqrt(x) as u128;
    let half_z = (z + 1) / 2;
    let prime_estimate = if z >= 17 {
        let zf = z as f64;
        (zf / zf.ln()).round() as u128
    } else {
        z
    };
    let base_bytes = (half_z + 1 + z + 1 + prime_estimate) * 8;
    let estimated_bytes = base_bytes * 17 / 10;
    estimated_bytes.div_ceil(1024 * 1024)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NtBatchJob {
    label: String,
    x: u128,
    threads: usize,
}

struct Cli {
    mode: Mode,
    threads: usize,
    hard_leaf_term_max: u128,
    easy_leaf_term_max: u128,
    experimental_mode: ExperimentalMode,
    alpha_override: Option<f64>,
}

enum Mode {
    Normal { label: String, x: u128 },
    NtBatch { jobs: Vec<NtBatchJob> },
    Profile { x: u128 },
    DrProfile { x: u128 },
    DrMeisselProfile { x: u128 },
    DrMeissel2Profile { x: u128 },
    DrMeissel3Profile { x: u128 },
    DrMeissel4Profile { x: u128 },
    DrV3Profile { x: u128 },
    DrV4Profile { x: u128 },
    LucyProfile { x: u128 },
    DrVsBaselineGrid,
    PhiBackendGrid,
    PhiBackendProfile { x: u128 },
    DrPhiBackendProfile { x: u128 },
    Sweep { x_max: u128 },
    CandidateGrid,
    CandidateSearch,
    CandidateSearchDense,
    CandidateFloorSearch,
    CandidateFamilyCompare,
    CandidateBandSearch,
    PhaseCEasyBandGrid,
    PhaseCEasyCompare,
    PhaseCEasySearch,
    PhaseCEasyCompareBands,
    PhaseCHardGrid,
    PhaseCHardSearch,
    PhaseCHardCompareBands,
    PhaseCPackageGrid,
    PhaseCPackageSearch,
    PhaseCLinkedPackageCompare,
    PhaseCLinkedGrid,
    PhaseCLinkedSearch,
    PhaseCLinkedCandidateCompare,
    PhaseCReferenceCompareDense,
    PhaseCBoundaryPackageCompare,
    PhaseCBoundarySearch,
    PhaseCBoundaryCandidateCompare,
    PhaseCBoundaryLocalSearch,
    PhaseCBufferedBoundaryCompare,
    PhaseCBufferedBoundarySearch,
    PhaseCQuotientWindowCompare,
    PhaseCQuotientWindowSearch,
    PhaseCQuotientWindowShiftedSearch,
    PhaseCBoundaryQuotientGuardCompare,
    PhaseCBoundaryQuotientGuardSearch,
    PhaseCBoundaryRelativeQuotientBandCompare,
    PhaseCBoundaryRelativeQuotientBandSearch,
    PhaseCBoundaryRelativeQuotientStepBandCompare,
    PhaseCBoundaryRelativeQuotientStepBandSearch,
    PhaseCStepBandLocalSearch,
    PhaseCBoundaryVsRelativeQuotientStepDense,
    PhaseCEasySpecializedGrid,
    PhaseCEasySpecializedCompare,
    PhaseCOrdinarySpecializedGrid,
    PhaseCOrdinarySpecializedCompare,
    PhaseCOrdinaryRelativeQuotientGrid,
    PhaseCOrdinaryRelativeQuotientCompare,
    PhaseCOrdinaryRelativeQuotientVsSpecialized,
    PhaseCOrdinaryRelativeQuotientSearch,
    PostPlateauOrdinaryShoulderGrid,
    PostPlateauOrdinaryShoulderSearch,
    PostPlateauOrdinaryEnvelopeGrid,
    PostPlateauOrdinaryEnvelopeSearch,
    PostPlateauOrdinaryEnvelopeVsShoulder,
    PostPlateauOrdinaryHierarchyGrid,
    PostPlateauOrdinaryHierarchyVsEnvelope,
    PostPlateauOrdinaryAssemblyGrid,
    PostPlateauOrdinaryAssemblyVsHierarchy,
    PostPlateauOrdinaryQuasiLiteratureGrid,
    PostPlateauOrdinaryQuasiLiteratureVsAssembly,
    PostPlateauOrdinaryQuasiLiteratureVsAssemblyDense,
    PostPlateauOrdinaryDrLikeGrid,
    PostPlateauOrdinaryDrLikeVsQuasiLiterature,
    PostPlateauTriptychCompare,
    PhaseCHardSpecializedGrid,
    PhaseCHardSpecializedCompare,
    PhaseCBoundaryRelativeQuotientStepBridgeCompare,
    PhaseCBoundaryRelativeQuotientStepBridgeSearch,
    PhaseCBoundaryVsRelativeQuotientStepBridgeDense,
    PhaseCStepBandVsStepBridgeDense,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExperimentalMode {
    None,
    CandidateEasyRelativeToHard,
    CandidateEasyTermBand,
    PhaseCEasyTermBand,
    PhaseCPackage,
    PhaseCLinkedPackage,
    EasyRelativeToHard { width: u128 },
    HardRelativeToEasy { width: u128 },
}

fn parse_cli(args: &[String]) -> Result<Cli, String> {
    let mut threads = default_threads();
    let mut nt_batch_jobs = Vec::new();
    let mut hard_leaf_term_max = rivat3::parameters::Parameters::DEFAULT_HARD_LEAF_TERM_MAX;
    let mut easy_leaf_term_max = rivat3::parameters::Parameters::DEFAULT_EASY_LEAF_TERM_VALUE;
    let mut experimental_mode = ExperimentalMode::None;
    let mut alpha_override: Option<f64> = None;
    let mut _used_t_flag = false;
    let mut _used_non_t_option = false;
    let mut mode: Option<Mode> = None;
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "-t" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "missing value after -t".to_string())?;
                threads = parse_threads(value)?;
                _used_t_flag = true;
                i += 2;
            }
            "-nt" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| "missing value after -nt".to_string())?;
                _used_non_t_option = true;
                if value.contains(',') {
                    nt_batch_jobs.push(parse_nt_batch_job(value)?);
                } else {
                    threads = parse_threads(value)?;
                }
                i += 2;
            }
            "-a" | "--alpha" => {
                let flag = args[i].clone();
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| format!("missing value after {}", flag))?;
                match parse_alpha(value) {
                    Ok(v) => alpha_override = Some(v),
                    Err(e) => eprintln!(
                        "Warning: {} — alpha sera déterminé automatiquement",
                        e
                    ),
                }
                _used_non_t_option = true;
                i += 2;
            }
            "--hard-leaf-term-max" => {
                let value = args
                    .get(i + 1)
                    .ok_or("missing value after --hard-leaf-term-max")?;
                hard_leaf_term_max = parse_hard_leaf_term_max(value)?;
                _used_non_t_option = true;
                i += 2;
            }
            "--easy-leaf-term-max" => {
                let value = args
                    .get(i + 1)
                    .ok_or("missing value after --easy-leaf-term-max")?;
                easy_leaf_term_max = parse_easy_leaf_term_max(value)?;
                _used_non_t_option = true;
                i += 2;
            }
            "--experimental-easy-relative-to-hard" => {
                let value = args
                    .get(i + 1)
                    .ok_or("missing value after --experimental-easy-relative-to-hard")?;
                _used_non_t_option = true;
                if !matches!(experimental_mode, ExperimentalMode::None) {
                    return Err(
                        "experimental easy/hard relative modes are mutually exclusive".to_string(),
                    );
                }
                experimental_mode = ExperimentalMode::EasyRelativeToHard {
                    width: parse_relative_width("--experimental-easy-relative-to-hard", value)?,
                };
                i += 2;
            }
            "--candidate-easy-relative-to-hard" => {
                _used_non_t_option = true;
                if !matches!(experimental_mode, ExperimentalMode::None) {
                    return Err(
                        "experimental easy/hard relative modes are mutually exclusive".to_string(),
                    );
                }
                experimental_mode = ExperimentalMode::CandidateEasyRelativeToHard;
                i += 1;
            }
            "--candidate-easy-term-band" => {
                _used_non_t_option = true;
                if !matches!(experimental_mode, ExperimentalMode::None) {
                    return Err(
                        "experimental easy/hard relative modes are mutually exclusive".to_string(),
                    );
                }
                experimental_mode = ExperimentalMode::CandidateEasyTermBand;
                i += 1;
            }
            "--phase-c-easy-term-band" => {
                _used_non_t_option = true;
                if !matches!(experimental_mode, ExperimentalMode::None) {
                    return Err(
                        "experimental easy/hard relative modes are mutually exclusive".to_string(),
                    );
                }
                experimental_mode = ExperimentalMode::PhaseCEasyTermBand;
                i += 1;
            }
            "--phase-c-package" => {
                _used_non_t_option = true;
                if !matches!(experimental_mode, ExperimentalMode::None) {
                    return Err(
                        "experimental easy/hard relative modes are mutually exclusive".to_string(),
                    );
                }
                experimental_mode = ExperimentalMode::PhaseCPackage;
                i += 1;
            }
            "--phase-c-linked-package" => {
                _used_non_t_option = true;
                if !matches!(experimental_mode, ExperimentalMode::None) {
                    return Err(
                        "experimental easy/hard relative modes are mutually exclusive".to_string(),
                    );
                }
                experimental_mode = ExperimentalMode::PhaseCLinkedPackage;
                i += 1;
            }
            "--experimental-hard-relative-to-easy" => {
                let value = args
                    .get(i + 1)
                    .ok_or("missing value after --experimental-hard-relative-to-easy")?;
                _used_non_t_option = true;
                if !matches!(experimental_mode, ExperimentalMode::None) {
                    return Err(
                        "experimental easy/hard relative modes are mutually exclusive".to_string(),
                    );
                }
                experimental_mode = ExperimentalMode::HardRelativeToEasy {
                    width: parse_relative_width("--experimental-hard-relative-to-easy", value)?,
                };
                i += 2;
            }
            "--profile" => {
                let take_value = args.get(i + 1).filter(|s| !s.starts_with('-'));
                let x = parse_x(take_value.map(|s| s.as_str()).unwrap_or("1000000000000"))?;
                mode = Some(Mode::Profile { x });
                i += if take_value.is_some() { 2 } else { 1 };
            }
            "--dr-profile" => {
                let take_value = args.get(i + 1).filter(|s| !s.starts_with('-'));
                let x = parse_x(take_value.map(|s| s.as_str()).unwrap_or("1000000000000"))?;
                mode = Some(Mode::DrProfile { x });
                i += if take_value.is_some() { 2 } else { 1 };
            }
            "--lucy-profile" => {
                let take_value = args.get(i + 1).filter(|s| !s.starts_with('-'));
                let x = parse_x(take_value.map(|s| s.as_str()).unwrap_or("1000000000000"))?;
                mode = Some(Mode::LucyProfile { x });
                i += if take_value.is_some() { 2 } else { 1 };
            }
            "--dr-meissel" => {
                let take_value = args.get(i + 1).filter(|s| !s.starts_with('-'));
                let x = parse_x(take_value.map(|s| s.as_str()).unwrap_or("1000000000000"))?;
                mode = Some(Mode::DrMeisselProfile { x });
                i += if take_value.is_some() { 2 } else { 1 };
            }
            "--dr-meissel2" => {
                let take_value = args.get(i + 1).filter(|s| !s.starts_with('-'));
                let x = parse_x(take_value.map(|s| s.as_str()).unwrap_or("1000000000000"))?;
                mode = Some(Mode::DrMeissel2Profile { x });
                i += if take_value.is_some() { 2 } else { 1 };
            }
            "--dr-meissel3" => {
                let take_value = args.get(i + 1).filter(|s| !s.starts_with('-'));
                let x = parse_x(take_value.map(|s| s.as_str()).unwrap_or("1000000000000"))?;
                mode = Some(Mode::DrMeissel3Profile { x });
                i += if take_value.is_some() { 2 } else { 1 };
            }
            "--dr-meissel4" => {
                let take_value = args.get(i + 1).filter(|s| !s.starts_with('-'));
                let x = parse_x(take_value.map(|s| s.as_str()).unwrap_or("1000000000000"))?;
                mode = Some(Mode::DrMeissel4Profile { x });
                i += if take_value.is_some() { 2 } else { 1 };
            }
            "--dr-v3" => {
                let take_value = args.get(i + 1).filter(|s| !s.starts_with('-'));
                let x = parse_x(take_value.map(|s| s.as_str()).unwrap_or("1000000000000"))?;
                mode = Some(Mode::DrV3Profile { x });
                i += if take_value.is_some() { 2 } else { 1 };
            }
            "--dr-v4" => {
                let take_value = args.get(i + 1).filter(|s| !s.starts_with('-'));
                let x = parse_x(take_value.map(|s| s.as_str()).unwrap_or("1000000000000"))?;
                mode = Some(Mode::DrV4Profile { x });
                i += if take_value.is_some() { 2 } else { 1 };
            }
            "--dr-vs-baseline-grid" => {
                mode = Some(Mode::DrVsBaselineGrid);
                i += 1;
            }
            "--phi-backend-grid" => {
                mode = Some(Mode::PhiBackendGrid);
                i += 1;
            }
            "--phi-backend-profile" => {
                let take_value = args.get(i + 1).filter(|s| !s.starts_with('-'));
                let x = parse_x(take_value.map(|s| s.as_str()).unwrap_or("1000000000000"))?;
                mode = Some(Mode::PhiBackendProfile { x });
                i += if take_value.is_some() { 2 } else { 1 };
            }
            "--dr-phi-backend-profile" => {
                let take_value = args.get(i + 1).filter(|s| !s.starts_with('-'));
                let x = parse_x(take_value.map(|s| s.as_str()).unwrap_or("1000000000000"))?;
                mode = Some(Mode::DrPhiBackendProfile { x });
                i += if take_value.is_some() { 2 } else { 1 };
            }
            "--sweep" => {
                let take_value = args.get(i + 1).filter(|s| !s.starts_with('-'));
                let x_max = take_value
                    .map(|s| parse_x(s))
                    .transpose()?
                    .unwrap_or(1_000_000_000_000);
                mode = Some(Mode::Sweep { x_max });
                i += if take_value.is_some() { 2 } else { 1 };
            }
            "--candidate-grid" => {
                mode = Some(Mode::CandidateGrid);
                i += 1;
            }
            "--candidate-search" => {
                mode = Some(Mode::CandidateSearch);
                i += 1;
            }
            "--candidate-search-dense" => {
                mode = Some(Mode::CandidateSearchDense);
                i += 1;
            }
            "--candidate-floor-search" => {
                mode = Some(Mode::CandidateFloorSearch);
                i += 1;
            }
            "--candidate-family-compare" => {
                mode = Some(Mode::CandidateFamilyCompare);
                i += 1;
            }
            "--candidate-band-search" => {
                mode = Some(Mode::CandidateBandSearch);
                i += 1;
            }
            "--phase-c-easy-band-grid" => {
                mode = Some(Mode::PhaseCEasyBandGrid);
                i += 1;
            }
            "--phase-c-easy-compare" => {
                mode = Some(Mode::PhaseCEasyCompare);
                i += 1;
            }
            "--phase-c-easy-search" => {
                mode = Some(Mode::PhaseCEasySearch);
                i += 1;
            }
            "--phase-c-easy-compare-bands" => {
                mode = Some(Mode::PhaseCEasyCompareBands);
                i += 1;
            }
            "--phase-c-hard-grid" => {
                mode = Some(Mode::PhaseCHardGrid);
                i += 1;
            }
            "--phase-c-hard-search" => {
                mode = Some(Mode::PhaseCHardSearch);
                i += 1;
            }
            "--phase-c-hard-compare-bands" => {
                mode = Some(Mode::PhaseCHardCompareBands);
                i += 1;
            }
            "--phase-c-package-grid" => {
                mode = Some(Mode::PhaseCPackageGrid);
                i += 1;
            }
            "--phase-c-package-search" => {
                mode = Some(Mode::PhaseCPackageSearch);
                i += 1;
            }
            "--phase-c-linked-package-compare" => {
                mode = Some(Mode::PhaseCLinkedPackageCompare);
                i += 1;
            }
            "--phase-c-linked-grid" => {
                mode = Some(Mode::PhaseCLinkedGrid);
                i += 1;
            }
            "--phase-c-linked-search" => {
                mode = Some(Mode::PhaseCLinkedSearch);
                i += 1;
            }
            "--phase-c-linked-candidate-compare" => {
                mode = Some(Mode::PhaseCLinkedCandidateCompare);
                i += 1;
            }
            "--phase-c-reference-compare-dense" => {
                mode = Some(Mode::PhaseCReferenceCompareDense);
                i += 1;
            }
            "--phase-c-boundary-package-compare" => {
                mode = Some(Mode::PhaseCBoundaryPackageCompare);
                i += 1;
            }
            "--phase-c-boundary-search" => {
                mode = Some(Mode::PhaseCBoundarySearch);
                i += 1;
            }
            "--phase-c-boundary-candidate-compare" => {
                mode = Some(Mode::PhaseCBoundaryCandidateCompare);
                i += 1;
            }
            "--phase-c-boundary-local-search" => {
                mode = Some(Mode::PhaseCBoundaryLocalSearch);
                i += 1;
            }
            "--phase-c-buffered-boundary-compare" => {
                mode = Some(Mode::PhaseCBufferedBoundaryCompare);
                i += 1;
            }
            "--phase-c-buffered-boundary-search" => {
                mode = Some(Mode::PhaseCBufferedBoundarySearch);
                i += 1;
            }
            "--phase-c-quotient-window-compare" => {
                mode = Some(Mode::PhaseCQuotientWindowCompare);
                i += 1;
            }
            "--phase-c-quotient-window-search" => {
                mode = Some(Mode::PhaseCQuotientWindowSearch);
                i += 1;
            }
            "--phase-c-quotient-window-shifted-search" => {
                mode = Some(Mode::PhaseCQuotientWindowShiftedSearch);
                i += 1;
            }
            "--phase-c-boundary-quotient-guard-compare" => {
                mode = Some(Mode::PhaseCBoundaryQuotientGuardCompare);
                i += 1;
            }
            "--phase-c-boundary-quotient-guard-search" => {
                mode = Some(Mode::PhaseCBoundaryQuotientGuardSearch);
                i += 1;
            }
            "--phase-c-boundary-relative-quotient-band-compare" => {
                mode = Some(Mode::PhaseCBoundaryRelativeQuotientBandCompare);
                i += 1;
            }
            "--phase-c-boundary-relative-quotient-band-search" => {
                mode = Some(Mode::PhaseCBoundaryRelativeQuotientBandSearch);
                i += 1;
            }
            "--phase-c-boundary-relative-quotient-step-band-compare" => {
                mode = Some(Mode::PhaseCBoundaryRelativeQuotientStepBandCompare);
                i += 1;
            }
            "--phase-c-boundary-relative-quotient-step-band-search" => {
                mode = Some(Mode::PhaseCBoundaryRelativeQuotientStepBandSearch);
                i += 1;
            }
            "--phase-c-step-band-local-search" => {
                mode = Some(Mode::PhaseCStepBandLocalSearch);
                i += 1;
            }
            "--phase-c-boundary-vs-relative-quotient-step-dense" => {
                mode = Some(Mode::PhaseCBoundaryVsRelativeQuotientStepDense);
                i += 1;
            }
            "--phase-c-easy-specialized-grid" => {
                mode = Some(Mode::PhaseCEasySpecializedGrid);
                i += 1;
            }
            "--phase-c-easy-specialized-compare" => {
                mode = Some(Mode::PhaseCEasySpecializedCompare);
                i += 1;
            }
            "--phase-c-hard-specialized-grid" => {
                mode = Some(Mode::PhaseCHardSpecializedGrid);
                i += 1;
            }
            "--phase-c-hard-specialized-compare" => {
                mode = Some(Mode::PhaseCHardSpecializedCompare);
                i += 1;
            }
            "--phase-c-ordinary-specialized-grid" => {
                mode = Some(Mode::PhaseCOrdinarySpecializedGrid);
                i += 1;
            }
            "--phase-c-ordinary-specialized-compare" => {
                mode = Some(Mode::PhaseCOrdinarySpecializedCompare);
                i += 1;
            }
            "--phase-c-ordinary-relative-quotient-grid" => {
                mode = Some(Mode::PhaseCOrdinaryRelativeQuotientGrid);
                i += 1;
            }
            "--phase-c-ordinary-relative-quotient-compare" => {
                mode = Some(Mode::PhaseCOrdinaryRelativeQuotientCompare);
                i += 1;
            }
            "--phase-c-ordinary-relative-quotient-vs-specialized" => {
                mode = Some(Mode::PhaseCOrdinaryRelativeQuotientVsSpecialized);
                i += 1;
            }
            "--phase-c-ordinary-relative-quotient-search" => {
                mode = Some(Mode::PhaseCOrdinaryRelativeQuotientSearch);
                i += 1;
            }
            "--post-plateau-ordinary-shoulder-grid" => {
                mode = Some(Mode::PostPlateauOrdinaryShoulderGrid);
                i += 1;
            }
            "--post-plateau-ordinary-shoulder-search" => {
                mode = Some(Mode::PostPlateauOrdinaryShoulderSearch);
                i += 1;
            }
            "--post-plateau-ordinary-envelope-grid" => {
                mode = Some(Mode::PostPlateauOrdinaryEnvelopeGrid);
                i += 1;
            }
            "--post-plateau-ordinary-envelope-search" => {
                mode = Some(Mode::PostPlateauOrdinaryEnvelopeSearch);
                i += 1;
            }
            "--post-plateau-ordinary-envelope-vs-shoulder" => {
                mode = Some(Mode::PostPlateauOrdinaryEnvelopeVsShoulder);
                i += 1;
            }
            "--post-plateau-ordinary-hierarchy-grid" => {
                mode = Some(Mode::PostPlateauOrdinaryHierarchyGrid);
                i += 1;
            }
            "--post-plateau-ordinary-hierarchy-vs-envelope" => {
                mode = Some(Mode::PostPlateauOrdinaryHierarchyVsEnvelope);
                i += 1;
            }
            "--post-plateau-ordinary-assembly-grid" => {
                mode = Some(Mode::PostPlateauOrdinaryAssemblyGrid);
                i += 1;
            }
            "--post-plateau-ordinary-assembly-vs-hierarchy" => {
                mode = Some(Mode::PostPlateauOrdinaryAssemblyVsHierarchy);
                i += 1;
            }
            "--post-plateau-ordinary-quasi-literature-grid" => {
                mode = Some(Mode::PostPlateauOrdinaryQuasiLiteratureGrid);
                i += 1;
            }
            "--post-plateau-ordinary-quasi-literature-vs-assembly" => {
                mode = Some(Mode::PostPlateauOrdinaryQuasiLiteratureVsAssembly);
                i += 1;
            }
            "--post-plateau-ordinary-quasi-literature-vs-assembly-dense" => {
                mode = Some(Mode::PostPlateauOrdinaryQuasiLiteratureVsAssemblyDense);
                i += 1;
            }
            "--post-plateau-ordinary-dr-like-grid" => {
                mode = Some(Mode::PostPlateauOrdinaryDrLikeGrid);
                i += 1;
            }
            "--post-plateau-ordinary-dr-like-vs-quasi-literature" => {
                mode = Some(Mode::PostPlateauOrdinaryDrLikeVsQuasiLiterature);
                i += 1;
            }
            "--post-plateau-triptych-compare" => {
                mode = Some(Mode::PostPlateauTriptychCompare);
                i += 1;
            }
            "--phase-c-boundary-relative-quotient-step-bridge-compare" => {
                mode = Some(Mode::PhaseCBoundaryRelativeQuotientStepBridgeCompare);
                i += 1;
            }
            "--phase-c-boundary-relative-quotient-step-bridge-search" => {
                mode = Some(Mode::PhaseCBoundaryRelativeQuotientStepBridgeSearch);
                i += 1;
            }
            "--phase-c-boundary-vs-relative-quotient-step-bridge-dense" => {
                mode = Some(Mode::PhaseCBoundaryVsRelativeQuotientStepBridgeDense);
                i += 1;
            }
            "--phase-c-step-band-vs-step-bridge-dense" => {
                mode = Some(Mode::PhaseCStepBandVsStepBridgeDense);
                i += 1;
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option: {}", other));
            }
            other => {
                if !nt_batch_jobs.is_empty() {
                    return Err(
                        "cannot mix positional x with -nt <x,threads> batch mode".to_string()
                    );
                }
                let x = parse_x(other)?;
                mode = Some(Mode::Normal {
                    label: other.to_string(),
                    x,
                });
                i += 1;
            }
        }
    }

    let mode = if !nt_batch_jobs.is_empty() {
        if mode.is_some() {
            return Err(
                "cannot combine -nt <x,threads> batch mode with another explicit mode".to_string(),
            );
        }
        Mode::NtBatch {
            jobs: nt_batch_jobs,
        }
    } else {
        mode.ok_or_else(|| {
            "missing x or mode (--profile / --dr-profile / --sweep / -nt <x,threads>)".to_string()
        })?
    };
    if easy_leaf_term_max > hard_leaf_term_max {
        return Err("easy-leaf-term-max must be <= hard-leaf-term-max".to_string());
    }
    if !matches!(experimental_mode, ExperimentalMode::None)
        && !matches!(
            mode,
            Mode::Profile { .. }
                | Mode::Sweep { .. }
                | Mode::CandidateGrid
                | Mode::CandidateSearch
                | Mode::CandidateSearchDense
                | Mode::CandidateFamilyCompare
                | Mode::PhaseCEasyBandGrid
                | Mode::PhaseCPackageGrid
                | Mode::PhaseCLinkedPackageCompare
                | Mode::PhaseCLinkedCandidateCompare
        )
    {
        return Err(
            "experimental easy/hard relative modes are only available with --profile, --sweep, --candidate-grid or candidate search modes"
                .to_string(),
        );
    }
    Ok(Cli {
        mode,
        threads,
        hard_leaf_term_max,
        easy_leaf_term_max,
        experimental_mode,
        alpha_override,
    })
}

/// Runs π(x) with per-phase timing printed to stdout.
#[allow(non_snake_case)]
fn run_profile(
    x: u128,
    threads: usize,
    hard_leaf_term_max: u128,
    easy_leaf_term_max: u128,
    experimental_mode: ExperimentalMode,
) {
    let (display_hard_leaf_term_max, display_easy_leaf_term_max) = match experimental_mode {
        ExperimentalMode::PhaseCEasyTermBand => (
            rivat3::parameters::DrTuning::PHASE_C_EASY_TERM_BAND_MAX + 1,
            rivat3::parameters::DrTuning::PHASE_C_EASY_TERM_BAND_MAX,
        ),
        ExperimentalMode::PhaseCPackage => (
            rivat3::parameters::DrTuning::PHASE_C_LINKED_EASY_MIN_TERM_FLOOR
                + rivat3::parameters::DrTuning::PHASE_C_LINKED_EASY_WIDTH
                - 1
                + rivat3::parameters::DrTuning::PHASE_C_LINKED_HARD_WIDTH,
            rivat3::parameters::DrTuning::PHASE_C_LINKED_EASY_MIN_TERM_FLOOR
                + rivat3::parameters::DrTuning::PHASE_C_LINKED_EASY_WIDTH
                - 1,
        ),
        ExperimentalMode::PhaseCLinkedPackage => (
            rivat3::parameters::DrTuning::PHASE_C_HARD_TERM_BAND_MAX,
            rivat3::parameters::DrTuning::PHASE_C_EASY_TERM_BAND_MAX,
        ),
        _ => (hard_leaf_term_max, easy_leaf_term_max),
    };
    println!(
        "── Profile  x = {}  |  threads S2 = {}  |  hard max = {}  |  easy max = {} ───────",
        fmt_thousands(x),
        threads,
        display_hard_leaf_term_max,
        display_easy_leaf_term_max
    );

    let z_u128 = isqrt(x);
    let y = icbrt(x);
    let z = z_u128 as usize;

    // Phase 1 — Lucy sieve + φ capture
    let t = Instant::now();
    let (small, large, phi_x_a) = lucy_hedgehog_with_phi(x, y);
    let t_lucy = t.elapsed();

    let a = pi_at(y, x, z, &small, &large) as usize;

    // Phase 2 — extract primes
    let t = Instant::now();
    let primes = extract_primes(&small, z);
    let t_primes = t.elapsed();

    // Phase 3 — S2
    let t = Instant::now();
    let s2_val = s2(x, a, z, &primes, &small, &large, threads);
    let t_s2 = t.elapsed();

    // Phase 4 — S3
    let t = Instant::now();
    let s3_val = s3(x, a, &primes);
    let t_s3 = t.elapsed();

    let result = phi_x_a + a as u128 - 1 - s2_val - s3_val;

    let total = t_lucy + t_primes + t_s2 + t_s3;

    println!(
        "  Lucy sieve     {:>10}  ({:.1}%)",
        fmt_elapsed(t_lucy),
        100.0 * t_lucy.as_secs_f64() / total.as_secs_f64()
    );
    println!(
        "  extract_primes {:>10}  ({:.1}%)",
        fmt_elapsed(t_primes),
        100.0 * t_primes.as_secs_f64() / total.as_secs_f64()
    );
    println!(
        "  S2             {:>10}  ({:.1}%)",
        fmt_elapsed(t_s2),
        100.0 * t_s2.as_secs_f64() / total.as_secs_f64()
    );
    println!(
        "  S3             {:>10}  ({:.1}%)",
        fmt_elapsed(t_s3),
        100.0 * t_s3.as_secs_f64() / total.as_secs_f64()
    );
    println!("  ─────────────────────────────────────────");
    println!(
        "  Total          {:>10}    π({}) = {}",
        fmt_elapsed(total),
        fmt_thousands(x),
        fmt_thousands(result)
    );
    let (dr_analysis, experimental_label, comparison) = match experimental_mode {
        ExperimentalMode::None => (
            dr::analyze_with_term_frontiers(x, hard_leaf_term_max, easy_leaf_term_max),
            None,
            None,
        ),
        ExperimentalMode::CandidateEasyRelativeToHard => {
            let comparison = dr::compare_current_vs_candidate_easy_relative_to_hard(
                x,
                hard_leaf_term_max,
                easy_leaf_term_max,
            );
            (
                comparison.experimental,
                Some(format!(
                    "candidate_easy_relative_to_hard(width={}, floor={})",
                    rivat3::parameters::DrTuning::CANDIDATE_EASY_RELATIVE_TO_HARD_WIDTH,
                    rivat3::parameters::DrTuning::CANDIDATE_EASY_RELATIVE_TO_HARD_MIN_TERM_FLOOR
                )),
                Some(comparison),
            )
        }
        ExperimentalMode::CandidateEasyTermBand => {
            let comparison = dr::compare_current_vs_candidate_easy_term_band(
                x,
                hard_leaf_term_max,
                easy_leaf_term_max,
            );
            (
                comparison.experimental,
                Some(format!(
                    "candidate_easy_term_band(min={}, max={})",
                    rivat3::parameters::DrTuning::CANDIDATE_EASY_TERM_BAND_MIN,
                    rivat3::parameters::DrTuning::CANDIDATE_EASY_TERM_BAND_MAX
                )),
                Some(comparison),
            )
        }
        ExperimentalMode::PhaseCEasyTermBand => {
            let comparison = dr::compare_current_vs_phase_c_easy_term_band(
                x,
                rivat3::parameters::DrTuning::PHASE_C_EASY_TERM_BAND_MAX + 1,
                easy_leaf_term_max,
            );
            (
                comparison.experimental,
                Some(format!(
                    "phase_c_easy_term_band(min={}, max={})",
                    rivat3::parameters::DrTuning::PHASE_C_EASY_TERM_BAND_MIN,
                    rivat3::parameters::DrTuning::PHASE_C_EASY_TERM_BAND_MAX
                )),
                Some(comparison),
            )
        }
        ExperimentalMode::PhaseCPackage => {
            let comparison = dr::compare_current_vs_phase_c_package(x);
            (
                comparison.experimental,
                Some(format!(
                    "phase_c_package_linked(easy_width={}, floor={}, hard_width={})",
                    rivat3::parameters::DrTuning::PHASE_C_LINKED_EASY_WIDTH,
                    rivat3::parameters::DrTuning::PHASE_C_LINKED_EASY_MIN_TERM_FLOOR,
                    rivat3::parameters::DrTuning::PHASE_C_LINKED_HARD_WIDTH
                )),
                Some(comparison),
            )
        }
        ExperimentalMode::PhaseCLinkedPackage => {
            let comparison = dr::compare_phase_c_package_vs_linked_package(x);
            (
                comparison.experimental,
                Some(format!(
                    "phase_c_linked_package(easy_width={}, floor={}, hard_width={})",
                    rivat3::parameters::DrTuning::PHASE_C_LINKED_EASY_WIDTH,
                    rivat3::parameters::DrTuning::PHASE_C_LINKED_EASY_MIN_TERM_FLOOR,
                    rivat3::parameters::DrTuning::PHASE_C_LINKED_HARD_WIDTH
                )),
                Some(comparison),
            )
        }
        ExperimentalMode::EasyRelativeToHard { width } => {
            let comparison = dr::compare_current_vs_experimental_easy_relative_to_hard(
                x,
                hard_leaf_term_max,
                easy_leaf_term_max,
                width,
            );
            (
                comparison.experimental,
                Some(format!("easy_relative_to_hard(width={})", width)),
                Some(comparison),
            )
        }
        ExperimentalMode::HardRelativeToEasy { width } => {
            let comparison = dr::compare_current_vs_experimental_hard_relative_to_easy(
                x,
                hard_leaf_term_max,
                easy_leaf_term_max,
                width,
            );
            (
                comparison.experimental,
                Some(format!("hard_relative_to_easy(width={})", width)),
                Some(comparison),
            )
        }
    };
    println!();
    println!("  DR skeleton");
    if let Some(label) = experimental_label {
        println!("    experimental_mode = {}", label);
    }
    println!("    alpha = {:.3}", dr_analysis.alpha);
    println!(
        "    a = {:>12}  phi(x,a) = {}",
        dr_analysis.a,
        fmt_thousands(dr_analysis.phi_x_a)
    );
    match (
        dr_analysis.easy_candidate_family,
        dr_analysis.easy_candidate_width,
        dr_analysis.easy_candidate_floor,
        dr_analysis.easy_candidate_term_min,
        dr_analysis.easy_candidate_term_max,
    ) {
        (Some("relative_to_hard"), Some(width), Some(floor), _, _) => {
            println!(
                "    easy_candidate = relative_to_hard(width={}, floor={})",
                width, floor
            );
        }
        (Some("term_band"), _, _, Some(min), Some(max)) => {
            println!("    easy_candidate = term_band(min={}, max={})", min, max);
        }
        (Some("phase_c_term_band"), _, _, Some(min), Some(max)) => {
            println!(
                "    easy_candidate = phase_c_term_band(min={}, max={})",
                min, max
            );
        }
        (Some("phase_c_package"), _, _, Some(min), Some(max)) => {
            println!(
                "    phase_c_package = linked easy[{}..={}] hard[{}..={}]",
                min, dr_analysis.easy_leaf_term_max, dr_analysis.hard_leaf_term_min, max
            );
        }
        (Some("phase_c_linked_package"), Some(width), Some(floor), Some(min), Some(max)) => {
            println!(
                "    phase_c_linked_package = easy relative_to_hard(width={}, floor={}) [{}..={}]  hard relative_to_easy(width={})",
                width,
                floor,
                min,
                dr_analysis.easy_leaf_term_max,
                max.saturating_sub(dr_analysis.easy_leaf_term_max)
            );
        }
        _ => {}
    }
    println!(
        "    s1_term_min_exclusive = {}",
        dr_analysis.s1_term_min_exclusive
    );
    println!(
        "    hard_leaf_term_range = [{}..={}]",
        dr_analysis.hard_leaf_term_min, dr_analysis.hard_leaf_term_max
    );
    println!("    hard_rule_kind = {}", dr_analysis.hard_rule_kind);
    if let Some(alpha) = dr_analysis.hard_rule_alpha {
        println!("    hard_rule_alpha = {:.3}", alpha);
    }
    println!(
        "    easy_leaf_term_range = [{}..={}]",
        dr_analysis.easy_leaf_term_min, dr_analysis.easy_leaf_term_max
    );
    println!("    easy_rule_kind = {}", dr_analysis.easy_rule_kind);
    if let Some(alpha) = dr_analysis.easy_rule_alpha {
        println!("    easy_rule_alpha = {:.3}", alpha);
    }
    println!(
        "    active/S1/S2_hard/S2_easy = {}/{}/{}/{}",
        dr_analysis.active_len,
        dr_analysis.s1_len,
        dr_analysis.s2_hard_len,
        dr_analysis.s2_easy_len
    );
    println!(
        "    contrib S1 = {}  S2_hard = {}  S2_easy = {}  S2_trivial = {}",
        fmt_thousands(dr_analysis.contributions.ordinary),
        fmt_thousands(dr_analysis.contributions.hard),
        fmt_thousands(dr_analysis.contributions.easy),
        fmt_thousands(dr_analysis.contributions.trivial)
    );
    let easy_specialized = dr::analyze_easy_specialized(x);
    println!(
        "    easy breakdown: residual={} ({})  transition={} ({})  specialized={} ({})",
        easy_specialized.residual_len,
        fmt_thousands(easy_specialized.residual_sum),
        easy_specialized.transition_len,
        fmt_thousands(easy_specialized.transition_sum),
        easy_specialized.specialized_len,
        fmt_thousands(easy_specialized.specialized_sum),
    );
    if let Some(q_ref) = easy_specialized.q_ref {
        println!(
            "    easy specialized q_ref = {}  q_step = {}",
            fmt_thousands(q_ref),
            easy_specialized.q_step
        );
    }
    let hard_specialized = dr::analyze_hard_specialized(x);
    println!(
        "    hard breakdown: residual={} ({})  transition={} ({})  specialized={} ({})",
        hard_specialized.residual_len,
        fmt_thousands(hard_specialized.residual_sum),
        hard_specialized.transition_len,
        fmt_thousands(hard_specialized.transition_sum),
        hard_specialized.specialized_len,
        fmt_thousands(hard_specialized.specialized_sum),
    );
    if let Some(q_ref) = hard_specialized.q_ref {
        println!(
            "    hard specialized q_ref = {}  q_step = {}",
            fmt_thousands(q_ref),
            hard_specialized.q_step
        );
    }
    let ordinary_specialized = dr::analyze_ordinary_specialized(x);
    let ordinary_post_plateau = dr::analyze_ordinary_post_plateau_profile(x);
    let ordinary_assembly = dr::analyze_ordinary_region_assembly(x);
    let ordinary_quasi_literature = dr::analyze_ordinary_quasi_literature_region(x);
    println!(
        "    ordinary terminal: residual={} ({})  pretr.={} ({})  trans.={} ({})  special.={} ({})",
        ordinary_specialized.residual_len,
        fmt_thousands(ordinary_specialized.residual_sum),
        ordinary_specialized.pretransition_len,
        fmt_thousands(ordinary_specialized.pretransition_sum),
        ordinary_specialized.transition_len,
        fmt_thousands(ordinary_specialized.transition_sum),
        ordinary_specialized.specialized_len,
        fmt_thousands(ordinary_specialized.specialized_sum),
    );
    println!(
        "    post_plateau ordinary: left={} ({})  region={} ({})  right={} ({})",
        ordinary_post_plateau.left_residual_len,
        fmt_thousands(ordinary_post_plateau.left_residual_sum),
        ordinary_post_plateau.region_len,
        fmt_thousands(ordinary_post_plateau.region_sum),
        ordinary_post_plateau.right_residual_len,
        fmt_thousands(ordinary_post_plateau.right_residual_sum),
    );
    println!(
        "    ordinary assembly: left_out={} ({})  left_adj={} ({})  central={} ({})  right_adj={} ({})  right_out={} ({})",
        ordinary_assembly.left_outer_support_len,
        fmt_thousands(ordinary_assembly.left_outer_support_sum),
        ordinary_assembly.left_adjacent_support_len,
        fmt_thousands(ordinary_assembly.left_adjacent_support_sum),
        ordinary_assembly.central_assembly_len,
        fmt_thousands(ordinary_assembly.central_assembly_sum),
        ordinary_assembly.right_adjacent_support_len,
        fmt_thousands(ordinary_assembly.right_adjacent_support_sum),
        ordinary_assembly.right_outer_support_len,
        fmt_thousands(ordinary_assembly.right_outer_support_sum),
    );
    println!(
        "    ordinary quasi-lit: left_out={} ({})  middle={} ({})  right_out={} ({})",
        ordinary_quasi_literature.left_outer_work_len,
        fmt_thousands(ordinary_quasi_literature.left_outer_work_sum),
        ordinary_quasi_literature.middle_work_len,
        fmt_thousands(ordinary_quasi_literature.middle_work_sum),
        ordinary_quasi_literature.right_outer_work_len,
        fmt_thousands(ordinary_quasi_literature.right_outer_work_sum),
    );
    println!(
        "    post_plateau ordinary delta: terminal len={}  region len={}  terminal sum={}  region sum={}",
        fmt_thousands(ordinary_post_plateau.terminal_len as u128),
        fmt_thousands(ordinary_post_plateau.region_len as u128),
        fmt_thousands(ordinary_post_plateau.terminal_sum),
        fmt_thousands(ordinary_post_plateau.region_sum),
    );
    println!(
        "    ordinary assembly delta: central len={}  support len={}  central sum={}  support sum={}",
        fmt_thousands(ordinary_assembly.central_assembly_len as u128),
        fmt_thousands(
            (ordinary_assembly.left_outer_support_len
                + ordinary_assembly.left_adjacent_support_len
                + ordinary_assembly.right_adjacent_support_len
                + ordinary_assembly.right_outer_support_len) as u128
        ),
        fmt_thousands(ordinary_assembly.central_assembly_sum),
        fmt_thousands(
            ordinary_assembly.left_outer_support_sum
                + ordinary_assembly.left_adjacent_support_sum
                + ordinary_assembly.right_adjacent_support_sum
                + ordinary_assembly.right_outer_support_sum
        ),
    );
    println!(
        "    ordinary quasi-lit delta: middle len={}  outer len={}  middle sum={}  outer sum={}",
        fmt_thousands(ordinary_quasi_literature.middle_work_len as u128),
        fmt_thousands(
            (ordinary_quasi_literature.left_outer_work_len
                + ordinary_quasi_literature.right_outer_work_len) as u128
        ),
        fmt_thousands(ordinary_quasi_literature.middle_work_sum),
        fmt_thousands(
            ordinary_quasi_literature.left_outer_work_sum
                + ordinary_quasi_literature.right_outer_work_sum
        ),
    );
    if let Some(q_ref) = ordinary_post_plateau.q_ref {
        println!(
            "    post_plateau ordinary q_ref = {}  q_step = {}",
            fmt_thousands(q_ref),
            ordinary_post_plateau.q_step
        );
    }
    if let Some(q_ref) = ordinary_assembly.q_ref {
        println!(
            "    ordinary assembly q_ref = {}  q_step = {}",
            fmt_thousands(q_ref),
            ordinary_assembly.q_step
        );
    }
    if let Some(q_ref) = ordinary_quasi_literature.q_ref {
        println!(
            "    ordinary quasi-lit q_ref = {}  q_step = {}",
            fmt_thousands(q_ref),
            ordinary_quasi_literature.q_step
        );
    }
    println!(
        "    S2_trivial is zero: {}",
        if dr_analysis.s2_trivial_is_zero {
            "yes"
        } else {
            "no"
        }
    );
    if let Some(comparison) = comparison {
        println!("    comparison_vs_current:");
        println!(
            "      current rules: hard={}  easy={}",
            comparison.current.hard_rule_kind, comparison.current.easy_rule_kind
        );
        println!(
            "      delta lens: S1={}  S2_hard={}  S2_easy={}",
            fmt_delta_usize(comparison.current.s1_len, comparison.experimental.s1_len),
            fmt_delta_usize(
                comparison.current.s2_hard_len,
                comparison.experimental.s2_hard_len
            ),
            fmt_delta_usize(
                comparison.current.s2_easy_len,
                comparison.experimental.s2_easy_len
            )
        );
        println!(
            "      delta contrib: S1={}  S2_hard={}  S2_easy={}",
            fmt_delta_u128(
                comparison.current.contributions.ordinary,
                comparison.experimental.contributions.ordinary,
            ),
            fmt_delta_u128(
                comparison.current.contributions.hard,
                comparison.experimental.contributions.hard,
            ),
            fmt_delta_u128(
                comparison.current.contributions.easy,
                comparison.experimental.contributions.easy,
            )
        );
        println!(
            "      result delta = {}",
            fmt_delta_u128(comparison.current.result, comparison.experimental.result)
        );
    }
    println!();
}

#[allow(non_snake_case)]
fn run_dr_profile(x: u128) {
    println!(
        "── DR Profile  x = {} ─────────────────────────────────────────",
        fmt_thousands(x)
    );
    let runtime = dr::profile_prime_pi_dr(x);
    let (v2_result, v2_times) = dr::prime_pi_dr_v2_timed(x);
    let v2_time: std::time::Duration = v2_times.iter().sum();

    println!(
        "  φ(x,a) / Lucy          {:>10}  ({:.1}%)",
        fmt_elapsed(runtime.phi_time),
        pct(runtime.phi_time, runtime.total_time)
    );
    println!(
        "  seed primes ≤ ∛x       {:>10}  ({:.1}%)",
        fmt_elapsed(runtime.seed_primes_time),
        pct(runtime.seed_primes_time, runtime.total_time)
    );
    println!(
        "  primes up to √x        {:>10}  ({:.1}%)",
        fmt_elapsed(runtime.sqrt_primes_time),
        pct(runtime.sqrt_primes_time, runtime.total_time)
    );
    println!(
        "  S2 BIT sweep           {:>10}  ({:.1}%)",
        fmt_elapsed(runtime.s2_time),
        pct(runtime.s2_time, runtime.total_time)
    );
    println!("  ─────────────────────────────────────────");
    println!(
        "  DR v1 Total (Lucy+BIT)  {:>10}    π({}) = {}  {}",
        fmt_elapsed(runtime.total_time),
        fmt_thousands(x),
        fmt_thousands(runtime.result),
        if runtime.result == baseline::prime_pi(x) {
            "✓"
        } else {
            "✗ WRONG"
        }
    );
    println!(
        "  DR v2 Total (Lucy+pcnt) {:>10}    {}",
        fmt_elapsed(v2_time),
        if v2_result == runtime.result { "✓" } else { "✗ WRONG" }
    );
    let v2_labels = ["    Lucy early-stop", "    primes_up_to   ", "    s2_popcount    "];
    for (lbl, &t) in v2_labels.iter().zip(v2_times.iter()) {
        if t > std::time::Duration::ZERO {
            println!("  {} {:>10}", lbl, fmt_elapsed(t));
        }
    }
    println!(
        "  a = {}  phi(x,a) = {}",
        fmt_thousands(runtime.a as u128),
        fmt_thousands(runtime.phi_x_a)
    );
}

fn run_dr_meissel_profile(x: u128, threads: usize) {
    use std::time::Instant;
    println!(
        "── DR Meissel Profile  x = {} ─────────────────────────────────────────",
        fmt_thousands(x)
    );

    // Baseline for comparison
    let t = Instant::now();
    let baseline_result = baseline::prime_pi_with_threads(x, threads);
    let baseline_time = t.elapsed();

    // DR v1: Lucy + BIT S2
    let runtime = dr::profile_prime_pi_dr(x);

    // DR v2: Lucy + popcount S2
    let t = Instant::now();
    let v2_result = dr::prime_pi_dr_v2(x);
    let v2_time = t.elapsed();

    // Meissel DR with per-step timing
    let (meissel_result, step_times) = dr::prime_pi_dr_meissel_timed(x);
    let meissel_time: std::time::Duration = step_times.iter().sum();

    let ratio_vs_base = |t: std::time::Duration| t.as_secs_f64() / baseline_time.as_secs_f64();

    struct SummaryRow<'a> {
        label: &'a str,
        time: std::time::Duration,
        ratio: f64,
        ok: bool,
    }

    let mut rows = vec![
        SummaryRow {
            label: "Baseline  (Lucy+table)",
            time: baseline_time,
            ratio: 1.0,
            ok: true,
        },
        SummaryRow {
            label: "DR v1     (Lucy+BIT)",
            time: runtime.total_time,
            ratio: ratio_vs_base(runtime.total_time),
            ok: runtime.result == baseline_result,
        },
        SummaryRow {
            label: "DR v2     (Lucy+pcnt)",
            time: v2_time,
            ratio: ratio_vs_base(v2_time),
            ok: v2_result == baseline_result,
        },
        SummaryRow {
            label: "DR Meissel",
            time: meissel_time,
            ratio: ratio_vs_base(meissel_time),
            ok: meissel_result == baseline_result,
        },
    ];
    rows.sort_unstable_by_key(|row| row.time);

    println!("  Final summary (sorted by runtime):");
    for row in rows {
        println!(
            "  {:<24} {:>10}  ratio {:>.3}×  {}",
            row.label,
            fmt_elapsed(row.time),
            row.ratio,
            if row.ok { "✓" } else { "✗ WRONG" }
        );
    }
    println!("  Meissel step breakdown:");
    let labels = [
        "  sieve+primes",
        "  collect_phi ",
        "  BIT sweep   ",
        "  phi()       ",
    ];
    for (i, (&t, &lbl)) in step_times[..4].iter().zip(labels.iter()).enumerate() {
        println!("    step{} ({lbl})  {}", i + 1, fmt_elapsed(t));
    }
}

fn run_dr_meissel2_profile(x: u128, threads: usize) {
    use std::time::Instant;
    println!(
        "── DR Meissel v2 (PhiCache)  x = {} ───────────────────────────────────────",
        fmt_thousands(x)
    );

    // Baseline
    let t = Instant::now();
    let baseline_result = baseline::prime_pi_with_threads(x, threads);
    let baseline_time = t.elapsed();

    // DR v2 : référence rapide actuelle
    let t = Instant::now();
    let v2_result = dr::prime_pi_dr_v2(x);
    let v2_time = t.elapsed();

    // Meissel v1 : avec phi_table + memo HashMap (53M entrées)
    let (m1_result, m1_times) = dr::prime_pi_dr_meissel_timed(x);
    let m1_time: std::time::Duration = m1_times.iter().sum();

    // Meissel v2 : avec PhiCache + memo réduit
    let (m2_result, m2_times) = dr::prime_pi_dr_meissel_v2_timed(x);
    let m2_time: std::time::Duration = m2_times.iter().sum();

    let ratio = |t: std::time::Duration| t.as_secs_f64() / baseline_time.as_secs_f64();

    struct Row<'a> {
        label: &'a str,
        time: std::time::Duration,
        ratio: f64,
        ok: bool,
    }
    let rows = [
        Row { label: "Baseline  (Lucy+table)", time: baseline_time, ratio: 1.0, ok: true },
        Row { label: "DR v2     (Lucy+pcnt) ", time: v2_time,       ratio: ratio(v2_time), ok: v2_result == baseline_result },
        Row { label: "Meissel v1 (phi_table)", time: m1_time,       ratio: ratio(m1_time), ok: m1_result == baseline_result },
        Row { label: "Meissel v2 (PhiCache) ", time: m2_time,       ratio: ratio(m2_time), ok: m2_result == baseline_result },
    ];
    for row in &rows {
        println!(
            "  {:<26} {:>10}  {:>6.3}×  {}",
            row.label,
            fmt_elapsed(row.time),
            row.ratio,
            if row.ok { "✓" } else { "✗ WRONG" }
        );
    }
    println!("  Meissel v2 step breakdown:");
    let labels = ["sieve+cache   ", "collect_phi   ", "BIT sweep     ", "phi_cached()  "];
    for (i, (&t, &lbl)) in m2_times[..4].iter().zip(labels.iter()).enumerate() {
        println!("    step{} ({lbl})  {}", i + 1, fmt_elapsed(t));
    }
}

fn run_dr_meissel3_profile(x: u128, threads: usize) {
    use std::time::Instant;
    println!(
        "── DR Meissel v3 (PiTable + phi_loop)  x = {} ─────────────────────────────",
        fmt_thousands(x)
    );

    // Baseline
    let t = Instant::now();
    let baseline_result = baseline::prime_pi_with_threads(x, threads);
    let baseline_time = t.elapsed();

    // DR v2 : référence rapide actuelle
    let t = Instant::now();
    let v2_result = dr::prime_pi_dr_v2(x);
    let v2_time = t.elapsed();

    // Meissel v2 : PhiCache + mémo (référence) — HashMap mémo ~200 M entrées à x=1e14 → OOM.
    // On limite à x ≤ 1e13 pour éviter l'allocation de plusieurs Go.
    const MEISSEL_V2_LIMIT: u128 = 10_000_000_000_000; // 1e13
    let (m2_result, m2_time) = if x <= MEISSEL_V2_LIMIT {
        let t = Instant::now();
        let r = dr::prime_pi_dr_meissel_v2(x);
        (Some(r), t.elapsed())
    } else {
        (None, std::time::Duration::ZERO)
    };

    // Meissel v3 : PiTable + phi_loop sans mémo
    let (m3_result, m3_times) = dr::prime_pi_dr_meissel_v3_timed(x);
    let m3_time: std::time::Duration = m3_times[..4].iter().sum();

    let ratio = |t: std::time::Duration| t.as_secs_f64() / baseline_time.as_secs_f64();

    struct Row<'a> {
        label: &'a str,
        time: std::time::Duration,
        ratio: f64,
        ok: bool,
        skip: bool,
    }
    let rows = [
        Row { label: "Baseline  (Lucy+table)", time: baseline_time, ratio: 1.0,           ok: true,                              skip: false },
        Row { label: "DR v2     (Lucy+pcnt) ", time: v2_time,       ratio: ratio(v2_time), ok: v2_result == baseline_result,       skip: false },
        Row { label: "Meissel v2 (PhiCache) ", time: m2_time,       ratio: ratio(m2_time), ok: m2_result.map_or(true, |r| r == baseline_result), skip: m2_result.is_none() },
        Row { label: "Meissel v3 (PiTable)  ", time: m3_time,       ratio: ratio(m3_time), ok: m3_result == baseline_result,       skip: false },
    ];
    for row in &rows {
        if row.skip {
            println!("  {:<26} {:>10}  {:>6}   (skipped — x > 1e13)", row.label, "--", "--");
            continue;
        }
        println!(
            "  {:<26} {:>10}  {:>6.3}×  {}",
            row.label,
            fmt_elapsed(row.time),
            row.ratio,
            if row.ok { "✓" } else { "✗ WRONG" }
        );
    }
    println!("  Meissel v3 step breakdown:");
    let labels = ["sieve         ", "PiTable+Cache ", "BIT sweep S₂  ", "phi_loop()    "];
    for (i, (&t, &lbl)) in m3_times[..4].iter().zip(labels.iter()).enumerate() {
        println!("    step{} ({lbl})  {}", i + 1, fmt_elapsed(t));
    }
}

fn run_dr_meissel4_profile(x: u128, _threads: usize) {
    println!(
        "── DR Meissel v4  x = {} ────────────────────────────────────────────────────",
        fmt_thousands(x)
    );

    let (result, times) = dr::prime_pi_dr_meissel_v4_timed(x);
    let total: std::time::Duration = times[..4].iter().sum();

    let labels = ["sieve      ", "S1 DFS     ", "S2_hard    ", "P2         "];
    for (i, (&t, &lbl)) in times[..4].iter().zip(labels.iter()).enumerate() {
        println!("  step{} ({lbl})  {}", i + 1, fmt_elapsed(t));
    }
    println!("  total              {}", fmt_elapsed(total));
    println!("  π({}) = {}", fmt_thousands(x), fmt_thousands(result));
}

fn run_dr_v3_profile(x: u128) {
    use std::time::Instant;
    println!(
        "── DR v3 Profile  x = {} ──────────────────────────────────────────────────",
        fmt_thousands(x)
    );

    let t = Instant::now();
    let v2_result = dr::prime_pi_dr_v2(x);
    let v2_time = t.elapsed();

    let (v3_result, step_times) = dr::prime_pi_dr_v3_timed(x);
    let v3_time: std::time::Duration = step_times[..3].iter().sum();

    let ok_v2 = v2_result == v2_result; // always true — just for symmetry
    let ok_v3 = v3_result == v2_result;

    println!(
        "  v2 (Lucy+popcount)          {}  {}",
        fmt_elapsed(v2_time),
        if ok_v2 { "✓" } else { "✗" }
    );
    println!(
        "  v3 (dense medium_pi)        {}  {}  [{:.2}×]",
        fmt_elapsed(v3_time),
        if ok_v3 { "✓" } else { "✗" },
        v2_time.as_secs_f64() / v3_time.as_secs_f64().max(1e-9)
    );
    println!(
        "    step1 (sieve+phi_table)   {}",
        fmt_elapsed(step_times[0])
    );
    println!(
        "    step2 (s2+medium_pi sweep) {}",
        fmt_elapsed(step_times[1])
    );
    println!(
        "    step3 (phi recursion)      {}",
        fmt_elapsed(step_times[2])
    );
    println!("  π({}) = {}", fmt_thousands(x), fmt_thousands(v3_result));
}

fn run_dr_v4_profile(x: u128) {
    use std::time::Instant;
    println!(
        "── DR v4 Profile  x = {} ──────────────────────────────────────────────────",
        fmt_thousands(x)
    );

    let t = Instant::now();
    let v2_result = dr::prime_pi_dr_v2(x);
    let v2_time = t.elapsed();

    let (v4_result, step_times) = dr::prime_pi_dr_v4_timed(x);
    let v4_time: std::time::Duration = step_times[..4].iter().sum();

    let ok_v4 = v4_result == v2_result;

    println!(
        "  v2 (Lucy+popcount)          {}  ✓",
        fmt_elapsed(v2_time),
    );
    println!(
        "  v4 (flat memo+popcount)     {}  {}  [{:.2}×]",
        fmt_elapsed(v4_time),
        if ok_v4 { "✓" } else { "✗" },
        v2_time.as_secs_f64() / v4_time.as_secs_f64().max(1e-9)
    );
    println!(
        "    step1 (sieve+phi_table)   {}",
        fmt_elapsed(step_times[0])
    );
    println!(
        "    step2 (collect queries)   {}",
        fmt_elapsed(step_times[1])
    );
    println!(
        "    step3 (S2+large_pi par)   {}",
        fmt_elapsed(step_times[2])
    );
    println!(
        "    step4 (phi flat memo)     {}",
        fmt_elapsed(step_times[3])
    );
    println!("  π({}) = {}", fmt_thousands(x), fmt_thousands(v4_result));
}

fn run_dr_vs_baseline_grid(threads: usize) {
    let xs = [
        1_000_000_000u128,
        10_000_000_000,
        100_000_000_000,
        1_000_000_000_000,
    ];

    println!(
        "── DR vs baseline grid  |  baseline S2 threads = {} ─────────────────────",
        threads
    );
    println!(
        "{:>14}  {:>10}  {:>10}  {:>8}  {:>8}",
        "x", "baseline", "DR", "DR/base", "result"
    );
    println!("{}", "-".repeat(62));

    for x in xs {
        let t = Instant::now();
        let baseline_result = baseline::prime_pi_with_threads(x, threads);
        let baseline_time = t.elapsed();

        let runtime = dr::profile_prime_pi_dr(x);
        let ratio = if baseline_time.is_zero() {
            0.0
        } else {
            runtime.total_time.as_secs_f64() / baseline_time.as_secs_f64()
        };
        let result_ok = if baseline_result == runtime.result {
            "ok"
        } else {
            "diff"
        };

        println!(
            "{:>14}  {:>10}  {:>10}  {:>8.2}  {:>8}",
            fmt_thousands(x),
            fmt_elapsed(baseline_time),
            fmt_elapsed(runtime.total_time),
            ratio,
            result_ok
        );
    }
}

fn run_nt_batch(jobs: &[NtBatchJob]) {
    #[derive(Clone)]
    struct NtBatchResult {
        label: String,
        x: u128,
        result: u128,
        elapsed: std::time::Duration,
    }

    let mut results = Vec::with_capacity(jobs.len());

    for job in jobs {
        let mem_mib = estimate_baseline_memory_mib(job.x);
        println!(
            "n = {}  |  Calcul en cours...  [meissel+lucy+s2, {} {}, {} MiB]",
            job.label,
            job.threads,
            if job.threads > 1 { "threads" } else { "thread" },
            mem_mib
        );
        let _ = std::io::stdout().flush();

        let t0 = Instant::now();
        let result = baseline::prime_pi_with_threads(job.x, job.threads);
        let elapsed = t0.elapsed();

        println!(
            "n = {}  |  π(n) = {}  |  {}",
            job.label,
            fmt_thousands(result),
            fmt_elapsed(elapsed)
        );
        println!();

        results.push(NtBatchResult {
            label: job.label.clone(),
            x: job.x,
            result,
            elapsed,
        });
    }

    if results.len() <= 1 {
        return;
    }

    results.sort_unstable_by_key(|row| row.x);

    println!("--------------------- Résumé ---------------------");
    println!(
        "{:<7}| {:<19}| {:>9} | {:>8}",
        "Limite", "π(n)", "Temps", "×Temps"
    );
    println!("-------+--------------------+-----------+---------");

    let mut total = std::time::Duration::ZERO;
    let mut previous: Option<std::time::Duration> = None;
    for row in &results {
        total += row.elapsed;
        let ratio = previous.map(|prev| row.elapsed.as_secs_f64() / prev.as_secs_f64());
        println!(
            "{:<7}| {:<19}| {:>9} | {:>8}",
            row.label,
            fmt_thousands(row.result),
            fmt_elapsed_seconds(row.elapsed),
            ratio
                .map(|v| format!("{v:.4}×"))
                .unwrap_or_else(|| "-".to_string())
        );
        previous = Some(row.elapsed);
    }

    println!("-------+--------------------+-----------+---------");
    println!(
        "{:<7}| {:<19}| {:>9} |",
        "Σ",
        "",
        fmt_elapsed_seconds(total)
    );
}


#[allow(dead_code)]
fn run_normal_with_progress_display(x: u128, threads: usize) {
    let mem_mib = estimate_baseline_memory_mib(x);
    println!(
        "n = {}  |  Calcul en cours...  [meissel+lucy+s2, {} {}, {} MiB]",
        fmt_thousands(x),
        threads,
        if threads > 1 { "threads" } else { "thread" },
        mem_mib
    );
    let _ = std::io::stdout().flush();

    let t0 = Instant::now();
    let result = baseline::prime_pi_with_threads(x, threads);
    let elapsed = t0.elapsed();

    println!(
        "n = {}  |  {}",
        fmt_thousands(x),
        fmt_elapsed_seconds(elapsed)
    );
    println!("π(n) = {}", fmt_thousands(result));
}

fn run_phi_backend_grid() {
    let xs = [
        1_000u128,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
    ];

    println!("── φ backend grid ─────────────────────────────────────────────────");
    println!(
        "{:>12}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}",
        "x", "Lucy", "Ref", "QRef", "a", "phi"
    );
    println!("{}", "-".repeat(82));

    for x in xs {
        let t = Instant::now();
        let lucy = rivat3::phi::phi_computation_with_backend(x, rivat3::phi::PhiBackend::Lucy);
        let lucy_time = t.elapsed();

        let t = Instant::now();
        let reference =
            rivat3::phi::phi_computation_with_backend(x, rivat3::phi::PhiBackend::Reference);
        let reference_time = t.elapsed();

        let t = Instant::now();
        let quotient = rivat3::phi::phi_computation_with_backend(
            x,
            rivat3::phi::PhiBackend::ReferenceQuotient,
        );
        let quotient_time = t.elapsed();

        let a_ok = if lucy.a == reference.a && lucy.a == quotient.a {
            "ok"
        } else {
            "diff"
        };
        let phi_ok = if lucy.phi_x_a == reference.phi_x_a && lucy.phi_x_a == quotient.phi_x_a {
            "ok"
        } else {
            "diff"
        };

        println!(
            "{:>12}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}",
            fmt_thousands(x),
            fmt_elapsed(lucy_time),
            fmt_elapsed(reference_time),
            fmt_elapsed(quotient_time),
            a_ok,
            phi_ok
        );
    }
}

fn run_phi_backend_profile(x: u128) {
    println!(
        "── φ backend profile  x = {} ─────────────────────────────────────",
        fmt_thousands(x)
    );

    let t = Instant::now();
    let lucy = rivat3::phi::phi_computation_with_backend(x, rivat3::phi::PhiBackend::Lucy);
    let lucy_time = t.elapsed();

    let t = Instant::now();
    let reference =
        rivat3::phi::phi_computation_with_backend(x, rivat3::phi::PhiBackend::Reference);
    let reference_time = t.elapsed();

    let t = Instant::now();
    let quotient =
        rivat3::phi::phi_computation_with_backend(x, rivat3::phi::PhiBackend::ReferenceQuotient);
    let quotient_time = t.elapsed();

    let ratio = if lucy_time.is_zero() {
        0.0
    } else {
        reference_time.as_secs_f64() / lucy_time.as_secs_f64()
    };
    let q_ratio = if lucy_time.is_zero() {
        0.0
    } else {
        quotient_time.as_secs_f64() / lucy_time.as_secs_f64()
    };

    println!("  Lucy       {:>10}", fmt_elapsed(lucy_time));
    println!("  Reference  {:>10}", fmt_elapsed(reference_time));
    println!("  Ref/Lucy   {:>10.2}", ratio);
    println!("  QRef       {:>10}", fmt_elapsed(quotient_time));
    println!("  QRef/Lucy  {:>10.2}", q_ratio);
    println!(
        "  a match    {:>10}    φ match {:>10}",
        if lucy.a == reference.a && lucy.a == quotient.a {
            "ok"
        } else {
            "diff"
        },
        if lucy.phi_x_a == reference.phi_x_a && lucy.phi_x_a == quotient.phi_x_a {
            "ok"
        } else {
            "diff"
        }
    );
    println!(
        "  a = {}    φ(x,a) = {}",
        fmt_thousands(lucy.a as u128),
        fmt_thousands(lucy.phi_x_a)
    );
}

fn run_dr_phi_backend_profile(x: u128) {
    println!(
        "── DR φ backend profile  x = {} ─────────────────────────────────",
        fmt_thousands(x)
    );

    let lucy = dr::profile_prime_pi_dr_with_backend(x, rivat3::phi::PhiBackend::Lucy);
    let reference = dr::profile_prime_pi_dr_with_backend(x, rivat3::phi::PhiBackend::Reference);
    let quotient =
        dr::profile_prime_pi_dr_with_backend(x, rivat3::phi::PhiBackend::ReferenceQuotient);

    let ratio = if lucy.total_time.is_zero() {
        0.0
    } else {
        reference.total_time.as_secs_f64() / lucy.total_time.as_secs_f64()
    };
    let q_ratio = if lucy.total_time.is_zero() {
        0.0
    } else {
        quotient.total_time.as_secs_f64() / lucy.total_time.as_secs_f64()
    };

    println!("  Lucy total      {:>10}", fmt_elapsed(lucy.total_time));
    println!(
        "  Reference total {:>10}",
        fmt_elapsed(reference.total_time)
    );
    println!("  Ref/Lucy total  {:>10.2}", ratio);
    println!("  QRef total      {:>10}", fmt_elapsed(quotient.total_time));
    println!("  QRef/Lucy total {:>10.2}", q_ratio);
    println!(
        "  result match    {:>10}    a match {:>10}",
        if lucy.result == reference.result && lucy.result == quotient.result {
            "ok"
        } else {
            "diff"
        },
        if lucy.a == reference.a && lucy.a == quotient.a {
            "ok"
        } else {
            "diff"
        }
    );
    println!(
        "  φ time Lucy     {:>10}    φ time Ref {:>10}",
        fmt_elapsed(lucy.phi_time),
        fmt_elapsed(reference.phi_time)
    );
    println!("  φ time QRef     {:>10}", fmt_elapsed(quotient.phi_time));
    println!(
        "  S2 time Lucy    {:>10}    S2 time Ref {:>10}",
        fmt_elapsed(lucy.s2_time),
        fmt_elapsed(reference.s2_time)
    );
    println!("  S2 time QRef    {:>10}", fmt_elapsed(quotient.s2_time));
}

/// Runs a benchmark sweep over powers of 10 up to x_max.
fn run_sweep(x_max: u128, threads: usize) {
    println!(
        "{:>22}  {:>12}  {:>10}  {:>10}  result",
        "x", "total", "lucy", "S2"
    );
    println!("{}", "-".repeat(75));

    let mut x = 10u128;
    while x <= x_max {
        let z_u128 = isqrt(x);
        let y = icbrt(x);
        let z = z_u128 as usize;

        let t0 = Instant::now();
        let (small, large, phi_x_a) = lucy_hedgehog_with_phi(x, y);
        let t_lucy = t0.elapsed();

        let a = pi_at(y, x, z, &small, &large) as usize;
        let primes = extract_primes(&small, z);

        let t1 = Instant::now();
        let s2_val = s2(x, a, z, &primes, &small, &large, threads);
        let t_s2 = t1.elapsed();

        let s3_val = s3(x, a, &primes);

        let result = phi_x_a + a as u128 - 1 - s2_val - s3_val;
        let total = t0.elapsed();

        println!(
            "{:>22}  {:>12}  {:>10}  {:>10}  {}",
            fmt_thousands(x),
            fmt_elapsed(total),
            fmt_elapsed(t_lucy),
            fmt_elapsed(t_s2),
            fmt_thousands(result)
        );

        x *= 10;
    }
}

fn run_experimental_sweep(
    x_max: u128,
    hard_leaf_term_max: u128,
    easy_leaf_term_max: u128,
    experimental_mode: ExperimentalMode,
) {
    println!(
        "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "x", "delta S1", "delta S2_hard", "delta S2_easy", "delta easy", "result Δ"
    );
    println!("{}", "-".repeat(95));

    let mut x = 10u128;
    while x <= x_max {
        let comparison = match experimental_mode {
            ExperimentalMode::CandidateEasyRelativeToHard => {
                dr::compare_current_vs_candidate_easy_relative_to_hard(
                    x,
                    hard_leaf_term_max,
                    easy_leaf_term_max,
                )
            }
            ExperimentalMode::CandidateEasyTermBand => {
                dr::compare_current_vs_candidate_easy_term_band(
                    x,
                    hard_leaf_term_max,
                    easy_leaf_term_max,
                )
            }
            ExperimentalMode::PhaseCEasyTermBand => dr::compare_current_vs_phase_c_easy_term_band(
                x,
                rivat3::parameters::DrTuning::PHASE_C_EASY_TERM_BAND_MAX + 1,
                easy_leaf_term_max,
            ),
            ExperimentalMode::PhaseCPackage => dr::compare_current_vs_phase_c_package(x),
            ExperimentalMode::PhaseCLinkedPackage => {
                dr::compare_phase_c_package_vs_linked_package(x)
            }
            ExperimentalMode::EasyRelativeToHard { width } => {
                dr::compare_current_vs_experimental_easy_relative_to_hard(
                    x,
                    hard_leaf_term_max,
                    easy_leaf_term_max,
                    width,
                )
            }
            ExperimentalMode::HardRelativeToEasy { width } => {
                dr::compare_current_vs_experimental_hard_relative_to_easy(
                    x,
                    hard_leaf_term_max,
                    easy_leaf_term_max,
                    width,
                )
            }
            ExperimentalMode::None => unreachable!("experimental sweep requires an active mode"),
        };

        println!(
            "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
            fmt_thousands(x),
            fmt_delta_usize(comparison.current.s1_len, comparison.experimental.s1_len),
            fmt_delta_usize(
                comparison.current.s2_hard_len,
                comparison.experimental.s2_hard_len,
            ),
            fmt_delta_usize(
                comparison.current.s2_easy_len,
                comparison.experimental.s2_easy_len,
            ),
            fmt_delta_u128(
                comparison.current.contributions.easy,
                comparison.experimental.contributions.easy,
            ),
            fmt_delta_u128(comparison.current.result, comparison.experimental.result),
        );

        x *= 10;
    }
}

fn run_candidate_family_compare() {
    let xs = candidate_grid_points();

    println!(
        "{:>22}  {:>18}  {:>18}  {:>12}",
        "x", "rel_to_hard Îeasy", "term_band Îeasy", "result Î”"
    );
    println!("{}", "-".repeat(82));

    for &x in xs {
        let rel = dr::compare_current_vs_candidate_easy_relative_to_hard(x, 3, 1);
        let band = dr::compare_current_vs_candidate_easy_term_band(x, 3, 1);
        println!(
            "{:>22}  {:>18}  {:>18}  {:>12}",
            fmt_thousands(x),
            fmt_delta_u128(
                rel.current.contributions.easy,
                rel.experimental.contributions.easy
            ),
            fmt_delta_u128(
                band.current.contributions.easy,
                band.experimental.contributions.easy
            ),
            if rel.current.result == rel.experimental.result
                && band.current.result == band.experimental.result
            {
                "0".to_string()
            } else {
                "non-zero".to_string()
            }
        );
    }
}

fn run_candidate_band_search() {
    let candidates = [(1_u128, 2_u128), (2, 2), (1, 3), (2, 3)];

    println!(
        "{:>14}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "band", "hits", "easy Δ", "hard Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(84));

    for &(min_term, max_term) in &candidates {
        let mut hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut hard_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison =
                dr::compare_current_vs_experimental_easy_term_band(x, 3, 1, min_term, max_term);
            let easy_delta = comparison
                .experimental
                .s2_easy_len
                .saturating_sub(comparison.current.s2_easy_len);
            let hard_delta = comparison
                .current
                .s2_hard_len
                .saturating_sub(comparison.experimental.s2_hard_len);

            if easy_delta > 0 || hard_delta > 0 {
                hits += 1;
                easy_delta_total += easy_delta;
                hard_delta_total += hard_delta;
                if first_hit.is_none() {
                    first_hit = Some(x);
                }
            }

            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>14}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            format!("[{}..={}]", min_term, max_term),
            hits,
            easy_delta_total,
            hard_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_phase_c_easy_band_grid() {
    println!(
        "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}",
        "x", "delta S1", "delta S2_hard", "delta S2_easy", "result Δ"
    );
    println!("{}", "-".repeat(82));

    for &x in candidate_grid_points() {
        let comparison = dr::compare_current_vs_phase_c_easy_term_band(
            x,
            rivat3::parameters::DrTuning::PHASE_C_EASY_TERM_BAND_MAX + 1,
            1,
        );
        println!(
            "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}",
            fmt_thousands(x),
            fmt_delta_usize(comparison.current.s1_len, comparison.experimental.s1_len),
            fmt_delta_usize(
                comparison.current.s2_hard_len,
                comparison.experimental.s2_hard_len,
            ),
            fmt_delta_usize(
                comparison.current.s2_easy_len,
                comparison.experimental.s2_easy_len,
            ),
            fmt_delta_u128(comparison.current.result, comparison.experimental.result),
        );
    }
}

fn run_phase_c_easy_compare() {
    println!(
        "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "x", "delta S1", "delta S2_hard", "delta S2_easy", "delta easy", "result Δ"
    );
    println!("{}", "-".repeat(95));

    for &x in candidate_grid_points() {
        let comparison = dr::compare_candidate_easy_reference_vs_phase_c_term_band(x, 3);
        println!(
            "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
            fmt_thousands(x),
            fmt_delta_usize(comparison.current.s1_len, comparison.experimental.s1_len),
            fmt_delta_usize(
                comparison.current.s2_hard_len,
                comparison.experimental.s2_hard_len,
            ),
            fmt_delta_usize(
                comparison.current.s2_easy_len,
                comparison.experimental.s2_easy_len,
            ),
            fmt_delta_u128(
                comparison.current.contributions.easy,
                comparison.experimental.contributions.easy,
            ),
            fmt_delta_u128(comparison.current.result, comparison.experimental.result),
        );
    }
}

fn run_phase_c_easy_search() {
    let candidates = [(1_u128, 3_u128), (2, 3), (1, 4), (2, 4)];

    println!(
        "{:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "band", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(98));

    for &(min_term, max_term) in &candidates {
        let mut s1_hits = 0usize;
        let mut hard_hits = 0usize;
        let mut easy_hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison = dr::compare_current_vs_experimental_easy_term_band(
                x,
                max_term + 1,
                1,
                min_term,
                max_term,
            );

            if comparison.current.s1_len != comparison.experimental.s1_len {
                s1_hits += 1;
            }
            if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
                hard_hits += 1;
            }
            if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
                easy_hits += 1;
                easy_delta_total += comparison
                    .experimental
                    .s2_easy_len
                    .saturating_sub(comparison.current.s2_easy_len);
            }

            if first_hit.is_none()
                && (comparison.current.s1_len != comparison.experimental.s1_len
                    || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                    || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
            {
                first_hit = Some(x);
            }

            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            format!("[{}..={}]", min_term, max_term),
            s1_hits,
            hard_hits,
            easy_hits,
            easy_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_phase_c_easy_compare_bands() {
    println!(
        "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "x", "delta S1", "delta S2_hard", "delta S2_easy", "delta easy", "result Δ"
    );
    println!("{}", "-".repeat(95));

    for &x in candidate_grid_points() {
        let comparison = dr::compare_phase_c_easy_term_bands(x, 1, 4, 1, 3);
        println!(
            "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
            fmt_thousands(x),
            fmt_delta_usize(comparison.current.s1_len, comparison.experimental.s1_len),
            fmt_delta_usize(
                comparison.current.s2_hard_len,
                comparison.experimental.s2_hard_len,
            ),
            fmt_delta_usize(
                comparison.current.s2_easy_len,
                comparison.experimental.s2_easy_len,
            ),
            fmt_delta_u128(
                comparison.current.contributions.easy,
                comparison.experimental.contributions.easy,
            ),
            fmt_delta_u128(comparison.current.result, comparison.experimental.result),
        );
    }
}

fn run_phase_c_hard_grid() {
    println!(
        "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}",
        "x", "delta S1", "delta S2_hard", "delta S2_easy", "result Δ"
    );
    println!("{}", "-".repeat(82));

    for &x in candidate_grid_points() {
        let comparison = dr::compare_phase_c_hard_term_band_with_current(
            x,
            rivat3::parameters::DrTuning::PHASE_C_HARD_TERM_BAND_MAX,
            rivat3::parameters::DrTuning::PHASE_C_EASY_TERM_BAND_MAX,
        );
        println!(
            "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}",
            fmt_thousands(x),
            fmt_delta_usize(comparison.current.s1_len, comparison.experimental.s1_len),
            fmt_delta_usize(
                comparison.current.s2_hard_len,
                comparison.experimental.s2_hard_len,
            ),
            fmt_delta_usize(
                comparison.current.s2_easy_len,
                comparison.experimental.s2_easy_len,
            ),
            fmt_delta_u128(comparison.current.result, comparison.experimental.result),
        );
    }
}

fn run_phase_c_hard_search() {
    let candidates = [(5_u128, 6_u128), (5, 7), (5, 8), (6, 6)];

    println!(
        "{:>14}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "band", "S1 hits", "hard hits", "easy hits", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(88));

    for &(min_term, max_term) in &candidates {
        let mut s1_hits = 0usize;
        let mut hard_hits = 0usize;
        let mut easy_hits = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison = dr::compare_phase_c_hard_term_bands(
                x,
                rivat3::parameters::DrTuning::PHASE_C_HARD_TERM_BAND_MIN,
                rivat3::parameters::DrTuning::PHASE_C_HARD_TERM_BAND_MAX,
                min_term,
                max_term,
            );

            if comparison.current.s1_len != comparison.experimental.s1_len {
                s1_hits += 1;
            }
            if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
                hard_hits += 1;
            }
            if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
                easy_hits += 1;
            }

            if first_hit.is_none()
                && (comparison.current.s1_len != comparison.experimental.s1_len
                    || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                    || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
            {
                first_hit = Some(x);
            }

            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>14}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            format!("[{}..={}]", min_term, max_term),
            s1_hits,
            hard_hits,
            easy_hits,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_phase_c_hard_compare_bands() {
    println!(
        "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}",
        "x", "delta S1", "delta S2_hard", "delta S2_easy", "result Δ"
    );
    println!("{}", "-".repeat(82));

    for &x in candidate_grid_points() {
        let comparison = dr::compare_phase_c_hard_term_bands(
            x,
            rivat3::parameters::DrTuning::PHASE_C_HARD_TERM_BAND_MIN,
            rivat3::parameters::DrTuning::PHASE_C_HARD_TERM_BAND_MAX,
            5,
            7,
        );
        println!(
            "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}",
            fmt_thousands(x),
            fmt_delta_usize(comparison.current.s1_len, comparison.experimental.s1_len),
            fmt_delta_usize(
                comparison.current.s2_hard_len,
                comparison.experimental.s2_hard_len,
            ),
            fmt_delta_usize(
                comparison.current.s2_easy_len,
                comparison.experimental.s2_easy_len,
            ),
            fmt_delta_u128(comparison.current.result, comparison.experimental.result),
        );
    }
}

fn run_phase_c_package_grid() {
    println!(
        "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "x", "delta S1", "delta S2_hard", "delta S2_easy", "delta easy", "result Δ"
    );
    println!("{}", "-".repeat(95));

    for &x in candidate_grid_points() {
        let comparison = dr::compare_current_vs_phase_c_package(x);
        println!(
            "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
            fmt_thousands(x),
            fmt_delta_usize(comparison.current.s1_len, comparison.experimental.s1_len),
            fmt_delta_usize(
                comparison.current.s2_hard_len,
                comparison.experimental.s2_hard_len,
            ),
            fmt_delta_usize(
                comparison.current.s2_easy_len,
                comparison.experimental.s2_easy_len,
            ),
            fmt_delta_u128(
                comparison.current.contributions.easy,
                comparison.experimental.contributions.easy,
            ),
            fmt_delta_u128(comparison.current.result, comparison.experimental.result),
        );
    }
}

fn run_phase_c_package_search() {
    let candidates = [(1_u128, 4_u128, 5_u128, 6_u128), (1, 4, 5, 7), (1, 4, 5, 8)];

    println!(
        "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "package", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(108));

    for &(easy_min, easy_max, hard_min, hard_max) in &candidates {
        let mut s1_hits = 0usize;
        let mut hard_hits = 0usize;
        let mut easy_hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison = dr::compare_phase_c_packages(
                x,
                rivat3::parameters::DrTuning::PHASE_C_EASY_TERM_BAND_MIN,
                rivat3::parameters::DrTuning::PHASE_C_EASY_TERM_BAND_MAX,
                rivat3::parameters::DrTuning::PHASE_C_HARD_TERM_BAND_MIN,
                rivat3::parameters::DrTuning::PHASE_C_HARD_TERM_BAND_MAX,
                easy_min,
                easy_max,
                hard_min,
                hard_max,
            );

            if comparison.current.s1_len != comparison.experimental.s1_len {
                s1_hits += 1;
            }
            if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
                hard_hits += 1;
            }
            if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
                easy_hits += 1;
                easy_delta_total += comparison
                    .experimental
                    .s2_easy_len
                    .saturating_sub(comparison.current.s2_easy_len);
            }

            if first_hit.is_none()
                && (comparison.current.s1_len != comparison.experimental.s1_len
                    || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                    || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
            {
                first_hit = Some(x);
            }

            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            format!(
                "[{}..={}] / [{}..={}]",
                easy_min, easy_max, hard_min, hard_max
            ),
            s1_hits,
            hard_hits,
            easy_hits,
            easy_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_phase_c_linked_package_compare() {
    println!(
        "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "x", "delta S1", "delta S2_hard", "delta S2_easy", "delta easy", "result Δ"
    );
    println!("{}", "-".repeat(95));

    for &x in candidate_grid_points() {
        let comparison = dr::compare_phase_c_package_vs_linked_package(x);
        println!(
            "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
            fmt_thousands(x),
            fmt_delta_usize(comparison.current.s1_len, comparison.experimental.s1_len),
            fmt_delta_usize(
                comparison.current.s2_hard_len,
                comparison.experimental.s2_hard_len,
            ),
            fmt_delta_usize(
                comparison.current.s2_easy_len,
                comparison.experimental.s2_easy_len,
            ),
            fmt_delta_u128(
                comparison.current.contributions.easy,
                comparison.experimental.contributions.easy,
            ),
            fmt_delta_u128(comparison.current.result, comparison.experimental.result),
        );
    }
}

fn run_phase_c_linked_grid() {
    println!(
        "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "x", "delta S1", "delta S2_hard", "delta S2_easy", "delta easy", "result Δ"
    );
    println!("{}", "-".repeat(95));

    let current = rivat3::parameters::DrTuning::phase_c_linked_candidate();
    let experimental = rivat3::parameters::PhaseCLinkedCandidate {
        easy_width: 5,
        easy_min_term_floor: 1,
        hard_width: 2,
    };

    for &x in candidate_grid_points() {
        let comparison = dr::compare_phase_c_linked_variants(x, current, experimental);
        println!(
            "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
            fmt_thousands(x),
            fmt_delta_usize(comparison.current.s1_len, comparison.experimental.s1_len),
            fmt_delta_usize(
                comparison.current.s2_hard_len,
                comparison.experimental.s2_hard_len,
            ),
            fmt_delta_usize(
                comparison.current.s2_easy_len,
                comparison.experimental.s2_easy_len,
            ),
            fmt_delta_u128(
                comparison.current.contributions.easy,
                comparison.experimental.contributions.easy,
            ),
            fmt_delta_u128(comparison.current.result, comparison.experimental.result),
        );
    }
}

fn run_phase_c_linked_search() {
    let current = rivat3::parameters::DrTuning::phase_c_linked_candidate();
    let candidates = [
        rivat3::parameters::PhaseCLinkedCandidate {
            easy_width: 5,
            easy_min_term_floor: 1,
            hard_width: 1,
        },
        rivat3::parameters::PhaseCLinkedCandidate {
            easy_width: 5,
            easy_min_term_floor: 1,
            hard_width: 2,
        },
        rivat3::parameters::PhaseCLinkedCandidate {
            easy_width: 5,
            easy_min_term_floor: 1,
            hard_width: 3,
        },
        rivat3::parameters::PhaseCLinkedCandidate {
            easy_width: 5,
            easy_min_term_floor: 2,
            hard_width: 2,
        },
        rivat3::parameters::PhaseCLinkedCandidate {
            easy_width: 6,
            easy_min_term_floor: 1,
            hard_width: 2,
        },
        rivat3::parameters::PhaseCLinkedCandidate {
            easy_width: 4,
            easy_min_term_floor: 2,
            hard_width: 2,
        },
    ];

    println!(
        "{:>24}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "linked profile", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(110));

    for candidate in candidates {
        let mut s1_hits = 0usize;
        let mut hard_hits = 0usize;
        let mut easy_hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison = dr::compare_phase_c_linked_variants(x, current, candidate);

            if comparison.current.s1_len != comparison.experimental.s1_len {
                s1_hits += 1;
            }
            if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
                hard_hits += 1;
            }
            if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
                easy_hits += 1;
                easy_delta_total += comparison
                    .experimental
                    .s2_easy_len
                    .saturating_sub(comparison.current.s2_easy_len);
            }
            if first_hit.is_none()
                && (comparison.current.s1_len != comparison.experimental.s1_len
                    || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                    || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
            {
                first_hit = Some(x);
            }
            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>24}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            format!(
                "(w={}, f={}, hw={})",
                candidate.easy_width, candidate.easy_min_term_floor, candidate.hard_width
            ),
            s1_hits,
            hard_hits,
            easy_hits,
            easy_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_phase_c_linked_candidate_compare() {
    println!(
        "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "x", "delta S1", "delta S2_hard", "delta S2_easy", "delta easy", "result Δ"
    );
    println!("{}", "-".repeat(95));

    for &x in candidate_grid_points() {
        let comparison = dr::compare_phase_c_package_vs_linked_candidate(x);
        println!(
            "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
            fmt_thousands(x),
            fmt_delta_usize(comparison.current.s1_len, comparison.experimental.s1_len),
            fmt_delta_usize(
                comparison.current.s2_hard_len,
                comparison.experimental.s2_hard_len,
            ),
            fmt_delta_usize(
                comparison.current.s2_easy_len,
                comparison.experimental.s2_easy_len,
            ),
            fmt_delta_u128(
                comparison.current.contributions.easy,
                comparison.experimental.contributions.easy,
            ),
            fmt_delta_u128(comparison.current.result, comparison.experimental.result),
        );
    }
}

fn phase_c_dense_points() -> &'static [u128] {
    &[
        1_000_u128,
        1_500,
        2_000,
        3_000,
        5_000,
        7_500,
        10_000,
        15_000,
        20_000,
        30_000,
        50_000,
        75_000,
        100_000,
        150_000,
        200_000,
        300_000,
        500_000,
        750_000,
        1_000_000,
        1_500_000,
        2_000_000,
        3_000_000,
        5_000_000,
        7_500_000,
        10_000_000,
        15_000_000,
        20_000_000,
        30_000_000,
        50_000_000,
        75_000_000,
        100_000_000,
    ]
}

fn run_phase_c_reference_compare_dense() {
    let mut s1_hits = 0usize;
    let mut hard_hits = 0usize;
    let mut easy_hits = 0usize;
    let mut easy_delta_total = 0usize;
    let mut first_hit: Option<u128> = None;
    let mut result_delta_nonzero = false;

    for &x in phase_c_dense_points() {
        let comparison = dr::compare_phase_c_package_vs_linked_candidate(x);

        if comparison.current.s1_len != comparison.experimental.s1_len {
            s1_hits += 1;
        }
        if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
            hard_hits += 1;
        }
        if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
            easy_hits += 1;
            easy_delta_total += comparison
                .experimental
                .s2_easy_len
                .saturating_sub(comparison.current.s2_easy_len);
        }
        if first_hit.is_none()
            && (comparison.current.s1_len != comparison.experimental.s1_len
                || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
        {
            first_hit = Some(x);
        }
        if comparison.current.result != comparison.experimental.result {
            result_delta_nonzero = true;
        }
    }

    println!(
        "{:>18}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "reference pair", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(100));
    println!(
        "{:>18}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "band vs linked",
        s1_hits,
        hard_hits,
        easy_hits,
        easy_delta_total,
        first_hit
            .map(fmt_thousands)
            .unwrap_or_else(|| "-".to_string()),
        if result_delta_nonzero {
            "non-zero"
        } else {
            "0"
        }
    );
}

fn run_phase_c_boundary_package_compare() {
    println!(
        "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "x", "delta S1", "delta S2_hard", "delta S2_easy", "delta easy", "result Δ"
    );
    println!("{}", "-".repeat(95));

    for &x in candidate_grid_points() {
        let comparison = dr::compare_phase_c_package_vs_boundary_package(x);
        println!(
            "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
            fmt_thousands(x),
            fmt_delta_usize(comparison.current.s1_len, comparison.experimental.s1_len),
            fmt_delta_usize(
                comparison.current.s2_hard_len,
                comparison.experimental.s2_hard_len,
            ),
            fmt_delta_usize(
                comparison.current.s2_easy_len,
                comparison.experimental.s2_easy_len,
            ),
            fmt_delta_u128(
                comparison.current.contributions.easy,
                comparison.experimental.contributions.easy,
            ),
            fmt_delta_u128(comparison.current.result, comparison.experimental.result),
        );
    }
}

fn run_phase_c_boundary_search() {
    let current = rivat3::parameters::DrTuning::phase_c_boundary_candidate();
    let candidates = [
        rivat3::parameters::PhaseCBoundaryCandidate {
            boundary_term: 4,
            easy_width: 4,
            hard_width: 2,
        },
        rivat3::parameters::PhaseCBoundaryCandidate {
            boundary_term: 4,
            easy_width: 4,
            hard_width: 3,
        },
        rivat3::parameters::PhaseCBoundaryCandidate {
            boundary_term: 5,
            easy_width: 4,
            hard_width: 2,
        },
        rivat3::parameters::PhaseCBoundaryCandidate {
            boundary_term: 5,
            easy_width: 5,
            hard_width: 2,
        },
    ];

    println!(
        "{:>9}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "candidate", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(95));

    for candidate in candidates {
        let mut s1_hits = 0usize;
        let mut hard_hits = 0usize;
        let mut easy_hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison = dr::compare_phase_c_boundary_variants(x, current, candidate);
            let delta_s1 = comparison.current.s1_len != comparison.experimental.s1_len;
            let delta_hard = comparison.current.s2_hard_len != comparison.experimental.s2_hard_len;
            let delta_easy = comparison.current.s2_easy_len != comparison.experimental.s2_easy_len;

            if delta_s1 || delta_hard || delta_easy {
                if first_hit.is_none() {
                    first_hit = Some(x);
                }
                if delta_s1 {
                    s1_hits += 1;
                }
                if delta_hard {
                    hard_hits += 1;
                }
                if delta_easy {
                    easy_hits += 1;
                }
                easy_delta_total += comparison
                    .experimental
                    .contributions
                    .easy
                    .abs_diff(comparison.current.contributions.easy)
                    as usize;
            }

            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>9}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            format!(
                "{}:{}:{}",
                candidate.boundary_term, candidate.easy_width, candidate.hard_width
            ),
            s1_hits,
            hard_hits,
            easy_hits,
            easy_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_phase_c_boundary_candidate_compare() {
    let mut s1_hits = 0usize;
    let mut hard_hits = 0usize;
    let mut easy_hits = 0usize;
    let mut easy_delta_total = 0usize;
    let mut first_hit: Option<u128> = None;
    let mut result_delta_nonzero = false;

    for &x in candidate_grid_points() {
        let comparison = dr::compare_phase_c_package_vs_boundary_candidate(x);

        if comparison.current.s1_len != comparison.experimental.s1_len {
            s1_hits += 1;
        }
        if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
            hard_hits += 1;
        }
        if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
            easy_hits += 1;
            easy_delta_total += comparison
                .experimental
                .contributions
                .easy
                .abs_diff(comparison.current.contributions.easy)
                as usize;
        }
        if first_hit.is_none()
            && (comparison.current.s1_len != comparison.experimental.s1_len
                || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
        {
            first_hit = Some(x);
        }
        if comparison.current.result != comparison.experimental.result {
            result_delta_nonzero = true;
        }
    }

    println!(
        "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "reference pair", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(100));
    println!(
        "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "linked vs boundary 4:4:2",
        s1_hits,
        hard_hits,
        easy_hits,
        easy_delta_total,
        first_hit
            .map(fmt_thousands)
            .unwrap_or_else(|| "-".to_string()),
        if result_delta_nonzero {
            "non-zero"
        } else {
            "0"
        }
    );
}

fn run_phase_c_boundary_local_search() {
    let candidates = [
        rivat3::parameters::PhaseCBoundaryCandidate {
            boundary_term: 3,
            easy_width: 3,
            hard_width: 2,
        },
        rivat3::parameters::PhaseCBoundaryCandidate {
            boundary_term: 4,
            easy_width: 3,
            hard_width: 2,
        },
        rivat3::parameters::PhaseCBoundaryCandidate {
            boundary_term: 4,
            easy_width: 4,
            hard_width: 1,
        },
        rivat3::parameters::PhaseCBoundaryCandidate {
            boundary_term: 4,
            easy_width: 4,
            hard_width: 2,
        },
        rivat3::parameters::PhaseCBoundaryCandidate {
            boundary_term: 4,
            easy_width: 4,
            hard_width: 3,
        },
        rivat3::parameters::PhaseCBoundaryCandidate {
            boundary_term: 4,
            easy_width: 5,
            hard_width: 1,
        },
        rivat3::parameters::PhaseCBoundaryCandidate {
            boundary_term: 5,
            easy_width: 4,
            hard_width: 1,
        },
    ];

    println!(
        "{:>9}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "candidate", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(95));

    for candidate in candidates {
        let mut s1_hits = 0usize;
        let mut hard_hits = 0usize;
        let mut easy_hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison = dr::compare_phase_c_package_vs_experimental_boundary_candidate(
                x,
                candidate.boundary_term,
                candidate.easy_width,
                candidate.hard_width,
            );

            if comparison.current.s1_len != comparison.experimental.s1_len {
                s1_hits += 1;
            }
            if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
                hard_hits += 1;
            }
            if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
                easy_hits += 1;
                easy_delta_total += comparison
                    .experimental
                    .contributions
                    .easy
                    .abs_diff(comparison.current.contributions.easy)
                    as usize;
            }
            if first_hit.is_none()
                && (comparison.current.s1_len != comparison.experimental.s1_len
                    || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                    || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
            {
                first_hit = Some(x);
            }
            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>9}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            format!(
                "{}:{}:{}",
                candidate.boundary_term, candidate.easy_width, candidate.hard_width
            ),
            s1_hits,
            hard_hits,
            easy_hits,
            easy_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_phase_c_buffered_boundary_compare() {
    let mut s1_hits = 0usize;
    let mut hard_hits = 0usize;
    let mut easy_hits = 0usize;
    let mut easy_delta_total = 0usize;
    let mut first_hit: Option<u128> = None;
    let mut result_delta_nonzero = false;

    for &x in candidate_grid_points() {
        let comparison = dr::compare_phase_c_package_vs_buffered_boundary_package(x);

        if comparison.current.s1_len != comparison.experimental.s1_len {
            s1_hits += 1;
        }
        if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
            hard_hits += 1;
        }
        if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
            easy_hits += 1;
            easy_delta_total += comparison
                .experimental
                .contributions
                .easy
                .abs_diff(comparison.current.contributions.easy)
                as usize;
        }
        if first_hit.is_none()
            && (comparison.current.s1_len != comparison.experimental.s1_len
                || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
        {
            first_hit = Some(x);
        }
        if comparison.current.result != comparison.experimental.result {
            result_delta_nonzero = true;
        }
    }

    println!(
        "{:>24}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "reference pair", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(104));
    println!(
        "{:>24}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "linked vs buffered 4:4:1:1",
        s1_hits,
        hard_hits,
        easy_hits,
        easy_delta_total,
        first_hit
            .map(fmt_thousands)
            .unwrap_or_else(|| "-".to_string()),
        if result_delta_nonzero {
            "non-zero"
        } else {
            "0"
        }
    );
}

fn run_phase_c_buffered_boundary_search() {
    let candidates = [
        (3_u128, 3_u128, 1_u128, 1_u128),
        (3, 3, 1, 2),
        (3, 4, 1, 1),
        (4_u128, 4_u128, 1_u128, 1_u128),
        (4, 4, 1, 2),
        (4, 3, 1, 1),
        (4, 3, 1, 2),
        (4, 4, 2, 1),
        (4, 5, 1, 2),
        (5, 4, 1, 1),
        (5, 5, 1, 1),
    ];

    println!(
        "{:>11}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "candidate", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(99));

    for (boundary_term, easy_width, gap_width, hard_width) in candidates {
        let mut s1_hits = 0usize;
        let mut hard_hits = 0usize;
        let mut easy_hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison =
                dr::compare_phase_c_package_vs_experimental_buffered_boundary_candidate(
                    x,
                    boundary_term,
                    easy_width,
                    gap_width,
                    hard_width,
                );

            if comparison.current.s1_len != comparison.experimental.s1_len {
                s1_hits += 1;
            }
            if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
                hard_hits += 1;
            }
            if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
                easy_hits += 1;
                easy_delta_total += comparison
                    .experimental
                    .contributions
                    .easy
                    .abs_diff(comparison.current.contributions.easy)
                    as usize;
            }
            if first_hit.is_none()
                && (comparison.current.s1_len != comparison.experimental.s1_len
                    || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                    || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
            {
                first_hit = Some(x);
            }
            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>11}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            format!("{boundary_term}:{easy_width}:{gap_width}:{hard_width}"),
            s1_hits,
            hard_hits,
            easy_hits,
            easy_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_phase_c_quotient_window_compare() {
    let mut s1_hits = 0usize;
    let mut hard_hits = 0usize;
    let mut easy_hits = 0usize;
    let mut easy_delta_total = 0usize;
    let mut first_hit: Option<u128> = None;
    let mut result_delta_nonzero = false;

    for &x in candidate_grid_points() {
        let comparison = dr::compare_phase_c_package_vs_quotient_window_package(x);

        if comparison.current.s1_len != comparison.experimental.s1_len {
            s1_hits += 1;
        }
        if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
            hard_hits += 1;
        }
        if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
            easy_hits += 1;
            easy_delta_total += comparison
                .experimental
                .contributions
                .easy
                .abs_diff(comparison.current.contributions.easy)
                as usize;
        }
        if first_hit.is_none()
            && (comparison.current.s1_len != comparison.experimental.s1_len
                || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
        {
            first_hit = Some(x);
        }
        if comparison.current.result != comparison.experimental.result {
            result_delta_nonzero = true;
        }
    }

    println!(
        "{:>23}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "reference pair", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(103));
    println!(
        "{:>23}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "linked vs quotient 0:1",
        s1_hits,
        hard_hits,
        easy_hits,
        easy_delta_total,
        first_hit
            .map(fmt_thousands)
            .unwrap_or_else(|| "-".to_string()),
        if result_delta_nonzero {
            "non-zero"
        } else {
            "0"
        }
    );
}

fn run_phase_c_quotient_window_search() {
    let candidates = [(0_u128, 1_u128), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)];

    println!(
        "{:>9}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "candidate", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(95));

    for (easy_q_offset_max, hard_q_width) in candidates {
        let mut s1_hits = 0usize;
        let mut hard_hits = 0usize;
        let mut easy_hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison = dr::compare_phase_c_package_vs_experimental_quotient_window_candidate(
                x,
                easy_q_offset_max,
                hard_q_width,
            );

            if comparison.current.s1_len != comparison.experimental.s1_len {
                s1_hits += 1;
            }
            if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
                hard_hits += 1;
            }
            if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
                easy_hits += 1;
                easy_delta_total += comparison
                    .experimental
                    .contributions
                    .easy
                    .abs_diff(comparison.current.contributions.easy)
                    as usize;
            }
            if first_hit.is_none()
                && (comparison.current.s1_len != comparison.experimental.s1_len
                    || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                    || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
            {
                first_hit = Some(x);
            }
            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>9}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            format!("{easy_q_offset_max}:{hard_q_width}"),
            s1_hits,
            hard_hits,
            easy_hits,
            easy_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_phase_c_quotient_window_shifted_search() {
    let candidates = [
        (1_u128, 1_u128, 1_u128),
        (1, 1, 2),
        (1, 2, 1),
        (1, 2, 2),
        (2, 2, 1),
        (2, 2, 2),
    ];

    println!(
        "{:>11}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "candidate", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(99));

    for (easy_q_offset_min, easy_q_offset_max, hard_q_width) in candidates {
        let mut s1_hits = 0usize;
        let mut hard_hits = 0usize;
        let mut easy_hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison =
                dr::compare_phase_c_package_vs_experimental_shifted_quotient_window_candidate(
                    x,
                    easy_q_offset_min,
                    easy_q_offset_max,
                    hard_q_width,
                );

            if comparison.current.s1_len != comparison.experimental.s1_len {
                s1_hits += 1;
            }
            if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
                hard_hits += 1;
            }
            if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
                easy_hits += 1;
                easy_delta_total += comparison
                    .experimental
                    .contributions
                    .easy
                    .abs_diff(comparison.current.contributions.easy)
                    as usize;
            }
            if first_hit.is_none()
                && (comparison.current.s1_len != comparison.experimental.s1_len
                    || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                    || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
            {
                first_hit = Some(x);
            }
            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>11}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            format!("{easy_q_offset_min}:{easy_q_offset_max}:{hard_q_width}"),
            s1_hits,
            hard_hits,
            easy_hits,
            easy_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_phase_c_boundary_quotient_guard_compare() {
    let mut s1_hits = 0usize;
    let mut hard_hits = 0usize;
    let mut easy_hits = 0usize;
    let mut easy_delta_total = 0usize;
    let mut first_hit: Option<u128> = None;
    let mut result_delta_nonzero = false;

    for &x in candidate_grid_points() {
        let comparison = dr::compare_phase_c_package_vs_boundary_quotient_guard_package(x);

        if comparison.current.s1_len != comparison.experimental.s1_len {
            s1_hits += 1;
        }
        if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
            hard_hits += 1;
        }
        if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
            easy_hits += 1;
            easy_delta_total += comparison
                .experimental
                .contributions
                .easy
                .abs_diff(comparison.current.contributions.easy)
                as usize;
        }
        if first_hit.is_none()
            && (comparison.current.s1_len != comparison.experimental.s1_len
                || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
        {
            first_hit = Some(x);
        }
        if comparison.current.result != comparison.experimental.result {
            result_delta_nonzero = true;
        }
    }

    println!(
        "{:>27}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "reference pair", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(107));
    println!(
        "{:>27}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "linked vs boundary+qguard",
        s1_hits,
        hard_hits,
        easy_hits,
        easy_delta_total,
        first_hit
            .map(fmt_thousands)
            .unwrap_or_else(|| "-".to_string()),
        if result_delta_nonzero {
            "non-zero"
        } else {
            "0"
        }
    );
}

fn run_phase_c_boundary_quotient_guard_search() {
    let candidates = [0_u128, 1_u128, 2_u128, 3_u128];

    println!(
        "{:>11}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "guard_q", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(99));

    for guard_q_offset in candidates {
        let mut s1_hits = 0usize;
        let mut hard_hits = 0usize;
        let mut easy_hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison =
                dr::compare_phase_c_package_vs_experimental_boundary_quotient_guard_candidate(
                    x,
                    4,
                    4,
                    2,
                    guard_q_offset,
                );

            if comparison.current.s1_len != comparison.experimental.s1_len {
                s1_hits += 1;
            }
            if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
                hard_hits += 1;
            }
            if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
                easy_hits += 1;
                easy_delta_total += comparison
                    .experimental
                    .contributions
                    .easy
                    .abs_diff(comparison.current.contributions.easy)
                    as usize;
            }
            if first_hit.is_none()
                && (comparison.current.s1_len != comparison.experimental.s1_len
                    || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                    || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
            {
                first_hit = Some(x);
            }
            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>11}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            guard_q_offset,
            s1_hits,
            hard_hits,
            easy_hits,
            easy_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_phase_c_boundary_relative_quotient_band_compare() {
    let mut s1_hits = 0usize;
    let mut hard_hits = 0usize;
    let mut easy_hits = 0usize;
    let mut easy_delta_total = 0usize;
    let mut first_hit: Option<u128> = None;
    let mut result_delta_nonzero = false;

    for &x in candidate_grid_points() {
        let comparison = dr::compare_phase_c_package_vs_boundary_relative_quotient_band_package(x);

        if comparison.current.s1_len != comparison.experimental.s1_len {
            s1_hits += 1;
        }
        if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
            hard_hits += 1;
        }
        if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
            easy_hits += 1;
            easy_delta_total += comparison
                .experimental
                .contributions
                .easy
                .abs_diff(comparison.current.contributions.easy)
                as usize;
        }
        if first_hit.is_none()
            && (comparison.current.s1_len != comparison.experimental.s1_len
                || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
        {
            first_hit = Some(x);
        }
        if comparison.current.result != comparison.experimental.result {
            result_delta_nonzero = true;
        }
    }

    println!(
        "{:>33}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "reference pair", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(113));
    println!(
        "{:>33}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "linked vs boundary+relativeq",
        s1_hits,
        hard_hits,
        easy_hits,
        easy_delta_total,
        first_hit
            .map(fmt_thousands)
            .unwrap_or_else(|| "-".to_string()),
        if result_delta_nonzero {
            "non-zero"
        } else {
            "0"
        }
    );
}

fn run_phase_c_boundary_relative_quotient_band_search() {
    let candidates = [(0_u128, 0_u128), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)];

    println!(
        "{:>11}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "bands", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(99));

    for (easy_q_band_width, hard_q_band_width) in candidates {
        let mut s1_hits = 0usize;
        let mut hard_hits = 0usize;
        let mut easy_hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison =
                dr::compare_phase_c_package_vs_experimental_boundary_relative_quotient_band_candidate(
                    x, 4, 4, 2, easy_q_band_width, hard_q_band_width,
                );

            if comparison.current.s1_len != comparison.experimental.s1_len {
                s1_hits += 1;
            }
            if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
                hard_hits += 1;
            }
            if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
                easy_hits += 1;
                easy_delta_total += comparison
                    .experimental
                    .contributions
                    .easy
                    .abs_diff(comparison.current.contributions.easy)
                    as usize;
            }
            if first_hit.is_none()
                && (comparison.current.s1_len != comparison.experimental.s1_len
                    || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                    || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
            {
                first_hit = Some(x);
            }
            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>11}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            format!("{easy_q_band_width}:{hard_q_band_width}"),
            s1_hits,
            hard_hits,
            easy_hits,
            easy_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_phase_c_boundary_relative_quotient_step_band_compare() {
    let mut s1_hits = 0usize;
    let mut hard_hits = 0usize;
    let mut easy_hits = 0usize;
    let mut easy_delta_total = 0usize;
    let mut first_hit: Option<u128> = None;
    let mut result_delta_nonzero = false;

    for &x in candidate_grid_points() {
        let comparison =
            dr::compare_phase_c_package_vs_boundary_relative_quotient_step_band_package(x);

        if comparison.current.s1_len != comparison.experimental.s1_len {
            s1_hits += 1;
        }
        if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
            hard_hits += 1;
        }
        if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
            easy_hits += 1;
            easy_delta_total += comparison
                .experimental
                .contributions
                .easy
                .abs_diff(comparison.current.contributions.easy)
                as usize;
        }
        if first_hit.is_none()
            && (comparison.current.s1_len != comparison.experimental.s1_len
                || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
        {
            first_hit = Some(x);
        }
        if comparison.current.result != comparison.experimental.result {
            result_delta_nonzero = true;
        }
    }

    println!(
        "{:>38}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "reference pair", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(118));
    println!(
        "{:>38}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "linked vs boundary+relativeq-step",
        s1_hits,
        hard_hits,
        easy_hits,
        easy_delta_total,
        first_hit
            .map(fmt_thousands)
            .unwrap_or_else(|| "-".to_string()),
        if result_delta_nonzero {
            "non-zero"
        } else {
            "0"
        }
    );
}

fn run_phase_c_boundary_relative_quotient_step_band_search() {
    let candidates = [(0_u128, 0_u128), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)];

    println!(
        "{:>11}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "steps", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(99));

    for (easy_q_step_multiplier, hard_q_step_multiplier) in candidates {
        let mut s1_hits = 0usize;
        let mut hard_hits = 0usize;
        let mut easy_hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison = dr::compare_phase_c_package_vs_experimental_boundary_relative_quotient_step_band_candidate(
                x,
                4,
                4,
                2,
                easy_q_step_multiplier,
                hard_q_step_multiplier,
            );

            if comparison.current.s1_len != comparison.experimental.s1_len {
                s1_hits += 1;
            }
            if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
                hard_hits += 1;
            }
            if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
                easy_hits += 1;
                easy_delta_total += comparison
                    .experimental
                    .contributions
                    .easy
                    .abs_diff(comparison.current.contributions.easy)
                    as usize;
            }
            if first_hit.is_none()
                && (comparison.current.s1_len != comparison.experimental.s1_len
                    || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                    || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
            {
                first_hit = Some(x);
            }
            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>11}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            format!("{easy_q_step_multiplier}:{hard_q_step_multiplier}"),
            s1_hits,
            hard_hits,
            easy_hits,
            easy_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_phase_c_step_band_local_search() {
    let candidates = [
        (4_u128, 4_u128, 2_u128, 1_u128, 1_u128),
        (4, 4, 2, 1, 2),
        (4, 4, 2, 2, 1),
        (4, 4, 2, 2, 2),
        (4, 5, 1, 1, 1),
        (4, 5, 2, 1, 1),
        (5, 4, 1, 1, 1),
        (5, 5, 1, 1, 1),
        (3, 3, 2, 1, 1),
    ];

    println!(
        "{:>19}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "candidate", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(107));

    for (boundary_term, easy_width, hard_width, easy_q_step_multiplier, hard_q_step_multiplier) in
        candidates
    {
        let mut s1_hits = 0usize;
        let mut hard_hits = 0usize;
        let mut easy_hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison =
                dr::compare_phase_c_package_vs_experimental_boundary_relative_quotient_step_band_candidate(
                    x,
                    boundary_term,
                    easy_width,
                    hard_width,
                    easy_q_step_multiplier,
                    hard_q_step_multiplier,
                );

            if comparison.current.s1_len != comparison.experimental.s1_len {
                s1_hits += 1;
            }
            if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
                hard_hits += 1;
            }
            if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
                easy_hits += 1;
                easy_delta_total += comparison
                    .experimental
                    .contributions
                    .easy
                    .abs_diff(comparison.current.contributions.easy)
                    as usize;
            }
            if first_hit.is_none()
                && (comparison.current.s1_len != comparison.experimental.s1_len
                    || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                    || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
            {
                first_hit = Some(x);
            }
            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>19}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            format!(
                "{}:{}:{}:{}:{}",
                boundary_term,
                easy_width,
                hard_width,
                easy_q_step_multiplier,
                hard_q_step_multiplier
            ),
            s1_hits,
            hard_hits,
            easy_hits,
            easy_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_phase_c_boundary_vs_relative_quotient_step_dense() {
    let mut s1_hits = 0usize;
    let mut hard_hits = 0usize;
    let mut easy_hits = 0usize;
    let mut easy_delta_total = 0usize;
    let mut first_hit: Option<u128> = None;
    let mut result_delta_nonzero = false;

    for &x in phase_c_dense_points() {
        let comparison =
            dr::compare_boundary_candidate_vs_experimental_boundary_relative_quotient_step_band_candidate(
                x, 4, 4, 2, 4, 4, 2, 1, 1,
            );

        if comparison.current.s1_len != comparison.experimental.s1_len {
            s1_hits += 1;
        }
        if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
            hard_hits += 1;
        }
        if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
            easy_hits += 1;
            easy_delta_total += comparison
                .experimental
                .contributions
                .easy
                .abs_diff(comparison.current.contributions.easy)
                as usize;
        }
        if first_hit.is_none()
            && (comparison.current.s1_len != comparison.experimental.s1_len
                || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
        {
            first_hit = Some(x);
        }
        if comparison.current.result != comparison.experimental.result {
            result_delta_nonzero = true;
        }
    }

    println!(
        "{:>36}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "reference pair", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(116));
    println!(
        "{:>36}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "boundary 4:4:2 vs relativeq-step 1:1",
        s1_hits,
        hard_hits,
        easy_hits,
        easy_delta_total,
        first_hit
            .map(fmt_thousands)
            .unwrap_or_else(|| "-".to_string()),
        if result_delta_nonzero {
            "non-zero"
        } else {
            "0"
        }
    );
}

fn run_phase_c_easy_specialized_grid() {
    println!(
        "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}",
        "x",
        "easy len",
        "residual",
        "trans.",
        "special.",
        "residual sum",
        "trans. sum",
        "special. sum"
    );
    println!("{}", "-".repeat(126));

    for &x in candidate_grid_points() {
        let analysis = dr::analyze_easy_specialized(x);
        println!(
            "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}",
            fmt_thousands(x),
            analysis.easy_len,
            analysis.residual_len,
            analysis.transition_len,
            analysis.specialized_len,
            fmt_thousands(analysis.residual_sum),
            fmt_thousands(analysis.transition_sum),
            fmt_thousands(analysis.specialized_sum),
        );
    }
}

fn run_phase_c_easy_specialized_compare() {
    println!(
        "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}  {:>14}  {:>14}  {:>14}",
        "x",
        "easy len",
        "residual",
        "trans.",
        "special.",
        "sum check",
        "residual sum",
        "trans. sum",
        "special. sum"
    );
    println!("{}", "-".repeat(132));

    for &x in candidate_grid_points() {
        let step_band =
            dr::analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                x, 4, 4, 2, 1, 1,
            );
        let specialized = dr::analyze_easy_specialized(x);
        let sum_check = specialized.easy_sum == step_band.contributions.easy;

        println!(
            "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}  {:>14}  {:>14}  {:>14}",
            fmt_thousands(x),
            specialized.easy_len,
            specialized.residual_len,
            specialized.transition_len,
            specialized.specialized_len,
            if sum_check { "ok" } else { "mismatch" },
            fmt_thousands(specialized.residual_sum),
            fmt_thousands(specialized.transition_sum),
            fmt_thousands(specialized.specialized_sum),
        );
    }
}

fn run_phase_c_hard_specialized_grid() {
    println!(
        "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}",
        "x",
        "hard len",
        "residual",
        "trans.",
        "special.",
        "residual sum",
        "trans. sum",
        "special. sum"
    );
    println!("{}", "-".repeat(126));

    for &x in candidate_grid_points() {
        let analysis = dr::analyze_hard_specialized(x);
        println!(
            "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}",
            fmt_thousands(x),
            analysis.hard_len,
            analysis.residual_len,
            analysis.transition_len,
            analysis.specialized_len,
            fmt_thousands(analysis.residual_sum),
            fmt_thousands(analysis.transition_sum),
            fmt_thousands(analysis.specialized_sum),
        );
    }
}

fn run_phase_c_hard_specialized_compare() {
    println!(
        "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}  {:>14}  {:>14}  {:>14}",
        "x",
        "hard len",
        "residual",
        "trans.",
        "special.",
        "sum check",
        "residual sum",
        "trans. sum",
        "special. sum"
    );
    println!("{}", "-".repeat(132));

    for &x in candidate_grid_points() {
        let hard_analysis =
            dr::analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                x, 4, 4, 2, 1, 1,
            );
        let specialized = dr::analyze_hard_specialized(x);
        let sum_check = specialized.hard_sum == hard_analysis.contributions.hard;

        println!(
            "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}  {:>14}  {:>14}  {:>14}",
            fmt_thousands(x),
            specialized.hard_len,
            specialized.residual_len,
            specialized.transition_len,
            specialized.specialized_len,
            if sum_check { "ok" } else { "mismatch" },
            fmt_thousands(specialized.residual_sum),
            fmt_thousands(specialized.transition_sum),
            fmt_thousands(specialized.specialized_sum),
        );
    }
}

fn run_phase_c_ordinary_specialized_grid() {
    println!(
        "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}  {:>14}",
        "x",
        "ordinary len",
        "residual",
        "pretr.",
        "trans.",
        "special.",
        "residual sum",
        "pretr. sum",
        "trans. sum",
        "special. sum"
    );
    println!("{}", "-".repeat(158));

    for &x in candidate_grid_points() {
        let analysis = dr::analyze_ordinary_specialized(x);
        println!(
            "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}  {:>14}",
            fmt_thousands(x),
            analysis.ordinary_len,
            analysis.residual_len,
            analysis.pretransition_len,
            analysis.transition_len,
            analysis.specialized_len,
            fmt_thousands(analysis.residual_sum),
            fmt_thousands(analysis.pretransition_sum),
            fmt_thousands(analysis.transition_sum),
            fmt_thousands(analysis.specialized_sum),
        );
    }
}

fn run_phase_c_ordinary_specialized_compare() {
    println!(
        "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}  {:>14}  {:>14}  {:>14}  {:>14}",
        "x",
        "ordinary len",
        "residual",
        "pretr.",
        "trans.",
        "special.",
        "sum check",
        "residual sum",
        "pretr. sum",
        "trans. sum",
        "special. sum"
    );
    println!("{}", "-".repeat(174));

    for &x in candidate_grid_points() {
        let ordinary_analysis =
            dr::analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                x, 4, 4, 2, 1, 1,
            );
        let specialized = dr::analyze_ordinary_specialized(x);
        let sum_check = specialized.ordinary_sum == ordinary_analysis.contributions.ordinary;

        println!(
            "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}  {:>14}  {:>14}  {:>14}  {:>14}",
            fmt_thousands(x),
            specialized.ordinary_len,
            specialized.residual_len,
            specialized.pretransition_len,
            specialized.transition_len,
            specialized.specialized_len,
            if sum_check { "ok" } else { "mismatch" },
            fmt_thousands(specialized.residual_sum),
            fmt_thousands(specialized.pretransition_sum),
            fmt_thousands(specialized.transition_sum),
            fmt_thousands(specialized.specialized_sum),
        );
    }
}

fn run_phase_c_ordinary_relative_quotient_grid() {
    println!(
        "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}",
        "x", "ordinary len", "left", "region", "right", "left sum", "region sum", "right sum"
    );
    println!("{}", "-".repeat(126));

    for &x in candidate_grid_points() {
        let analysis = dr::analyze_ordinary_relative_quotient_region(x);
        println!(
            "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}",
            fmt_thousands(x),
            analysis.ordinary_len,
            analysis.left_residual_len,
            analysis.region_len,
            analysis.right_residual_len,
            fmt_thousands(analysis.left_residual_sum),
            fmt_thousands(analysis.region_sum),
            fmt_thousands(analysis.right_residual_sum),
        );
    }
}

fn run_phase_c_ordinary_relative_quotient_compare() {
    println!(
        "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>12}  {:>14}  {:>14}  {:>14}",
        "x",
        "ordinary len",
        "left",
        "region",
        "right",
        "sum check",
        "left sum",
        "region sum",
        "right sum"
    );
    println!("{}", "-".repeat(140));

    for &x in candidate_grid_points() {
        let ordinary_analysis =
            dr::analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                x, 4, 4, 2, 1, 1,
            );
        let specialized = dr::analyze_ordinary_relative_quotient_region(x);
        let sum_check = specialized.ordinary_sum == ordinary_analysis.contributions.ordinary;

        println!(
            "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>12}  {:>14}  {:>14}  {:>14}",
            fmt_thousands(x),
            specialized.ordinary_len,
            specialized.left_residual_len,
            specialized.region_len,
            specialized.right_residual_len,
            if sum_check { "ok" } else { "mismatch" },
            fmt_thousands(specialized.left_residual_sum),
            fmt_thousands(specialized.region_sum),
            fmt_thousands(specialized.right_residual_sum),
        );
    }
}

fn run_phase_c_ordinary_relative_quotient_vs_specialized() {
    println!(
        "{:>22}  {:>14}  {:>14}  {:>14}  {:>16}  {:>16}",
        "x", "ordinary len", "term. len", "rel. len", "term. sum", "rel. sum"
    );
    println!("{}", "-".repeat(112));

    for &x in candidate_grid_points() {
        let comparison = dr::compare_ordinary_specialized_vs_relative_quotient_region(x);
        println!(
            "{:>22}  {:>14}  {:>14}  {:>14}  {:>16}  {:>16}",
            fmt_thousands(x),
            comparison.ordinary_len,
            comparison.current_terminal_len,
            comparison.relative_region_len,
            fmt_thousands(comparison.current_terminal_sum),
            fmt_thousands(comparison.relative_region_sum),
        );
    }
}

fn run_phase_c_ordinary_relative_quotient_search() {
    let candidates = [(-1_isize, 1_u128), (0, 1), (1, 1), (0, 2), (1, 2)];

    println!(
        "{:>16}  {:>12}  {:>14}  {:>16}  {:>10}",
        "variant", "hits", "region avg", "region sum Δ", "exact"
    );
    println!("{}", "-".repeat(82));

    for (shift, scale) in candidates {
        let mut hits = 0usize;
        let mut total_region_len = 0usize;
        let mut total_region_delta = 0u128;
        let mut exact = true;

        for &x in candidate_grid_points() {
            let baseline = dr::analyze_ordinary_relative_quotient_region(x);
            let variant = dr::analyze_ordinary_relative_quotient_region_variant(x, shift, scale);
            if baseline.region_len != variant.region_len
                || baseline.region_sum != variant.region_sum
            {
                hits += 1;
            }
            total_region_len += variant.region_len;
            total_region_delta += baseline.region_sum.abs_diff(variant.region_sum);
            exact &= baseline.ordinary_sum == variant.ordinary_sum;
        }

        let avg = total_region_len as f64 / candidate_grid_points().len() as f64;
        println!(
            "{:>16}  {:>12}  {:>14.2}  {:>16}  {:>10}",
            format!("{}:{}", shift, scale),
            hits,
            avg,
            fmt_thousands(total_region_delta),
            if exact { "yes" } else { "no" }
        );
    }
}

fn run_post_plateau_triptych_compare() {
    println!(
        "{:>22}  {:>12}  {:>14}  {:>10}  {:>14}  {:>10}  {:>14}  {:>10}  {:>14}  {:>10}  {:>14}  {:>12}  {:>14}",
        "x",
        "easy len",
        "easy focus",
        "ord.term",
        "ord.term sum",
        "ord.reg",
        "ord.reg sum",
        "ord.asm",
        "ord.asm sum",
        "ord.qlit",
        "ord.qlit sum",
        "hard len",
        "hard focus"
    );
    println!("{}", "-".repeat(199));

    for &x in candidate_grid_points() {
        let analysis = dr::analyze_post_plateau_triptych(x);
        println!(
            "{:>22}  {:>12}  {:>14}  {:>10}  {:>14}  {:>10}  {:>14}  {:>10}  {:>14}  {:>10}  {:>14}  {:>12}  {:>14}",
            fmt_thousands(x),
            analysis.easy_len,
            format!(
                "{} ({})",
                analysis.easy_focus_len,
                fmt_thousands(analysis.easy_focus_sum)
            ),
            analysis.ordinary_terminal_len,
            fmt_thousands(analysis.ordinary_terminal_sum),
            analysis.ordinary_region_len,
            fmt_thousands(analysis.ordinary_region_sum),
            analysis.ordinary_assembly_core_len,
            fmt_thousands(analysis.ordinary_assembly_core_sum),
            analysis.ordinary_quasi_literature_middle_len,
            fmt_thousands(analysis.ordinary_quasi_literature_middle_sum),
            analysis.hard_len,
            format!(
                "{} ({})",
                analysis.hard_focus_len,
                fmt_thousands(analysis.hard_focus_sum)
            ),
        );
    }
}

fn run_post_plateau_ordinary_shoulder_grid() {
    println!(
        "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}",
        "x",
        "ordinary len",
        "left sh.",
        "core",
        "right sh.",
        "q_step",
        "left sh. sum",
        "core sum",
        "right sh. sum"
    );
    println!("{}", "-".repeat(146));

    for &x in candidate_grid_points() {
        let analysis = dr::analyze_ordinary_relative_quotient_shoulder_region(x);
        println!(
            "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}",
            fmt_thousands(x),
            analysis.ordinary_len,
            analysis.left_shoulder_len,
            analysis.core_len,
            analysis.right_shoulder_len,
            analysis.q_step,
            fmt_thousands(analysis.left_shoulder_sum),
            fmt_thousands(analysis.core_sum),
            fmt_thousands(analysis.right_shoulder_sum),
        );
    }
}

fn run_post_plateau_ordinary_shoulder_search() {
    let candidates = [
        (0_isize, 1_u128, 2_u128),
        (0, 1, 3),
        (1, 1, 2),
        (1, 1, 3),
        (0, 2, 3),
    ];

    println!(
        "{:>12}  {:>10}  {:>12}  {:>14}  {:>10}",
        "variant", "hits", "core avg", "shoulder avg", "exact"
    );
    println!("{}", "-".repeat(68));

    for (shift, core_scale, shoulder_scale) in candidates {
        let mut hits = 0usize;
        let mut total_core = 0usize;
        let mut total_shoulder = 0usize;
        let mut exact = true;

        for &x in candidate_grid_points() {
            let baseline = dr::analyze_ordinary_relative_quotient_shoulder_region(x);
            let variant = dr::analyze_ordinary_relative_quotient_shoulder_region_variant(
                x,
                shift,
                core_scale,
                shoulder_scale,
            );
            if baseline.core_len != variant.core_len
                || baseline.left_shoulder_len != variant.left_shoulder_len
                || baseline.right_shoulder_len != variant.right_shoulder_len
                || baseline.core_sum != variant.core_sum
                || baseline.left_shoulder_sum != variant.left_shoulder_sum
                || baseline.right_shoulder_sum != variant.right_shoulder_sum
            {
                hits += 1;
            }
            total_core += variant.core_len;
            total_shoulder += variant.left_shoulder_len + variant.right_shoulder_len;
            exact &= baseline.ordinary_sum == variant.ordinary_sum;
        }

        let n = candidate_grid_points().len() as f64;
        println!(
            "{:>12}  {:>10}  {:>12.2}  {:>14.2}  {:>10}",
            format!("{}:{}:{}", shift, core_scale, shoulder_scale),
            hits,
            total_core as f64 / n,
            total_shoulder as f64 / n,
            if exact { "yes" } else { "no" }
        );
    }
}

fn run_post_plateau_ordinary_envelope_grid() {
    println!(
        "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}",
        "x",
        "ordinary len",
        "left env.",
        "core",
        "right env.",
        "q_step",
        "left env. sum",
        "core sum",
        "right env. sum"
    );
    println!("{}", "-".repeat(146));

    for &x in candidate_grid_points() {
        let analysis = dr::analyze_ordinary_relative_quotient_envelope_region(x);
        println!(
            "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}",
            fmt_thousands(x),
            analysis.ordinary_len,
            analysis.left_envelope_len,
            analysis.core_len,
            analysis.right_envelope_len,
            analysis.q_step,
            fmt_thousands(analysis.left_envelope_sum),
            fmt_thousands(analysis.core_sum),
            fmt_thousands(analysis.right_envelope_sum),
        );
    }
}

fn run_post_plateau_ordinary_envelope_vs_shoulder() {
    println!(
        "{:>22}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}  {:>14}",
        "x", "sh. avg", "env. avg", "sh. sum", "env. sum", "core sum", "ordinary sum"
    );
    println!("{}", "-".repeat(118));

    for &x in candidate_grid_points() {
        let shoulder = dr::analyze_ordinary_relative_quotient_shoulder_region(x);
        let envelope = dr::analyze_ordinary_relative_quotient_envelope_region(x);
        let shoulder_len = shoulder.left_shoulder_len + shoulder.right_shoulder_len;
        let shoulder_sum = shoulder.left_shoulder_sum + shoulder.right_shoulder_sum;
        let envelope_len = envelope.left_envelope_len + envelope.right_envelope_len;
        let envelope_sum = envelope.left_envelope_sum + envelope.right_envelope_sum;

        println!(
            "{:>22}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}  {:>14}",
            fmt_thousands(x),
            shoulder_len,
            envelope_len,
            fmt_thousands(shoulder_sum),
            fmt_thousands(envelope_sum),
            fmt_thousands(envelope.core_sum),
            fmt_thousands(envelope.ordinary_sum),
        );
    }
}

fn run_post_plateau_ordinary_envelope_search() {
    let candidates = [
        (0_isize, 1_u128, 3_u128),
        (0, 1, 4),
        (1, 1, 3),
        (1, 1, 4),
        (0, 2, 4),
    ];

    println!(
        "{:>12}  {:>10}  {:>12}  {:>14}  {:>10}",
        "variant", "hits", "core avg", "env. avg", "exact"
    );
    println!("{}", "-".repeat(68));

    for (shift, core_scale, env_scale) in candidates {
        let mut hits = 0usize;
        let mut total_core = 0usize;
        let mut total_env = 0usize;
        let mut exact = true;

        for &x in candidate_grid_points() {
            let baseline = dr::analyze_ordinary_relative_quotient_envelope_region(x);
            let variant = dr::analyze_ordinary_relative_quotient_envelope_region_variant(
                x, shift, core_scale, env_scale,
            );
            if baseline.core_len != variant.core_len
                || baseline.left_envelope_len != variant.left_envelope_len
                || baseline.right_envelope_len != variant.right_envelope_len
                || baseline.core_sum != variant.core_sum
                || baseline.left_envelope_sum != variant.left_envelope_sum
                || baseline.right_envelope_sum != variant.right_envelope_sum
            {
                hits += 1;
            }
            total_core += variant.core_len;
            total_env += variant.left_envelope_len + variant.right_envelope_len;
            exact &= baseline.ordinary_sum == variant.ordinary_sum;
        }

        let n = candidate_grid_points().len() as f64;
        println!(
            "{:>12}  {:>10}  {:>12.2}  {:>14.2}  {:>10}",
            format!("{}:{}:{}", shift, core_scale, env_scale),
            hits,
            total_core as f64 / n,
            total_env as f64 / n,
            if exact { "yes" } else { "no" }
        );
    }
}

fn run_post_plateau_ordinary_hierarchy_grid() {
    println!(
        "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "x",
        "ordinary len",
        "left out.",
        "left near",
        "core",
        "right near",
        "right out.",
        "q_step",
        "core sum"
    );
    println!("{}", "-".repeat(146));

    for &x in candidate_grid_points() {
        let analysis = dr::analyze_ordinary_relative_quotient_hierarchy_region(x);
        println!(
            "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            fmt_thousands(x),
            analysis.ordinary_len,
            analysis.left_outer_band_len,
            analysis.left_near_band_len,
            analysis.inner_core_len,
            analysis.right_near_band_len,
            analysis.right_outer_band_len,
            analysis.q_step,
            fmt_thousands(analysis.inner_core_sum),
        );
    }
}

fn run_post_plateau_ordinary_hierarchy_vs_envelope() {
    println!(
        "{:>22}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}  {:>14}",
        "x", "env. len", "hier. len", "env. sum", "hier. sum", "core sum", "ordinary sum"
    );
    println!("{}", "-".repeat(118));

    for &x in candidate_grid_points() {
        let envelope = dr::analyze_ordinary_relative_quotient_envelope_region(x);
        let hierarchy = dr::analyze_ordinary_relative_quotient_hierarchy_region(x);
        let envelope_len = envelope.left_envelope_len + envelope.right_envelope_len;
        let envelope_sum = envelope.left_envelope_sum + envelope.right_envelope_sum;
        let hierarchy_len = hierarchy.left_outer_band_len
            + hierarchy.left_near_band_len
            + hierarchy.right_near_band_len
            + hierarchy.right_outer_band_len;
        let hierarchy_sum = hierarchy.left_outer_band_sum
            + hierarchy.left_near_band_sum
            + hierarchy.right_near_band_sum
            + hierarchy.right_outer_band_sum;

        println!(
            "{:>22}  {:>10}  {:>10}  {:>14}  {:>14}  {:>14}  {:>14}",
            fmt_thousands(x),
            envelope_len,
            hierarchy_len,
            fmt_thousands(envelope_sum),
            fmt_thousands(hierarchy_sum),
            fmt_thousands(hierarchy.inner_core_sum),
            fmt_thousands(hierarchy.ordinary_sum),
        );
    }
}

fn run_post_plateau_ordinary_assembly_grid() {
    println!(
        "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "x", "ordinary len", "left out.", "left adj.", "central", "right adj.", "right out."
    );
    println!("{}", "-".repeat(114));

    for &x in candidate_grid_points() {
        let analysis = dr::analyze_ordinary_region_assembly(x);
        println!(
            "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            fmt_thousands(x),
            analysis.ordinary_len,
            analysis.left_outer_support_len,
            analysis.left_adjacent_support_len,
            analysis.central_assembly_len,
            analysis.right_adjacent_support_len,
            analysis.right_outer_support_len,
        );
    }
}

fn run_post_plateau_ordinary_assembly_vs_hierarchy() {
    println!(
        "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}",
        "x", "hier. core", "asm. core", "hier. sup", "asm. sup", "hier. sum", "asm. sum"
    );
    println!("{}", "-".repeat(110));

    for &x in candidate_grid_points() {
        let hierarchy = dr::analyze_ordinary_relative_quotient_hierarchy_region(x);
        let assembly = dr::analyze_ordinary_region_assembly(x);
        let hierarchy_support_len = hierarchy.left_outer_band_len
            + hierarchy.left_near_band_len
            + hierarchy.right_near_band_len
            + hierarchy.right_outer_band_len;
        let assembly_support_len = assembly.left_outer_support_len
            + assembly.left_adjacent_support_len
            + assembly.right_adjacent_support_len
            + assembly.right_outer_support_len;

        println!(
            "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}",
            fmt_thousands(x),
            hierarchy.inner_core_len,
            assembly.central_assembly_len,
            hierarchy_support_len,
            assembly_support_len,
            fmt_thousands(hierarchy.inner_core_sum),
            fmt_thousands(assembly.central_assembly_sum),
        );
    }
}

fn run_post_plateau_ordinary_quasi_literature_grid() {
    println!(
        "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}",
        "x", "ordinary len", "left out.", "middle", "right out."
    );
    println!("{}", "-".repeat(78));

    for &x in candidate_grid_points() {
        let analysis = dr::analyze_ordinary_quasi_literature_region(x);
        println!(
            "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}",
            fmt_thousands(x),
            analysis.ordinary_len,
            analysis.left_outer_work_len,
            analysis.middle_work_len,
            analysis.right_outer_work_len,
        );
    }
}

fn run_post_plateau_ordinary_quasi_literature_vs_assembly() {
    println!(
        "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}",
        "x", "asm. core", "qlit mid", "asm. sup", "qlit out", "asm. sum", "qlit sum"
    );
    println!("{}", "-".repeat(110));

    for &x in candidate_grid_points() {
        let assembly = dr::analyze_ordinary_region_assembly(x);
        let qlit = dr::analyze_ordinary_quasi_literature_region(x);
        let assembly_support_len = assembly.left_outer_support_len
            + assembly.left_adjacent_support_len
            + assembly.right_adjacent_support_len
            + assembly.right_outer_support_len;
        let qlit_outer_len = qlit.left_outer_work_len + qlit.right_outer_work_len;
        let assembly_sum = assembly.left_outer_support_sum
            + assembly.left_adjacent_support_sum
            + assembly.central_assembly_sum
            + assembly.right_adjacent_support_sum
            + assembly.right_outer_support_sum;
        let qlit_sum = qlit.left_outer_work_sum + qlit.middle_work_sum + qlit.right_outer_work_sum;

        println!(
            "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}",
            fmt_thousands(x),
            assembly.central_assembly_len,
            qlit.middle_work_len,
            assembly_support_len,
            qlit_outer_len,
            fmt_thousands(assembly_sum),
            fmt_thousands(qlit_sum),
        );
    }
}

fn run_post_plateau_ordinary_quasi_literature_vs_assembly_dense() {
    let mut middle_hits = 0usize;
    let mut outer_hits = 0usize;
    let mut sum_delta_total = 0u128;
    let mut max_middle_gain = 0usize;
    let mut first_hit: Option<u128> = None;

    for &x in phase_c_dense_points() {
        let assembly = dr::analyze_ordinary_region_assembly(x);
        let qlit = dr::analyze_ordinary_quasi_literature_region(x);

        let assembly_support_len = assembly.left_outer_support_len
            + assembly.left_adjacent_support_len
            + assembly.right_adjacent_support_len
            + assembly.right_outer_support_len;
        let qlit_outer_len = qlit.left_outer_work_len + qlit.right_outer_work_len;
        let assembly_total_sum = assembly.left_outer_support_sum
            + assembly.left_adjacent_support_sum
            + assembly.central_assembly_sum
            + assembly.right_adjacent_support_sum
            + assembly.right_outer_support_sum;
        let qlit_total_sum =
            qlit.left_outer_work_sum + qlit.middle_work_sum + qlit.right_outer_work_sum;

        if qlit.middle_work_len != assembly.central_assembly_len {
            middle_hits += 1;
            let gain = qlit.middle_work_len.abs_diff(assembly.central_assembly_len);
            max_middle_gain = max_middle_gain.max(gain);
            if first_hit.is_none() {
                first_hit = Some(x);
            }
        }
        if qlit_outer_len != assembly_support_len {
            outer_hits += 1;
            if first_hit.is_none() {
                first_hit = Some(x);
            }
        }

        sum_delta_total += qlit_total_sum.abs_diff(assembly_total_sum);
    }

    println!(
        "{:>24}  {:>11}  {:>10}  {:>14}  {:>10}  {:>14}",
        "reference pair", "mid hits", "out hits", "sum Δ total", "mid max Δ", "first hit"
    );
    println!("{}", "-".repeat(100));
    println!(
        "{:>24}  {:>11}  {:>10}  {:>14}  {:>10}  {:>14}",
        "assembly vs qlit",
        middle_hits,
        outer_hits,
        fmt_thousands(sum_delta_total),
        max_middle_gain,
        first_hit
            .map(fmt_thousands)
            .unwrap_or_else(|| "-".to_string()),
    );
}

fn run_post_plateau_ordinary_dr_like_grid() {
    println!(
        "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "x", "ordinary len", "left out.", "left tr.", "central", "right tr.", "right out."
    );
    println!("{}", "-".repeat(114));

    for &x in candidate_grid_points() {
        let analysis = dr::analyze_ordinary_dr_like_region(x);
        println!(
            "{:>22}  {:>14}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            fmt_thousands(x),
            analysis.ordinary_len,
            analysis.left_outer_work_len,
            analysis.left_transfer_work_len,
            analysis.central_work_region_len,
            analysis.right_transfer_work_len,
            analysis.right_outer_work_len,
        );
    }
}

fn run_post_plateau_ordinary_dr_like_vs_quasi_literature() {
    println!(
        "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}",
        "x", "qlit mid", "dr core", "qlit out", "dr out", "qlit sum", "dr sum"
    );
    println!("{}", "-".repeat(110));

    for &x in candidate_grid_points() {
        let qlit = dr::analyze_ordinary_quasi_literature_region(x);
        let dr_like = dr::analyze_ordinary_dr_like_region(x);
        let qlit_outer_len = qlit.left_outer_work_len + qlit.right_outer_work_len;
        let dr_outer_len = dr_like.left_outer_work_len + dr_like.right_outer_work_len;
        let qlit_sum = qlit.left_outer_work_sum + qlit.middle_work_sum + qlit.right_outer_work_sum;
        let dr_sum = dr_like.left_outer_work_sum
            + dr_like.left_transfer_work_sum
            + dr_like.central_work_region_sum
            + dr_like.right_transfer_work_sum
            + dr_like.right_outer_work_sum;

        println!(
            "{:>22}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>14}",
            fmt_thousands(x),
            qlit.middle_work_len,
            dr_like.central_work_region_len,
            qlit_outer_len,
            dr_outer_len,
            fmt_thousands(qlit_sum),
            fmt_thousands(dr_sum),
        );
    }
}

fn run_phase_c_boundary_relative_quotient_step_bridge_compare() {
    let mut s1_hits = 0usize;
    let mut hard_hits = 0usize;
    let mut easy_hits = 0usize;
    let mut easy_delta_total = 0usize;
    let mut first_hit: Option<u128> = None;
    let mut result_delta_nonzero = false;

    for &x in candidate_grid_points() {
        let comparison =
            dr::compare_phase_c_package_vs_experimental_boundary_relative_quotient_step_bridge_candidate(
                x, 4, 4, 2, 1, 1, 1,
            );

        if comparison.current.s1_len != comparison.experimental.s1_len {
            s1_hits += 1;
        }
        if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
            hard_hits += 1;
        }
        if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
            easy_hits += 1;
            easy_delta_total += comparison
                .experimental
                .contributions
                .easy
                .abs_diff(comparison.current.contributions.easy)
                as usize;
        }
        if first_hit.is_none()
            && (comparison.current.s1_len != comparison.experimental.s1_len
                || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
        {
            first_hit = Some(x);
        }
        if comparison.current.result != comparison.experimental.result {
            result_delta_nonzero = true;
        }
    }

    println!(
        "{:>40}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "reference pair", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(120));
    println!(
        "{:>40}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "linked vs boundary+relativeq-step-bridge",
        s1_hits,
        hard_hits,
        easy_hits,
        easy_delta_total,
        first_hit
            .map(fmt_thousands)
            .unwrap_or_else(|| "-".to_string()),
        if result_delta_nonzero {
            "non-zero"
        } else {
            "0"
        }
    );
}

fn run_phase_c_boundary_relative_quotient_step_bridge_search() {
    let candidates = [1_u128, 2_u128, 3_u128];

    println!(
        "{:>11}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "bridge", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(99));

    for bridge_width in candidates {
        let mut s1_hits = 0usize;
        let mut hard_hits = 0usize;
        let mut easy_hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison =
                dr::compare_phase_c_package_vs_experimental_boundary_relative_quotient_step_bridge_candidate(
                    x, 4, 4, 2, 1, 1, bridge_width,
                );

            if comparison.current.s1_len != comparison.experimental.s1_len {
                s1_hits += 1;
            }
            if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
                hard_hits += 1;
            }
            if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
                easy_hits += 1;
                easy_delta_total += comparison
                    .experimental
                    .contributions
                    .easy
                    .abs_diff(comparison.current.contributions.easy)
                    as usize;
            }
            if first_hit.is_none()
                && (comparison.current.s1_len != comparison.experimental.s1_len
                    || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                    || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
            {
                first_hit = Some(x);
            }
            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>11}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            bridge_width,
            s1_hits,
            hard_hits,
            easy_hits,
            easy_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_phase_c_boundary_vs_relative_quotient_step_bridge_dense() {
    let mut s1_hits = 0usize;
    let mut hard_hits = 0usize;
    let mut easy_hits = 0usize;
    let mut easy_delta_total = 0usize;
    let mut first_hit: Option<u128> = None;
    let mut result_delta_nonzero = false;

    for &x in phase_c_dense_points() {
        let comparison =
            dr::compare_boundary_candidate_vs_experimental_boundary_relative_quotient_step_bridge_candidate(
                x, 4, 4, 2, 4, 4, 2, 1, 1, 1,
            );

        if comparison.current.s1_len != comparison.experimental.s1_len {
            s1_hits += 1;
        }
        if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
            hard_hits += 1;
        }
        if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
            easy_hits += 1;
            easy_delta_total += comparison
                .experimental
                .contributions
                .easy
                .abs_diff(comparison.current.contributions.easy)
                as usize;
        }
        if first_hit.is_none()
            && (comparison.current.s1_len != comparison.experimental.s1_len
                || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
        {
            first_hit = Some(x);
        }
        if comparison.current.result != comparison.experimental.result {
            result_delta_nonzero = true;
        }
    }

    println!(
        "{:>43}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "reference pair", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(123));
    println!(
        "{:>43}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "boundary 4:4:2 vs relativeq-step-bridge 1:1:1",
        s1_hits,
        hard_hits,
        easy_hits,
        easy_delta_total,
        first_hit
            .map(fmt_thousands)
            .unwrap_or_else(|| "-".to_string()),
        if result_delta_nonzero {
            "non-zero"
        } else {
            "0"
        }
    );
}

fn run_phase_c_step_band_vs_step_bridge_dense() {
    let mut s1_hits = 0usize;
    let mut hard_hits = 0usize;
    let mut easy_hits = 0usize;
    let mut easy_delta_total = 0usize;
    let mut first_hit: Option<u128> = None;
    let mut result_delta_nonzero = false;

    for &x in phase_c_dense_points() {
        let comparison = dr::compare_step_band_vs_experimental_step_bridge_candidate(
            x, 4, 4, 2, 1, 1, 4, 4, 2, 1, 1, 1,
        );

        if comparison.current.s1_len != comparison.experimental.s1_len {
            s1_hits += 1;
        }
        if comparison.current.s2_hard_len != comparison.experimental.s2_hard_len {
            hard_hits += 1;
        }
        if comparison.current.s2_easy_len != comparison.experimental.s2_easy_len {
            easy_hits += 1;
            easy_delta_total += comparison
                .experimental
                .contributions
                .easy
                .abs_diff(comparison.current.contributions.easy)
                as usize;
        }
        if first_hit.is_none()
            && (comparison.current.s1_len != comparison.experimental.s1_len
                || comparison.current.s2_hard_len != comparison.experimental.s2_hard_len
                || comparison.current.s2_easy_len != comparison.experimental.s2_easy_len)
        {
            first_hit = Some(x);
        }
        if comparison.current.result != comparison.experimental.result {
            result_delta_nonzero = true;
        }
    }

    println!(
        "{:>41}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "reference pair", "S1 hits", "hard hits", "easy hits", "easy Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(121));
    println!(
        "{:>41}  {:>10}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "relativeq-step 1:1 vs step-bridge 1:1:1",
        s1_hits,
        hard_hits,
        easy_hits,
        easy_delta_total,
        first_hit
            .map(fmt_thousands)
            .unwrap_or_else(|| "-".to_string()),
        if result_delta_nonzero {
            "non-zero"
        } else {
            "0"
        }
    );
}

fn run_candidate_grid() {
    let xs = [
        1_000_u128,
        2_000,
        5_000,
        10_000,
        20_000,
        50_000,
        100_000,
        200_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
        20_000_000,
        50_000_000,
        100_000_000,
    ];

    println!(
        "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "x", "delta S1", "delta S2_hard", "delta S2_easy", "delta easy", "result Δ"
    );
    println!("{}", "-".repeat(95));

    for &x in &xs {
        let comparison = dr::compare_current_vs_candidate_easy_relative_to_hard(x, 3, 1);
        println!(
            "{:>22}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
            fmt_thousands(x),
            fmt_delta_usize(comparison.current.s1_len, comparison.experimental.s1_len),
            fmt_delta_usize(
                comparison.current.s2_hard_len,
                comparison.experimental.s2_hard_len,
            ),
            fmt_delta_usize(
                comparison.current.s2_easy_len,
                comparison.experimental.s2_easy_len,
            ),
            fmt_delta_u128(
                comparison.current.contributions.easy,
                comparison.experimental.contributions.easy,
            ),
            fmt_delta_u128(comparison.current.result, comparison.experimental.result),
        );
    }
}

fn candidate_grid_points() -> &'static [u128] {
    &[
        1_000_u128,
        2_000,
        5_000,
        10_000,
        20_000,
        50_000,
        100_000,
        200_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
        20_000_000,
        50_000_000,
        100_000_000,
    ]
}

fn run_candidate_search() {
    println!(
        "{:>8}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "width", "hits", "easy Δ", "hard Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(78));

    for width in 1_u128..=4 {
        let mut hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut hard_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in candidate_grid_points() {
            let comparison =
                dr::compare_current_vs_experimental_easy_relative_to_hard(x, 3, 1, width);
            let easy_delta = comparison
                .experimental
                .s2_easy_len
                .saturating_sub(comparison.current.s2_easy_len);
            let hard_delta = comparison
                .current
                .s2_hard_len
                .saturating_sub(comparison.experimental.s2_hard_len);

            if easy_delta > 0 || hard_delta > 0 {
                hits += 1;
                easy_delta_total += easy_delta;
                hard_delta_total += hard_delta;
                if first_hit.is_none() {
                    first_hit = Some(x);
                }
            }

            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>8}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            width,
            hits,
            easy_delta_total,
            hard_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_candidate_search_dense() {
    let xs = [
        1_000_u128,
        1_500,
        2_000,
        3_000,
        5_000,
        7_500,
        10_000,
        15_000,
        20_000,
        30_000,
        50_000,
        75_000,
        100_000,
        150_000,
        200_000,
        300_000,
        500_000,
        750_000,
        1_000_000,
        1_500_000,
        2_000_000,
        3_000_000,
        5_000_000,
        7_500_000,
        10_000_000,
        15_000_000,
        20_000_000,
        30_000_000,
        50_000_000,
        75_000_000,
        100_000_000,
    ];

    println!(
        "{:>8}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "width", "hits", "easy Δ", "hard Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(78));

    for width in 1_u128..=4 {
        let mut hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut hard_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in &xs {
            let comparison =
                dr::compare_current_vs_experimental_easy_relative_to_hard(x, 3, 1, width);
            let easy_delta = comparison
                .experimental
                .s2_easy_len
                .saturating_sub(comparison.current.s2_easy_len);
            let hard_delta = comparison
                .current
                .s2_hard_len
                .saturating_sub(comparison.experimental.s2_hard_len);

            if easy_delta > 0 || hard_delta > 0 {
                hits += 1;
                easy_delta_total += easy_delta;
                hard_delta_total += hard_delta;
                if first_hit.is_none() {
                    first_hit = Some(x);
                }
            }

            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>8}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            width,
            hits,
            easy_delta_total,
            hard_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn run_candidate_floor_search() {
    let xs = [
        1_000_u128,
        1_500,
        2_000,
        3_000,
        5_000,
        7_500,
        10_000,
        15_000,
        20_000,
        30_000,
        50_000,
        75_000,
        100_000,
        150_000,
        200_000,
        300_000,
        500_000,
        750_000,
        1_000_000,
        1_500_000,
        2_000_000,
        3_000_000,
        5_000_000,
        7_500_000,
        10_000_000,
        15_000_000,
        20_000_000,
        30_000_000,
        50_000_000,
        75_000_000,
        100_000_000,
    ];

    println!(
        "{:>8}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
        "floor", "hits", "easy Δ", "hard Δ", "first hit", "result Δ"
    );
    println!("{}", "-".repeat(78));

    for floor in 1_u128..=3 {
        let mut hits = 0usize;
        let mut easy_delta_total = 0usize;
        let mut hard_delta_total = 0usize;
        let mut first_hit: Option<u128> = None;
        let mut result_delta_nonzero = false;

        for &x in &xs {
            let comparison =
                dr::compare_current_vs_candidate_easy_relative_to_hard_with_floor(x, 3, 1, floor);

            let easy_delta = comparison
                .experimental
                .s2_easy_len
                .saturating_sub(comparison.current.s2_easy_len);
            let hard_delta = comparison
                .current
                .s2_hard_len
                .saturating_sub(comparison.experimental.s2_hard_len);

            if easy_delta > 0 || hard_delta > 0 {
                hits += 1;
                easy_delta_total += easy_delta;
                hard_delta_total += hard_delta;
                if first_hit.is_none() {
                    first_hit = Some(x);
                }
            }

            if comparison.current.result != comparison.experimental.result {
                result_delta_nonzero = true;
            }
        }

        println!(
            "{:>8}  {:>10}  {:>10}  {:>10}  {:>14}  {:>10}",
            floor,
            hits,
            easy_delta_total,
            hard_delta_total,
            first_hit
                .map(fmt_thousands)
                .unwrap_or_else(|| "-".to_string()),
            if result_delta_nonzero {
                "non-zero"
            } else {
                "0"
            }
        );
    }
}

fn print_usage(program: &str) {
    eprintln!("Usage:");
    eprintln!(
        "  {} <x> [-t <threads> | -nt <threads>] [-a <alpha> | --alpha <alpha>] [--hard-leaf-term-max <n>] [--easy-leaf-term-max <n>]",
        program
    );
    eprintln!("  {} -nt <x,threads> [-nt <x,threads> ...]", program);
    eprintln!(
        "  {} --profile <x> [-t <threads> | -nt <threads>] [--hard-leaf-term-max <n>] [--easy-leaf-term-max <n>] [--candidate-easy-relative-to-hard | --experimental-easy-relative-to-hard <width> | --experimental-hard-relative-to-easy <width>]",
        program
    );
    eprintln!("  {} --dr-profile <x>", program);
    eprintln!(
        "  {} --dr-meissel [x] [-t <threads> | -nt <threads>]",
        program
    );
    eprintln!(
        "  {} --dr-vs-baseline-grid [-t <threads> | -nt <threads>]",
        program
    );
    eprintln!("  {} --phi-backend-grid", program);
    eprintln!("  {} --phi-backend-profile <x>", program);
    eprintln!("  {} --dr-phi-backend-profile <x>", program);
    eprintln!(
        "  {} --profile <x> [-t <threads> | -nt <threads>] [--hard-leaf-term-max <n>] [--easy-leaf-term-max <n>] [--candidate-easy-relative-to-hard | --candidate-easy-term-band | --phase-c-easy-term-band | --phase-c-package | --phase-c-linked-package | --experimental-easy-relative-to-hard <width> | --experimental-hard-relative-to-easy <width>]",
        program
    );
    eprintln!(
        "  {} --sweep [x_max] [-t <threads> | -nt <threads>] [--hard-leaf-term-max <n>] [--easy-leaf-term-max <n>] [--candidate-easy-relative-to-hard | --candidate-easy-term-band | --phase-c-easy-term-band | --phase-c-package | --phase-c-linked-package | --experimental-easy-relative-to-hard <width> | --experimental-hard-relative-to-easy <width>]",
        program
    );
    eprintln!("  {} --candidate-grid", program);
    eprintln!("  {} --candidate-search", program);
    eprintln!("  {} --candidate-search-dense", program);
    eprintln!("  {} --candidate-floor-search", program);
    eprintln!("  {} --candidate-family-compare", program);
    eprintln!("  {} --candidate-band-search", program);
    eprintln!("  {} --phase-c-easy-band-grid", program);
    eprintln!("  {} --phase-c-easy-compare", program);
    eprintln!("  {} --phase-c-easy-search", program);
    eprintln!("  {} --phase-c-easy-compare-bands", program);
    eprintln!("  {} --post-plateau-ordinary-shoulder-grid", program);
    eprintln!("  {} --post-plateau-ordinary-shoulder-search", program);
    eprintln!("  {} --post-plateau-ordinary-envelope-grid", program);
    eprintln!("  {} --post-plateau-ordinary-envelope-search", program);
    eprintln!("  {} --post-plateau-ordinary-envelope-vs-shoulder", program);
    eprintln!("  {} --post-plateau-ordinary-hierarchy-grid", program);
    eprintln!(
        "  {} --post-plateau-ordinary-hierarchy-vs-envelope",
        program
    );
    eprintln!("  {} --post-plateau-ordinary-assembly-grid", program);
    eprintln!(
        "  {} --post-plateau-ordinary-assembly-vs-hierarchy",
        program
    );
    eprintln!(
        "  {} --post-plateau-ordinary-quasi-literature-grid",
        program
    );
    eprintln!(
        "  {} --post-plateau-ordinary-quasi-literature-vs-assembly",
        program
    );
    eprintln!(
        "  {} --post-plateau-ordinary-quasi-literature-vs-assembly-dense",
        program
    );
    eprintln!("  {} --post-plateau-ordinary-dr-like-grid", program);
    eprintln!(
        "  {} --post-plateau-ordinary-dr-like-vs-quasi-literature",
        program
    );
    eprintln!("  {} --post-plateau-triptych-compare", program);
    eprintln!("  {} --phase-c-hard-grid", program);
    eprintln!("  {} --phase-c-hard-search", program);
    eprintln!("  {} --phase-c-hard-compare-bands", program);
    eprintln!("  {} --phase-c-package-grid", program);
    eprintln!("  {} --phase-c-package-search", program);
    eprintln!("  {} --phase-c-linked-package-compare", program);
    eprintln!("  {} --phase-c-linked-grid", program);
    eprintln!("  {} --phase-c-linked-search", program);
    eprintln!("  {} --phase-c-linked-candidate-compare", program);
    eprintln!("  {} --phase-c-reference-compare-dense", program);
    eprintln!("  {} --phase-c-boundary-package-compare", program);
    eprintln!("  {} --phase-c-boundary-search", program);
    eprintln!("  {} --phase-c-boundary-candidate-compare", program);
    eprintln!("  {} --phase-c-boundary-local-search", program);
    eprintln!("  {} --phase-c-buffered-boundary-compare", program);
    eprintln!("  {} --phase-c-buffered-boundary-search", program);
    eprintln!("  {} --phase-c-quotient-window-compare", program);
    eprintln!("  {} --phase-c-quotient-window-search", program);
    eprintln!("  {} --phase-c-quotient-window-shifted-search", program);
    eprintln!("  {} --phase-c-boundary-quotient-guard-compare", program);
    eprintln!("  {} --phase-c-boundary-quotient-guard-search", program);
    eprintln!(
        "  {} --phase-c-boundary-relative-quotient-band-compare",
        program
    );
    eprintln!(
        "  {} --phase-c-boundary-relative-quotient-band-search",
        program
    );
    eprintln!(
        "  {} --phase-c-boundary-relative-quotient-step-band-compare",
        program
    );
    eprintln!(
        "  {} --phase-c-boundary-relative-quotient-step-band-search",
        program
    );
    eprintln!("  {} --phase-c-step-band-local-search", program);
    eprintln!(
        "  {} --phase-c-boundary-vs-relative-quotient-step-dense",
        program
    );
    eprintln!("  {} --phase-c-easy-specialized-grid", program);
    eprintln!("  {} --phase-c-easy-specialized-compare", program);
    eprintln!("  {} --phase-c-hard-specialized-grid", program);
    eprintln!("  {} --phase-c-hard-specialized-compare", program);
    eprintln!(
        "  {} --phase-c-boundary-relative-quotient-step-bridge-compare",
        program
    );
    eprintln!(
        "  {} --phase-c-boundary-relative-quotient-step-bridge-search",
        program
    );
    eprintln!(
        "  {} --phase-c-boundary-vs-relative-quotient-step-bridge-dense",
        program
    );
    eprintln!("  {} --phase-c-step-band-vs-step-bridge-dense", program);
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {} 1e12", program);
    eprintln!("  {} 1e12 -t 4", program);
    eprintln!("  {} -nt 1e14,8 -nt 1e11,4 -nt 1e13,5 -nt 1e12,4", program);
    eprintln!(
        "  {} --profile 1e13 -t 4 --hard-leaf-term-max 3 --easy-leaf-term-max 1",
        program
    );
    eprintln!("  {} --dr-profile 1e11", program);
    eprintln!("  {} --dr-meissel 1e11 -nt 4", program);
    eprintln!("  {} --dr-vs-baseline-grid -t 4", program);
    eprintln!("  {} --phi-backend-grid", program);
    eprintln!("  {} --phi-backend-profile 1e11", program);
    eprintln!("  {} --dr-phi-backend-profile 1e11", program);
    eprintln!(
        "  {} --profile 2e7 --hard-leaf-term-max 3 --candidate-easy-relative-to-hard",
        program
    );
    eprintln!(
        "  {} --profile 2e7 --hard-leaf-term-max 3 --candidate-easy-term-band",
        program
    );
    eprintln!("  {} --profile 2e7 --phase-c-easy-term-band", program);
    eprintln!("  {} --profile 2e7 --phase-c-package", program);
    eprintln!("  {} --profile 2e7 --phase-c-linked-package", program);
    eprintln!(
        "  {} --profile 2e7 --hard-leaf-term-max 3 --experimental-easy-relative-to-hard 2",
        program
    );
    eprintln!(
        "  {} --profile 2e7 --easy-leaf-term-max 2 --experimental-hard-relative-to-easy 1",
        program
    );
    eprintln!(
        "  {} --sweep 1e8 --hard-leaf-term-max 3 --candidate-easy-relative-to-hard",
        program
    );
    eprintln!(
        "  {} --sweep 1e8 --hard-leaf-term-max 3 --experimental-easy-relative-to-hard 2",
        program
    );
    eprintln!("  {} --candidate-grid", program);
    eprintln!("  {} --candidate-search", program);
    eprintln!("  {} --candidate-search-dense", program);
    eprintln!("  {} --candidate-floor-search", program);
    eprintln!("  {} --candidate-family-compare", program);
    eprintln!("  {} --candidate-band-search", program);
    eprintln!("  {} --phase-c-easy-band-grid", program);
    eprintln!("  {} --phase-c-easy-compare", program);
    eprintln!("  {} --phase-c-easy-search", program);
    eprintln!("  {} --phase-c-easy-compare-bands", program);
    eprintln!("  {} --phase-c-hard-grid", program);
    eprintln!("  {} --phase-c-hard-search", program);
    eprintln!("  {} --phase-c-hard-compare-bands", program);
    eprintln!("  {} --phase-c-package-grid", program);
    eprintln!("  {} --phase-c-package-search", program);
    eprintln!("  {} --phase-c-linked-package-compare", program);
    eprintln!("  {} --phase-c-linked-grid", program);
    eprintln!("  {} --phase-c-linked-search", program);
    eprintln!("  {} --phase-c-linked-candidate-compare", program);
    eprintln!("  {} --phase-c-reference-compare-dense", program);
    eprintln!("  {} --phase-c-boundary-package-compare", program);
    eprintln!("  {} --phase-c-boundary-search", program);
    eprintln!("  {} --phase-c-boundary-candidate-compare", program);
    eprintln!("  {} --phase-c-boundary-local-search", program);
    eprintln!("  {} --phase-c-buffered-boundary-compare", program);
    eprintln!("  {} --phase-c-buffered-boundary-search", program);
    eprintln!("  {} --phase-c-quotient-window-compare", program);
    eprintln!("  {} --phase-c-quotient-window-search", program);
    eprintln!("  {} --phase-c-quotient-window-shifted-search", program);
    eprintln!("  {} --phase-c-boundary-quotient-guard-compare", program);
    eprintln!("  {} --phase-c-boundary-quotient-guard-search", program);
    eprintln!(
        "  {} --phase-c-boundary-relative-quotient-band-compare",
        program
    );
    eprintln!(
        "  {} --phase-c-boundary-relative-quotient-band-search",
        program
    );
    eprintln!(
        "  {} --phase-c-boundary-relative-quotient-step-band-compare",
        program
    );
    eprintln!(
        "  {} --phase-c-boundary-relative-quotient-step-band-search",
        program
    );
    eprintln!("  {} --phase-c-step-band-local-search", program);
    eprintln!(
        "  {} --phase-c-boundary-vs-relative-quotient-step-dense",
        program
    );
    eprintln!("  {} --phase-c-easy-specialized-grid", program);
    eprintln!("  {} --phase-c-easy-specialized-compare", program);
    eprintln!("  {} --phase-c-hard-specialized-grid", program);
    eprintln!("  {} --phase-c-hard-specialized-compare", program);
    eprintln!(
        "  {} --phase-c-boundary-relative-quotient-step-bridge-compare",
        program
    );
    eprintln!(
        "  {} --phase-c-boundary-relative-quotient-step-bridge-search",
        program
    );
    eprintln!(
        "  {} --phase-c-boundary-vs-relative-quotient-step-bridge-dense",
        program
    );
    eprintln!("  {} --phase-c-step-band-vs-step-bridge-dense", program);
    eprintln!("  {} --post-plateau-ordinary-hierarchy-grid", program);
    eprintln!(
        "  {} --post-plateau-ordinary-hierarchy-vs-envelope",
        program
    );
    eprintln!("  {} --post-plateau-ordinary-assembly-grid", program);
    eprintln!(
        "  {} --post-plateau-ordinary-assembly-vs-hierarchy",
        program
    );
    eprintln!(
        "  {} --post-plateau-ordinary-quasi-literature-grid",
        program
    );
    eprintln!(
        "  {} --post-plateau-ordinary-quasi-literature-vs-assembly",
        program
    );
    eprintln!(
        "  {} --post-plateau-ordinary-quasi-literature-vs-assembly-dense",
        program
    );
    eprintln!("  {} --post-plateau-ordinary-dr-like-grid", program);
    eprintln!(
        "  {} --post-plateau-ordinary-dr-like-vs-quasi-literature",
        program
    );
    eprintln!("  {} --sweep 1e14 -t 4", program);
}

fn main() {
    // Rayon worker threads default to 1 MB stack on Windows.
    // s2_popcount_par uses ~96 KB per closure (SegSieve 64 KB + prefix 32 KB)
    // plus ~15 levels of Rayon internal divide-and-conquer at x=1e15 (19 074 blocks).
    // 15 × ~60 KB + 96 KB ≈ 1 MB → overflows at 1e15. Set to 32 MB to be safe.
    rayon::ThreadPoolBuilder::new()
        .stack_size(32 * 1024 * 1024)
        .build_global()
        .expect("failed to build Rayon thread pool");

    let (l3_mb, _cores, _threads) = rivat3::parameters::detect_hw();

    let args: Vec<String> = env::args().collect();

    let cli = match parse_cli(&args) {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("Error: {}", e);
            print_usage(&args[0]);
            std::process::exit(1);
        }
    };

    if let Some(alpha) = cli.alpha_override {
        match validate_alpha_override(alpha, &cli.mode) {
            Ok(()) => {
                let _ = rivat3::parameters::set_alpha_override(alpha);
            }
            Err(e) => eprintln!(
                "Warning: {} — alpha sera déterminé automatiquement",
                e
            ),
        }
    }

    match cli.mode {
        Mode::Sweep { x_max } => {
            if matches!(cli.experimental_mode, ExperimentalMode::None) {
                run_sweep(x_max, cli.threads);
            } else {
                run_experimental_sweep(
                    x_max,
                    cli.hard_leaf_term_max,
                    cli.easy_leaf_term_max,
                    cli.experimental_mode,
                );
            }
        }
        Mode::CandidateGrid => run_candidate_grid(),
        Mode::CandidateSearch => run_candidate_search(),
        Mode::CandidateSearchDense => run_candidate_search_dense(),
        Mode::CandidateFloorSearch => run_candidate_floor_search(),
        Mode::CandidateFamilyCompare => run_candidate_family_compare(),
        Mode::CandidateBandSearch => run_candidate_band_search(),
        Mode::PhaseCEasyBandGrid => run_phase_c_easy_band_grid(),
        Mode::PhaseCEasyCompare => run_phase_c_easy_compare(),
        Mode::PhaseCEasySearch => run_phase_c_easy_search(),
        Mode::PhaseCEasyCompareBands => run_phase_c_easy_compare_bands(),
        Mode::PhaseCHardGrid => run_phase_c_hard_grid(),
        Mode::PhaseCHardSearch => run_phase_c_hard_search(),
        Mode::PhaseCHardCompareBands => run_phase_c_hard_compare_bands(),
        Mode::PhaseCPackageGrid => run_phase_c_package_grid(),
        Mode::PhaseCPackageSearch => run_phase_c_package_search(),
        Mode::PhaseCLinkedPackageCompare => run_phase_c_linked_package_compare(),
        Mode::PhaseCLinkedGrid => run_phase_c_linked_grid(),
        Mode::PhaseCLinkedSearch => run_phase_c_linked_search(),
        Mode::PhaseCLinkedCandidateCompare => run_phase_c_linked_candidate_compare(),
        Mode::PhaseCReferenceCompareDense => run_phase_c_reference_compare_dense(),
        Mode::PhaseCBoundaryPackageCompare => run_phase_c_boundary_package_compare(),
        Mode::PhaseCBoundarySearch => run_phase_c_boundary_search(),
        Mode::PhaseCBoundaryCandidateCompare => run_phase_c_boundary_candidate_compare(),
        Mode::PhaseCBoundaryLocalSearch => run_phase_c_boundary_local_search(),
        Mode::PhaseCBufferedBoundaryCompare => run_phase_c_buffered_boundary_compare(),
        Mode::PhaseCBufferedBoundarySearch => run_phase_c_buffered_boundary_search(),
        Mode::PhaseCQuotientWindowCompare => run_phase_c_quotient_window_compare(),
        Mode::PhaseCQuotientWindowSearch => run_phase_c_quotient_window_search(),
        Mode::PhaseCQuotientWindowShiftedSearch => run_phase_c_quotient_window_shifted_search(),
        Mode::PhaseCBoundaryQuotientGuardCompare => run_phase_c_boundary_quotient_guard_compare(),
        Mode::PhaseCBoundaryQuotientGuardSearch => run_phase_c_boundary_quotient_guard_search(),
        Mode::PhaseCBoundaryRelativeQuotientBandCompare => {
            run_phase_c_boundary_relative_quotient_band_compare()
        }
        Mode::PhaseCBoundaryRelativeQuotientBandSearch => {
            run_phase_c_boundary_relative_quotient_band_search()
        }
        Mode::PhaseCBoundaryRelativeQuotientStepBandCompare => {
            run_phase_c_boundary_relative_quotient_step_band_compare()
        }
        Mode::PhaseCBoundaryRelativeQuotientStepBandSearch => {
            run_phase_c_boundary_relative_quotient_step_band_search()
        }
        Mode::PhaseCStepBandLocalSearch => run_phase_c_step_band_local_search(),
        Mode::PhaseCBoundaryVsRelativeQuotientStepDense => {
            run_phase_c_boundary_vs_relative_quotient_step_dense()
        }
        Mode::PhaseCEasySpecializedGrid => run_phase_c_easy_specialized_grid(),
        Mode::PhaseCEasySpecializedCompare => run_phase_c_easy_specialized_compare(),
        Mode::PhaseCOrdinarySpecializedGrid => run_phase_c_ordinary_specialized_grid(),
        Mode::PhaseCOrdinarySpecializedCompare => run_phase_c_ordinary_specialized_compare(),
        Mode::PhaseCOrdinaryRelativeQuotientGrid => run_phase_c_ordinary_relative_quotient_grid(),
        Mode::PhaseCOrdinaryRelativeQuotientCompare => {
            run_phase_c_ordinary_relative_quotient_compare()
        }
        Mode::PhaseCOrdinaryRelativeQuotientVsSpecialized => {
            run_phase_c_ordinary_relative_quotient_vs_specialized()
        }
        Mode::PhaseCOrdinaryRelativeQuotientSearch => {
            run_phase_c_ordinary_relative_quotient_search()
        }
        Mode::PostPlateauOrdinaryShoulderGrid => run_post_plateau_ordinary_shoulder_grid(),
        Mode::PostPlateauOrdinaryShoulderSearch => run_post_plateau_ordinary_shoulder_search(),
        Mode::PostPlateauOrdinaryEnvelopeGrid => run_post_plateau_ordinary_envelope_grid(),
        Mode::PostPlateauOrdinaryEnvelopeSearch => run_post_plateau_ordinary_envelope_search(),
        Mode::PostPlateauOrdinaryEnvelopeVsShoulder => {
            run_post_plateau_ordinary_envelope_vs_shoulder()
        }
        Mode::PostPlateauOrdinaryHierarchyGrid => run_post_plateau_ordinary_hierarchy_grid(),
        Mode::PostPlateauOrdinaryHierarchyVsEnvelope => {
            run_post_plateau_ordinary_hierarchy_vs_envelope()
        }
        Mode::PostPlateauOrdinaryAssemblyGrid => run_post_plateau_ordinary_assembly_grid(),
        Mode::PostPlateauOrdinaryAssemblyVsHierarchy => {
            run_post_plateau_ordinary_assembly_vs_hierarchy()
        }
        Mode::PostPlateauOrdinaryQuasiLiteratureGrid => {
            run_post_plateau_ordinary_quasi_literature_grid()
        }
        Mode::PostPlateauOrdinaryQuasiLiteratureVsAssembly => {
            run_post_plateau_ordinary_quasi_literature_vs_assembly()
        }
        Mode::PostPlateauOrdinaryQuasiLiteratureVsAssemblyDense => {
            run_post_plateau_ordinary_quasi_literature_vs_assembly_dense()
        }
        Mode::PostPlateauOrdinaryDrLikeGrid => run_post_plateau_ordinary_dr_like_grid(),
        Mode::PostPlateauOrdinaryDrLikeVsQuasiLiterature => {
            run_post_plateau_ordinary_dr_like_vs_quasi_literature()
        }
        Mode::PostPlateauTriptychCompare => run_post_plateau_triptych_compare(),
        Mode::PhaseCHardSpecializedGrid => run_phase_c_hard_specialized_grid(),
        Mode::PhaseCHardSpecializedCompare => run_phase_c_hard_specialized_compare(),
        Mode::PhaseCBoundaryRelativeQuotientStepBridgeCompare => {
            run_phase_c_boundary_relative_quotient_step_bridge_compare()
        }
        Mode::PhaseCBoundaryRelativeQuotientStepBridgeSearch => {
            run_phase_c_boundary_relative_quotient_step_bridge_search()
        }
        Mode::PhaseCBoundaryVsRelativeQuotientStepBridgeDense => {
            run_phase_c_boundary_vs_relative_quotient_step_bridge_dense()
        }
        Mode::PhaseCStepBandVsStepBridgeDense => run_phase_c_step_band_vs_step_bridge_dense(),
        Mode::Profile { x } => run_profile(
            x,
            cli.threads,
            cli.hard_leaf_term_max,
            cli.easy_leaf_term_max,
            cli.experimental_mode,
        ),
        Mode::NtBatch { jobs } => run_nt_batch(&jobs),
        Mode::DrProfile { x } => run_dr_profile(x),
        // These modes use deep phi recursion (phi_loop_rec up to a~5000 levels at x=1e14+).
        // The default main-thread stack (1 MB on Windows) overflows; use 64 MB.
        Mode::DrMeisselProfile { x } => {
            let t = cli.threads;
            std::thread::Builder::new().stack_size(64 << 20)
                .spawn(move || run_dr_meissel_profile(x, t)).unwrap().join().unwrap();
        }
        Mode::DrMeissel2Profile { x } => {
            let t = cli.threads;
            std::thread::Builder::new().stack_size(64 << 20)
                .spawn(move || run_dr_meissel2_profile(x, t)).unwrap().join().unwrap();
        }
        Mode::DrMeissel3Profile { x } => {
            let t = cli.threads;
            std::thread::Builder::new().stack_size(64 << 20)
                .spawn(move || run_dr_meissel3_profile(x, t)).unwrap().join().unwrap();
        }
        Mode::DrMeissel4Profile { x } => {
            let t = cli.threads;
            std::thread::Builder::new().stack_size(64 << 20)
                .spawn(move || run_dr_meissel4_profile(x, t)).unwrap().join().unwrap();
        }
        Mode::DrV3Profile { x } => run_dr_v3_profile(x),
        Mode::DrV4Profile { x } => run_dr_v4_profile(x),
        Mode::LucyProfile { x } => { lucy_phi_early_stop_profiled(x); }
        Mode::DrVsBaselineGrid => run_dr_vs_baseline_grid(cli.threads),
        Mode::PhiBackendGrid => run_phi_backend_grid(),
        Mode::PhiBackendProfile { x } => run_phi_backend_profile(x),
        Mode::DrPhiBackendProfile { x } => run_dr_phi_backend_profile(x),
        Mode::Normal { label, x } => {
            use std::io::Write;
            let alpha = rivat3::parameters::choose_alpha(x);
            println!(
                "n = {}  |  Calcul en cours...  [primerivat {} | L3={}Mo α={}]",
                label, env!("GIT_HASH"), l3_mb, alpha
            );
            let _ = std::io::stdout().flush();
            let t0 = Instant::now();
            let result = std::thread::Builder::new()
                .stack_size(64 << 20)
                .spawn(move || dr::prime_pi_dr_meissel_v4(x))
                .unwrap()
                .join()
                .unwrap();
            let elapsed = t0.elapsed();
            println!(
                "n = {}  |  π(n) = {}  |  {}",
                label,
                fmt_thousands(result),
                fmt_elapsed_seconds(elapsed),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ExperimentalMode, Mode, parse_cli};

    #[test]
    fn parse_cli_accepts_hard_leaf_term_max() {
        let args = vec![
            "rivat3".to_string(),
            "--profile".to_string(),
            "1e6".to_string(),
            "--hard-leaf-term-max".to_string(),
            "3".to_string(),
            "--easy-leaf-term-max".to_string(),
            "2".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::Profile { x: 1_000_000 }));
        assert_eq!(cli.hard_leaf_term_max, 3);
        assert_eq!(cli.easy_leaf_term_max, 2);
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_alpha_short_and_long() {
        let short = parse_cli(&[
            "rivat3".to_string(),
            "1e6".to_string(),
            "-a".to_string(),
            "2".to_string(),
        ])
        .expect("CLI should parse");
        assert_eq!(short.alpha_override, Some(2.0));

        let long = parse_cli(&[
            "rivat3".to_string(),
            "1e6".to_string(),
            "--alpha".to_string(),
            "1.5".to_string(),
        ])
        .expect("CLI should parse");
        assert_eq!(long.alpha_override, Some(1.5));
    }

    #[test]
    fn parse_cli_falls_back_to_auto_on_invalid_alpha() {
        for value in ["0", "0.5", "2.1", "3"] {
            let cli = parse_cli(&[
                "rivat3".to_string(),
                "1e6".to_string(),
                "--alpha".to_string(),
                value.to_string(),
            ])
            .unwrap_or_else(|e| panic!("expected ok for alpha={value}, got err: {e}"));
            assert!(
                cli.alpha_override.is_none(),
                "alpha={value} should fall back to auto",
            );
        }
    }

    #[test]
    fn validate_alpha_override_enforces_x_threshold() {
        use super::{Mode, validate_alpha_override};
        assert!(validate_alpha_override(1.5, &Mode::Normal { label: "1e15".to_string(), x: 1_000_000_000_000_000 }).is_ok());
        assert!(validate_alpha_override(1.5, &Mode::Normal { label: "2e15".to_string(), x: 2_000_000_000_000_000 }).is_err());
        assert!(validate_alpha_override(1.0, &Mode::Normal { label: "1e17".to_string(), x: 100_000_000_000_000_000 }).is_ok());
        assert!(validate_alpha_override(2.0, &Mode::Normal { label: "1e17".to_string(), x: 100_000_000_000_000_000 }).is_ok());
    }

    #[test]
    fn parse_cli_accepts_dr_profile_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--dr-profile".to_string(),
            "1e11".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::DrProfile { x: 100_000_000_000 }));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_dr_vs_baseline_grid_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--dr-vs-baseline-grid".to_string(),
            "-t".to_string(),
            "4".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::DrVsBaselineGrid));
        assert_eq!(cli.threads, 4);
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_nt_alias() {
        let args = vec![
            "rivat3".to_string(),
            "--dr-meissel".to_string(),
            "1e11".to_string(),
            "-nt".to_string(),
            "4".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::DrMeisselProfile { x: 100_000_000_000 }
        ));
        assert_eq!(cli.threads, 4);
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_nt_batch_mode() {
        let args = vec![
            "rivat3".to_string(),
            "-nt".to_string(),
            "1e15,3".to_string(),
            "-nt".to_string(),
            "1e11,1".to_string(),
            "-nt".to_string(),
            "1e14,3".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        match cli.mode {
            Mode::NtBatch { jobs } => {
                assert_eq!(jobs.len(), 3);
                assert_eq!(jobs[0].label, "1e15");
                assert_eq!(jobs[0].x, 1_000_000_000_000_000);
                assert_eq!(jobs[0].threads, 3);
                assert_eq!(jobs[1].label, "1e11");
                assert_eq!(jobs[1].threads, 1);
                assert_eq!(jobs[2].label, "1e14");
                assert_eq!(jobs[2].threads, 3);
            }
            _ => panic!("expected NtBatch mode"),
        }
    }

    #[test]
    fn parse_cli_accepts_phi_backend_grid_mode() {
        let args = vec!["rivat3".to_string(), "--phi-backend-grid".to_string()];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhiBackendGrid));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phi_backend_profile_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phi-backend-profile".to_string(),
            "1e11".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PhiBackendProfile { x: 100_000_000_000 }
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_dr_phi_backend_profile_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--dr-phi-backend-profile".to_string(),
            "1e11".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::DrPhiBackendProfile { x: 100_000_000_000 }
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_rejects_small_hard_leaf_term_max() {
        let args = vec![
            "rivat3".to_string(),
            "100".to_string(),
            "--hard-leaf-term-max".to_string(),
            "1".to_string(),
        ];

        assert!(parse_cli(&args).is_err());
    }

    #[test]
    fn parse_cli_rejects_easy_leaf_term_max_above_hard_max() {
        let args = vec![
            "rivat3".to_string(),
            "100".to_string(),
            "--hard-leaf-term-max".to_string(),
            "2".to_string(),
            "--easy-leaf-term-max".to_string(),
            "3".to_string(),
        ];

        assert!(parse_cli(&args).is_err());
    }

    #[test]
    fn parse_cli_accepts_experimental_easy_relative_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--profile".to_string(),
            "2e7".to_string(),
            "--hard-leaf-term-max".to_string(),
            "3".to_string(),
            "--experimental-easy-relative-to-hard".to_string(),
            "2".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::Profile { x: 20_000_000 }));
        assert_eq!(
            cli.experimental_mode,
            ExperimentalMode::EasyRelativeToHard { width: 2 }
        );
    }

    #[test]
    fn parse_cli_accepts_candidate_easy_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--profile".to_string(),
            "2e7".to_string(),
            "--hard-leaf-term-max".to_string(),
            "3".to_string(),
            "--candidate-easy-relative-to-hard".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::Profile { x: 20_000_000 }));
        assert_eq!(
            cli.experimental_mode,
            ExperimentalMode::CandidateEasyRelativeToHard
        );
    }

    #[test]
    fn parse_cli_accepts_candidate_easy_term_band_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--profile".to_string(),
            "2e7".to_string(),
            "--hard-leaf-term-max".to_string(),
            "3".to_string(),
            "--candidate-easy-term-band".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::Profile { x: 20_000_000 }));
        assert_eq!(
            cli.experimental_mode,
            ExperimentalMode::CandidateEasyTermBand
        );
    }

    #[test]
    fn parse_cli_accepts_candidate_grid_mode() {
        let args = vec!["rivat3".to_string(), "--candidate-grid".to_string()];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::CandidateGrid));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_candidate_search_mode() {
        let args = vec!["rivat3".to_string(), "--candidate-search".to_string()];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::CandidateSearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_candidate_search_dense_mode() {
        let args = vec!["rivat3".to_string(), "--candidate-search-dense".to_string()];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::CandidateSearchDense));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_candidate_floor_search_mode() {
        let args = vec!["rivat3".to_string(), "--candidate-floor-search".to_string()];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::CandidateFloorSearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_candidate_family_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--candidate-family-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::CandidateFamilyCompare));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_candidate_band_search_mode() {
        let args = vec!["rivat3".to_string(), "--candidate-band-search".to_string()];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::CandidateBandSearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_easy_term_band_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--profile".to_string(),
            "2e7".to_string(),
            "--phase-c-easy-term-band".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::Profile { x: 20_000_000 }));
        assert_eq!(cli.experimental_mode, ExperimentalMode::PhaseCEasyTermBand);
    }

    #[test]
    fn parse_cli_accepts_phase_c_package_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--profile".to_string(),
            "2e7".to_string(),
            "--phase-c-package".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::Profile { x: 20_000_000 }));
        assert_eq!(cli.experimental_mode, ExperimentalMode::PhaseCPackage);
    }

    #[test]
    fn parse_cli_accepts_phase_c_linked_package_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--profile".to_string(),
            "2e7".to_string(),
            "--phase-c-linked-package".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::Profile { x: 20_000_000 }));
        assert_eq!(cli.experimental_mode, ExperimentalMode::PhaseCLinkedPackage);
    }

    #[test]
    fn parse_cli_accepts_phase_c_easy_band_grid_mode() {
        let args = vec!["rivat3".to_string(), "--phase-c-easy-band-grid".to_string()];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCEasyBandGrid));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_easy_compare_mode() {
        let args = vec!["rivat3".to_string(), "--phase-c-easy-compare".to_string()];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCEasyCompare));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_easy_search_mode() {
        let args = vec!["rivat3".to_string(), "--phase-c-easy-search".to_string()];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCEasySearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_easy_compare_bands_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-easy-compare-bands".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCEasyCompareBands));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_hard_grid_mode() {
        let args = vec!["rivat3".to_string(), "--phase-c-hard-grid".to_string()];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCHardGrid));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_hard_search_mode() {
        let args = vec!["rivat3".to_string(), "--phase-c-hard-search".to_string()];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCHardSearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_hard_compare_bands_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-hard-compare-bands".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCHardCompareBands));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_package_grid_mode() {
        let args = vec!["rivat3".to_string(), "--phase-c-package-grid".to_string()];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCPackageGrid));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_package_search_mode() {
        let args = vec!["rivat3".to_string(), "--phase-c-package-search".to_string()];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCPackageSearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_linked_package_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-linked-package-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCLinkedPackageCompare));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_linked_grid_mode() {
        let args = vec!["rivat3".to_string(), "--phase-c-linked-grid".to_string()];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCLinkedGrid));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_linked_search_mode() {
        let args = vec!["rivat3".to_string(), "--phase-c-linked-search".to_string()];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCLinkedSearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_linked_candidate_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-linked-candidate-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCLinkedCandidateCompare));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_reference_compare_dense_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-reference-compare-dense".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCReferenceCompareDense));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_boundary_package_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-boundary-package-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCBoundaryPackageCompare));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_boundary_search_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-boundary-search".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCBoundarySearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_boundary_candidate_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-boundary-candidate-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCBoundaryCandidateCompare));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_boundary_local_search_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-boundary-local-search".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCBoundaryLocalSearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_buffered_boundary_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-buffered-boundary-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCBufferedBoundaryCompare));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_buffered_boundary_search_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-buffered-boundary-search".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCBufferedBoundarySearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_quotient_window_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-quotient-window-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCQuotientWindowCompare));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_quotient_window_search_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-quotient-window-search".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCQuotientWindowSearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_quotient_window_shifted_search_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-quotient-window-shifted-search".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCQuotientWindowShiftedSearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_boundary_quotient_guard_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-boundary-quotient-guard-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCBoundaryQuotientGuardCompare));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_boundary_quotient_guard_search_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-boundary-quotient-guard-search".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCBoundaryQuotientGuardSearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_boundary_relative_quotient_band_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-boundary-relative-quotient-band-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PhaseCBoundaryRelativeQuotientBandCompare
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_boundary_relative_quotient_band_search_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-boundary-relative-quotient-band-search".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PhaseCBoundaryRelativeQuotientBandSearch
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_boundary_relative_quotient_step_band_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-boundary-relative-quotient-step-band-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PhaseCBoundaryRelativeQuotientStepBandCompare
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_boundary_relative_quotient_step_band_search_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-boundary-relative-quotient-step-band-search".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PhaseCBoundaryRelativeQuotientStepBandSearch
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_step_band_local_search_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-step-band-local-search".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCStepBandLocalSearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_boundary_vs_relative_quotient_step_dense_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-boundary-vs-relative-quotient-step-dense".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PhaseCBoundaryVsRelativeQuotientStepDense
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_easy_specialized_grid_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-easy-specialized-grid".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCEasySpecializedGrid));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_easy_specialized_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-easy-specialized-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCEasySpecializedCompare));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_hard_specialized_grid_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-hard-specialized-grid".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCHardSpecializedGrid));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_hard_specialized_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-hard-specialized-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCHardSpecializedCompare));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_ordinary_specialized_grid_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-ordinary-specialized-grid".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCOrdinarySpecializedGrid));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_ordinary_specialized_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-ordinary-specialized-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCOrdinarySpecializedCompare));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_ordinary_relative_quotient_grid_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-ordinary-relative-quotient-grid".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCOrdinaryRelativeQuotientGrid));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_ordinary_relative_quotient_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-ordinary-relative-quotient-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PhaseCOrdinaryRelativeQuotientCompare
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_ordinary_relative_quotient_vs_specialized_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-ordinary-relative-quotient-vs-specialized".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PhaseCOrdinaryRelativeQuotientVsSpecialized
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_ordinary_relative_quotient_search_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-ordinary-relative-quotient-search".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PhaseCOrdinaryRelativeQuotientSearch
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_post_plateau_ordinary_shoulder_grid_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--post-plateau-ordinary-shoulder-grid".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PostPlateauOrdinaryShoulderGrid));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_post_plateau_ordinary_shoulder_search_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--post-plateau-ordinary-shoulder-search".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PostPlateauOrdinaryShoulderSearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_post_plateau_ordinary_envelope_grid_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--post-plateau-ordinary-envelope-grid".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PostPlateauOrdinaryEnvelopeGrid));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_post_plateau_ordinary_envelope_search_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--post-plateau-ordinary-envelope-search".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PostPlateauOrdinaryEnvelopeSearch));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_post_plateau_ordinary_envelope_vs_shoulder_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--post-plateau-ordinary-envelope-vs-shoulder".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PostPlateauOrdinaryEnvelopeVsShoulder
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_post_plateau_ordinary_hierarchy_grid_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--post-plateau-ordinary-hierarchy-grid".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PostPlateauOrdinaryHierarchyGrid));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_post_plateau_ordinary_hierarchy_vs_envelope_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--post-plateau-ordinary-hierarchy-vs-envelope".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PostPlateauOrdinaryHierarchyVsEnvelope
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_post_plateau_ordinary_assembly_grid_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--post-plateau-ordinary-assembly-grid".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PostPlateauOrdinaryAssemblyGrid));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_post_plateau_ordinary_assembly_vs_hierarchy_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--post-plateau-ordinary-assembly-vs-hierarchy".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PostPlateauOrdinaryAssemblyVsHierarchy
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_post_plateau_ordinary_quasi_literature_grid_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--post-plateau-ordinary-quasi-literature-grid".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PostPlateauOrdinaryQuasiLiteratureGrid
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_post_plateau_ordinary_quasi_literature_vs_assembly_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--post-plateau-ordinary-quasi-literature-vs-assembly".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PostPlateauOrdinaryQuasiLiteratureVsAssembly
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_post_plateau_ordinary_quasi_literature_vs_assembly_dense_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--post-plateau-ordinary-quasi-literature-vs-assembly-dense".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PostPlateauOrdinaryQuasiLiteratureVsAssemblyDense
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_post_plateau_ordinary_dr_like_grid_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--post-plateau-ordinary-dr-like-grid".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PostPlateauOrdinaryDrLikeGrid));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_post_plateau_ordinary_dr_like_vs_quasi_literature_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--post-plateau-ordinary-dr-like-vs-quasi-literature".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PostPlateauOrdinaryDrLikeVsQuasiLiterature
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_post_plateau_triptych_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--post-plateau-triptych-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PostPlateauTriptychCompare));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_boundary_relative_quotient_step_bridge_compare_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-boundary-relative-quotient-step-bridge-compare".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PhaseCBoundaryRelativeQuotientStepBridgeCompare
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_boundary_relative_quotient_step_bridge_search_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-boundary-relative-quotient-step-bridge-search".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PhaseCBoundaryRelativeQuotientStepBridgeSearch
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_boundary_vs_relative_quotient_step_bridge_dense_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-boundary-vs-relative-quotient-step-bridge-dense".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(
            cli.mode,
            Mode::PhaseCBoundaryVsRelativeQuotientStepBridgeDense
        ));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_accepts_phase_c_step_band_vs_step_bridge_dense_mode() {
        let args = vec![
            "rivat3".to_string(),
            "--phase-c-step-band-vs-step-bridge-dense".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::PhaseCStepBandVsStepBridgeDense));
        assert_eq!(cli.experimental_mode, ExperimentalMode::None);
    }

    #[test]
    fn parse_cli_rejects_experimental_mode_outside_profile() {
        let args = vec![
            "rivat3".to_string(),
            "2e7".to_string(),
            "--experimental-easy-relative-to-hard".to_string(),
            "2".to_string(),
        ];

        assert!(parse_cli(&args).is_err());
    }

    #[test]
    fn parse_cli_accepts_experimental_mode_in_sweep() {
        let args = vec![
            "rivat3".to_string(),
            "--sweep".to_string(),
            "1e8".to_string(),
            "--hard-leaf-term-max".to_string(),
            "3".to_string(),
            "--experimental-easy-relative-to-hard".to_string(),
            "2".to_string(),
        ];

        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::Sweep { x_max: 100_000_000 }));
        assert_eq!(
            cli.experimental_mode,
            ExperimentalMode::EasyRelativeToHard { width: 2 }
        );
    }

    #[test]
    fn parse_cli_rejects_multiple_experimental_modes() {
        let args = vec![
            "rivat3".to_string(),
            "--profile".to_string(),
            "2e7".to_string(),
            "--experimental-easy-relative-to-hard".to_string(),
            "2".to_string(),
            "--experimental-hard-relative-to-easy".to_string(),
            "1".to_string(),
        ];

        assert!(parse_cli(&args).is_err());
    }
}
