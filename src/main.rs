use rivat3::dr;
use rivat3::sieve::lucy_phi_early_stop_profiled;
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
        | Mode::DrMeissel4Profile { x }
        | Mode::LucyProfile { x } => Some(*x),
        Mode::Sweep { x_max } => Some(*x_max),
        Mode::Batch { jobs } => jobs.iter().map(|j| j.x).max(),
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

#[allow(dead_code)]
fn pct(part: std::time::Duration, total: std::time::Duration) -> f64 {
    if total.is_zero() {
        0.0
    } else {
        100.0 * part.as_secs_f64() / total.as_secs_f64()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BatchJob {
    label: String,
    x: u128,
}

struct Cli {
    mode: Mode,
    threads: usize,
    #[allow(dead_code)]
    hard_leaf_term_max: u128,
    #[allow(dead_code)]
    easy_leaf_term_max: u128,
    #[allow(dead_code)]
    experimental_mode: ExperimentalMode,
    alpha_override: Option<f64>,
}

enum Mode {
    Normal { label: String, x: u128 },
    Batch { jobs: Vec<BatchJob> },
    DrMeissel4Profile { x: u128 },
    LucyProfile { x: u128 },
    Sweep { x_max: u128 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExperimentalMode {
    None,
}

fn parse_cli(args: &[String]) -> Result<Cli, String> {
    let mut threads = default_threads();
    let mut positional_jobs: Vec<BatchJob> = Vec::new();
    let mut hard_leaf_term_max = rivat3::parameters::Parameters::DEFAULT_HARD_LEAF_TERM_MAX;
    let mut easy_leaf_term_max = rivat3::parameters::Parameters::DEFAULT_EASY_LEAF_TERM_VALUE;
    let experimental_mode = ExperimentalMode::None;
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
            "--lucy-profile" => {
                let take_value = args.get(i + 1).filter(|s| !s.starts_with('-'));
                let x = parse_x(take_value.map(|s| s.as_str()).unwrap_or("1000000000000"))?;
                mode = Some(Mode::LucyProfile { x });
                i += if take_value.is_some() { 2 } else { 1 };
            }
            "--dr-meissel4" | "--dr-profile" => {
                let take_value = args.get(i + 1).filter(|s| !s.starts_with('-'));
                let x = parse_x(take_value.map(|s| s.as_str()).unwrap_or("1000000000000"))?;
                mode = Some(Mode::DrMeissel4Profile { x });
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
            other if other.starts_with('-') => {
                return Err(format!("unknown option: {}", other));
            }
            other => {
                let x = parse_x(other)?;
                positional_jobs.push(BatchJob { label: other.to_string(), x });
                i += 1;
            }
        }
    }

    let mode = if !positional_jobs.is_empty() {
        if mode.is_some() {
            return Err(
                "cannot combine positional x values with another explicit mode".to_string(),
            );
        }
        if positional_jobs.len() == 1 {
            let job = positional_jobs.into_iter().next().unwrap();
            Mode::Normal { label: job.label, x: job.x }
        } else {
            Mode::Batch { jobs: positional_jobs }
        }
    } else {
        mode.ok_or_else(|| {
            "missing x or mode (--dr-profile / --lucy-profile / --sweep)".to_string()
        })?
    };
    if easy_leaf_term_max > hard_leaf_term_max {
        return Err("easy-leaf-term-max must be <= hard-leaf-term-max".to_string());
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

fn run_batch(jobs: &[BatchJob]) {
    #[derive(Clone)]
    struct BatchResult {
        label: String,
        x: u128,
        result: u128,
        elapsed: std::time::Duration,
    }

    let (l3_mb, _, _) = rivat3::parameters::detect_hw();
    let mut results = Vec::with_capacity(jobs.len());

    for job in jobs {
        let alpha = rivat3::parameters::choose_alpha(job.x);
        let engine = if dr::uses_baseline_fallback(job.x) { "baseline" } else { "DR-v4" };
        println!(
            "n = {}  |  Calcul en cours...  [primerivat {} | L3={}Mo α={} {}]",
            job.label,
            env!("GIT_HASH"),
            l3_mb,
            alpha,
            engine
        );
        let _ = std::io::stdout().flush();

        let x = job.x;
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
            job.label,
            fmt_thousands(result),
            fmt_elapsed(elapsed)
        );
        println!();

        results.push(BatchResult {
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
fn run_sweep(x_max: u128, _threads: usize) {
    println!(
        "{:>22}  {:>12}  result",
        "x", "total"
    );
    println!("{}", "-".repeat(55));

    let mut x = 10u128;
    while x <= x_max {
        let xc = x;
        let t0 = Instant::now();
        let result = std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(move || dr::prime_pi_dr_meissel_v4(xc))
            .unwrap()
            .join()
            .unwrap();
        let total = t0.elapsed();

        println!(
            "{:>22}  {:>12}  {}",
            fmt_thousands(x),
            fmt_elapsed(total),
            fmt_thousands(result)
        );

        x *= 10;
    }
}

fn print_usage(program: &str) {
    eprintln!("Usage:");
    eprintln!("  {} <x> [<x> ...]         Compute π(x) (batch mode when >1 x given)", program);
    eprintln!("  {} --sweep [x_max]       Sweep x = 10, 100, … up to x_max (default 1e12)", program);
    eprintln!("  {} --dr-profile <x>      Timing breakdown for π(x) via the DR engine", program);
    eprintln!("  {} --lucy-profile <x>    Timing breakdown for π(x) via the Lucy baseline", program);
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -a <α>, --alpha <α>");
    eprintln!("                     Override the hardware-adaptive α selection.");
    eprintln!("                     α ∈ [1, 2] for x ≤ 1e15; α ∈ {{1, 2}} only for x > 1e15.");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {} 1e13", program);
    eprintln!("  {} 1e17 --alpha 2", program);
    eprintln!("  {} 1e11 1e12 1e13 1e14", program);
    eprintln!("  {} --sweep 1e14", program);
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
        Mode::Sweep { x_max } => run_sweep(x_max, cli.threads),
        Mode::Batch { jobs } => run_batch(&jobs),
        Mode::DrMeissel4Profile { x } => {
            // Deep phi recursion at x>=1e14 needs a larger main-thread stack.
            let t = cli.threads;
            std::thread::Builder::new().stack_size(64 << 20)
                .spawn(move || run_dr_meissel4_profile(x, t)).unwrap().join().unwrap();
        }
        Mode::LucyProfile { x } => { lucy_phi_early_stop_profiled(x); }
        Mode::Normal { label, x } => {
            use std::io::Write;
            let alpha = rivat3::parameters::choose_alpha(x);
            let engine = if dr::uses_baseline_fallback(x) { "baseline" } else { "DR-v4" };
            println!(
                "n = {}  |  Calcul en cours...  [primerivat {} | L3={}Mo α={} {}]",
                label, env!("GIT_HASH"), l3_mb, alpha, engine
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
    use super::{parse_cli, Mode};

    #[test]
    fn parse_cli_accepts_normal_x() {
        let args = vec!["rivat3".to_string(), "1e12".to_string()];
        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::Normal { x: 1_000_000_000_000, .. }));
    }

    #[test]
    fn parse_cli_accepts_alpha_override() {
        let args = vec![
            "rivat3".to_string(),
            "1e15".to_string(),
            "--alpha".to_string(),
            "2".to_string(),
        ];
        let cli = parse_cli(&args).expect("CLI should parse");
        assert_eq!(cli.alpha_override, Some(2.0));
    }

    #[test]
    fn parse_cli_accepts_sweep_mode() {
        let args = vec!["rivat3".to_string(), "--sweep".to_string(), "1e12".to_string()];
        let cli = parse_cli(&args).expect("CLI should parse");
        assert!(matches!(cli.mode, Mode::Sweep { x_max: 1_000_000_000_000 }));
    }

    #[test]
    fn parse_cli_rejects_unknown_option() {
        let args = vec!["rivat3".to_string(), "--nope".to_string()];
        assert!(parse_cli(&args).is_err());
    }
}
