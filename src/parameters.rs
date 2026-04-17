use crate::math::{icbrt, isqrt};
use crate::sieve::pi_at;
use std::sync::OnceLock;

static ALPHA_OVERRIDE: OnceLock<f64> = OnceLock::new();

/// Pins the process-wide alpha value, bypassing hardware-adaptive selection.
/// Only the first call takes effect; returns `Err` if already set.
pub fn set_alpha_override(alpha: f64) -> Result<(), f64> {
    ALPHA_OVERRIDE.set(alpha)
}

/// Hardware-adaptive alpha selector for the DR algorithm.
///
/// Returns 2.0 only when x is large enough AND the CPU has a small L3
/// with few cores — otherwise α=1.0 wins (measured: α=2.0 regresses 35%
/// on i5-13450HX at 1e17 but gains 12% on i5-9300H).
///
/// A process-wide override set via [`set_alpha_override`] takes precedence.
pub fn choose_alpha(x: u128) -> f64 {
    if let Some(&alpha) = ALPHA_OVERRIDE.get() {
        return alpha;
    }
    if x < 30_000_000_000_000_000u128 {
        return 1.0;
    }
    let l3_mb = cache_size::l3_cache_size().unwrap_or(8 << 20) >> 20;
    let cores = num_cpus::get_physical();
    if l3_mb < 16 && cores <= 8 { 2.0 } else { 1.0 }
}

/// One-time detection snapshot for startup logging.
pub fn detect_hw() -> (usize, usize, usize) {
    let l3_mb = cache_size::l3_cache_size().unwrap_or(8 << 20) >> 20;
    let cores = num_cpus::get_physical();
    let threads = num_cpus::get();
    (l3_mb, cores, threads)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrTuning {
    pub alpha: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EasyRelativeToHardCandidate {
    pub width: u128,
    pub min_term_floor: u128,
}

impl EasyRelativeToHardCandidate {
    pub fn build_rule(self, hard_term_min: u128, alpha: f64) -> S2EasyRule {
        S2EasyRule::alpha_balanced_relative_to_hard_with_floor(
            hard_term_min,
            self.width,
            self.min_term_floor,
            alpha,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EasyTermBandCandidate {
    pub min_term: u128,
    pub max_term: u128,
}

impl EasyTermBandCandidate {
    pub fn build_rule(self, alpha: f64) -> S2EasyRule {
        S2EasyRule::alpha_balanced_term_range(self.min_term, self.max_term, alpha)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardTermBandCandidate {
    pub min_term: u128,
    pub max_term: u128,
}

impl HardTermBandCandidate {
    pub fn build_rule(self, alpha: f64) -> S2HardRule {
        S2HardRule::alpha_balanced_term_range(self.min_term, self.max_term, alpha)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseCLinkedCandidate {
    pub easy_width: u128,
    pub easy_min_term_floor: u128,
    pub hard_width: u128,
}

impl PhaseCLinkedCandidate {
    pub fn build_frontier_set(self, alpha: f64) -> FrontierSet {
        let hard_rule = S2HardRule::alpha_balanced_relative_to_easy(
            self.easy_upper_bound(),
            self.hard_width,
            alpha,
        );
        let easy_rule = S2EasyRule::alpha_balanced_relative_to_hard_with_floor(
            hard_rule.term_min(),
            self.easy_width,
            self.easy_min_term_floor,
            alpha,
        );

        FrontierSet {
            s1: S1Rule {
                term_min_exclusive: hard_rule.term_max(),
            },
            s2_hard: hard_rule,
            s2_easy: easy_rule,
        }
    }

    pub fn easy_upper_bound(self) -> u128 {
        self.easy_min_term_floor
            .max(1)
            .saturating_add(self.easy_width.max(1) - 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseCBoundaryCandidate {
    pub boundary_term: u128,
    pub easy_width: u128,
    pub hard_width: u128,
}

impl PhaseCBoundaryCandidate {
    pub fn build_frontier_set(self, alpha: f64) -> FrontierSet {
        let easy_min = self
            .boundary_term
            .saturating_sub(self.easy_width.max(1) - 1)
            .max(1);
        let easy_rule = S2EasyRule::alpha_balanced_term_range(
            easy_min,
            self.boundary_term.max(easy_min),
            alpha,
        );
        let hard_rule = S2HardRule::alpha_balanced_term_range(
            self.boundary_term.saturating_add(1),
            self.boundary_term.saturating_add(self.hard_width.max(1)),
            alpha,
        );

        FrontierSet {
            s1: S1Rule {
                term_min_exclusive: hard_rule.term_max(),
            },
            s2_hard: hard_rule,
            s2_easy: easy_rule,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseCBufferedBoundaryCandidate {
    pub boundary_term: u128,
    pub easy_width: u128,
    pub gap_width: u128,
    pub hard_width: u128,
}

impl PhaseCBufferedBoundaryCandidate {
    pub fn build_frontier_set(self, alpha: f64) -> FrontierSet {
        let easy_min = self
            .boundary_term
            .saturating_sub(self.easy_width.max(1) - 1)
            .max(1);
        let easy_max = self.boundary_term.max(easy_min);
        let hard_min = easy_max
            .saturating_add(self.gap_width.max(1))
            .saturating_add(1);
        let hard_max = hard_min.saturating_add(self.hard_width.max(1) - 1);
        let easy_rule = S2EasyRule::alpha_balanced_term_range(easy_min, easy_max, alpha);
        let hard_rule = S2HardRule::alpha_balanced_term_range(hard_min, hard_max, alpha);

        FrontierSet {
            s1: S1Rule {
                term_min_exclusive: hard_rule.term_max(),
            },
            s2_hard: hard_rule,
            s2_easy: easy_rule,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseCQuotientWindowCandidate {
    pub easy_q_offset_max: u128,
    pub hard_q_width: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseCBoundaryQuotientGuardCandidate {
    pub boundary_term: u128,
    pub easy_width: u128,
    pub hard_width: u128,
    pub guard_q_offset: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseCBoundaryRelativeQuotientBandCandidate {
    pub boundary_term: u128,
    pub easy_width: u128,
    pub hard_width: u128,
    pub easy_q_band_width: u128,
    pub hard_q_band_width: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseCBoundaryRelativeQuotientStepBandCandidate {
    pub boundary_term: u128,
    pub easy_width: u128,
    pub hard_width: u128,
    pub easy_q_step_multiplier: u128,
    pub hard_q_step_multiplier: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseCBoundaryRelativeQuotientStepBridgeCandidate {
    pub boundary_term: u128,
    pub easy_width: u128,
    pub hard_width: u128,
    pub easy_q_step_multiplier: u128,
    pub hard_q_step_multiplier: u128,
    pub bridge_width: u128,
}

impl DrTuning {
    pub const DEFAULT_ALPHA: f64 = 1.0;
    pub const CANDIDATE_EASY_RELATIVE_TO_HARD_WIDTH: u128 = 2;
    pub const CANDIDATE_EASY_RELATIVE_TO_HARD_MIN_TERM_FLOOR: u128 = 1;
    pub const CANDIDATE_EASY_TERM_BAND_MIN: u128 = 1;
    pub const CANDIDATE_EASY_TERM_BAND_MAX: u128 = 2;
    pub const PHASE_C_EASY_TERM_BAND_MIN: u128 = 1;
    pub const PHASE_C_EASY_TERM_BAND_MAX: u128 = 4;
    pub const PHASE_C_HARD_TERM_BAND_MIN: u128 = 5;
    pub const PHASE_C_HARD_TERM_BAND_MAX: u128 = 6;
    pub const PHASE_C_LINKED_EASY_WIDTH: u128 = 5;
    pub const PHASE_C_LINKED_EASY_MIN_TERM_FLOOR: u128 = 1;
    pub const PHASE_C_LINKED_HARD_WIDTH: u128 = 1;
    pub const PHASE_C_BOUNDARY_TERM: u128 = 5;
    pub const PHASE_C_BOUNDARY_EASY_WIDTH: u128 = 5;
    pub const PHASE_C_BOUNDARY_HARD_WIDTH: u128 = 1;
    pub const PHASE_C_BUFFERED_BOUNDARY_TERM: u128 = 4;
    pub const PHASE_C_BUFFERED_EASY_WIDTH: u128 = 4;
    pub const PHASE_C_BUFFERED_GAP_WIDTH: u128 = 1;
    pub const PHASE_C_BUFFERED_HARD_WIDTH: u128 = 1;
    pub const PHASE_C_QUOTIENT_WINDOW_EASY_Q_OFFSET_MAX: u128 = 0;
    pub const PHASE_C_QUOTIENT_WINDOW_HARD_Q_WIDTH: u128 = 1;
    pub const PHASE_C_BOUNDARY_QUOTIENT_GUARD_TERM: u128 = 4;
    pub const PHASE_C_BOUNDARY_QUOTIENT_GUARD_EASY_WIDTH: u128 = 4;
    pub const PHASE_C_BOUNDARY_QUOTIENT_GUARD_HARD_WIDTH: u128 = 2;
    pub const PHASE_C_BOUNDARY_QUOTIENT_GUARD_Q_OFFSET: u128 = 0;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_BAND_TERM: u128 = 4;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_BAND_EASY_WIDTH: u128 = 4;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_BAND_HARD_WIDTH: u128 = 2;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_BAND_EASY_Q_BAND_WIDTH: u128 = 0;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_BAND_HARD_Q_BAND_WIDTH: u128 = 0;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BAND_TERM: u128 = 4;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BAND_EASY_WIDTH: u128 = 4;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BAND_HARD_WIDTH: u128 = 2;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BAND_EASY_Q_STEP_MULTIPLIER: u128 = 0;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BAND_HARD_Q_STEP_MULTIPLIER: u128 = 0;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BRIDGE_TERM: u128 = 4;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BRIDGE_EASY_WIDTH: u128 = 4;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BRIDGE_HARD_WIDTH: u128 = 2;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BRIDGE_EASY_Q_STEP_MULTIPLIER: u128 = 1;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BRIDGE_HARD_Q_STEP_MULTIPLIER: u128 = 1;
    pub const PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BRIDGE_WIDTH: u128 = 1;

    pub fn meissel_default() -> Self {
        Self {
            alpha: Self::DEFAULT_ALPHA,
        }
    }

    pub fn candidate_easy_relative_to_hard() -> EasyRelativeToHardCandidate {
        EasyRelativeToHardCandidate {
            width: Self::CANDIDATE_EASY_RELATIVE_TO_HARD_WIDTH,
            min_term_floor: Self::CANDIDATE_EASY_RELATIVE_TO_HARD_MIN_TERM_FLOOR,
        }
    }

    pub fn candidate_easy_term_band() -> EasyTermBandCandidate {
        EasyTermBandCandidate {
            min_term: Self::CANDIDATE_EASY_TERM_BAND_MIN,
            max_term: Self::CANDIDATE_EASY_TERM_BAND_MAX,
        }
    }

    pub fn phase_c_easy_term_band() -> EasyTermBandCandidate {
        EasyTermBandCandidate {
            min_term: Self::PHASE_C_EASY_TERM_BAND_MIN,
            max_term: Self::PHASE_C_EASY_TERM_BAND_MAX,
        }
    }

    pub fn phase_c_hard_term_band() -> HardTermBandCandidate {
        HardTermBandCandidate {
            min_term: Self::PHASE_C_HARD_TERM_BAND_MIN,
            max_term: Self::PHASE_C_HARD_TERM_BAND_MAX,
        }
    }

    pub fn phase_c_linked_candidate() -> PhaseCLinkedCandidate {
        PhaseCLinkedCandidate {
            easy_width: Self::PHASE_C_LINKED_EASY_WIDTH,
            easy_min_term_floor: Self::PHASE_C_LINKED_EASY_MIN_TERM_FLOOR,
            hard_width: Self::PHASE_C_LINKED_HARD_WIDTH,
        }
    }

    pub fn phase_c_boundary_candidate() -> PhaseCBoundaryCandidate {
        PhaseCBoundaryCandidate {
            boundary_term: Self::PHASE_C_BOUNDARY_TERM,
            easy_width: Self::PHASE_C_BOUNDARY_EASY_WIDTH,
            hard_width: Self::PHASE_C_BOUNDARY_HARD_WIDTH,
        }
    }

    pub fn phase_c_buffered_boundary_candidate() -> PhaseCBufferedBoundaryCandidate {
        PhaseCBufferedBoundaryCandidate {
            boundary_term: Self::PHASE_C_BUFFERED_BOUNDARY_TERM,
            easy_width: Self::PHASE_C_BUFFERED_EASY_WIDTH,
            gap_width: Self::PHASE_C_BUFFERED_GAP_WIDTH,
            hard_width: Self::PHASE_C_BUFFERED_HARD_WIDTH,
        }
    }

    pub fn phase_c_quotient_window_candidate() -> PhaseCQuotientWindowCandidate {
        PhaseCQuotientWindowCandidate {
            easy_q_offset_max: Self::PHASE_C_QUOTIENT_WINDOW_EASY_Q_OFFSET_MAX,
            hard_q_width: Self::PHASE_C_QUOTIENT_WINDOW_HARD_Q_WIDTH,
        }
    }

    pub fn phase_c_boundary_quotient_guard_candidate() -> PhaseCBoundaryQuotientGuardCandidate {
        PhaseCBoundaryQuotientGuardCandidate {
            boundary_term: Self::PHASE_C_BOUNDARY_QUOTIENT_GUARD_TERM,
            easy_width: Self::PHASE_C_BOUNDARY_QUOTIENT_GUARD_EASY_WIDTH,
            hard_width: Self::PHASE_C_BOUNDARY_QUOTIENT_GUARD_HARD_WIDTH,
            guard_q_offset: Self::PHASE_C_BOUNDARY_QUOTIENT_GUARD_Q_OFFSET,
        }
    }

    pub fn phase_c_boundary_relative_quotient_band_candidate()
    -> PhaseCBoundaryRelativeQuotientBandCandidate {
        PhaseCBoundaryRelativeQuotientBandCandidate {
            boundary_term: Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_BAND_TERM,
            easy_width: Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_BAND_EASY_WIDTH,
            hard_width: Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_BAND_HARD_WIDTH,
            easy_q_band_width: Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_BAND_EASY_Q_BAND_WIDTH,
            hard_q_band_width: Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_BAND_HARD_Q_BAND_WIDTH,
        }
    }

    pub fn phase_c_boundary_relative_quotient_step_band_candidate()
    -> PhaseCBoundaryRelativeQuotientStepBandCandidate {
        PhaseCBoundaryRelativeQuotientStepBandCandidate {
            boundary_term: Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BAND_TERM,
            easy_width: Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BAND_EASY_WIDTH,
            hard_width: Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BAND_HARD_WIDTH,
            easy_q_step_multiplier:
                Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BAND_EASY_Q_STEP_MULTIPLIER,
            hard_q_step_multiplier:
                Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BAND_HARD_Q_STEP_MULTIPLIER,
        }
    }

    pub fn phase_c_boundary_relative_quotient_step_bridge_candidate()
    -> PhaseCBoundaryRelativeQuotientStepBridgeCandidate {
        PhaseCBoundaryRelativeQuotientStepBridgeCandidate {
            boundary_term: Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BRIDGE_TERM,
            easy_width: Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BRIDGE_EASY_WIDTH,
            hard_width: Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BRIDGE_HARD_WIDTH,
            easy_q_step_multiplier:
                Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BRIDGE_EASY_Q_STEP_MULTIPLIER,
            hard_q_step_multiplier:
                Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BRIDGE_HARD_Q_STEP_MULTIPLIER,
            bridge_width: Self::PHASE_C_BOUNDARY_RELATIVE_QUOTIENT_STEP_BRIDGE_WIDTH,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexDomain {
    pub start: usize,
    pub end: usize,
}

impl IndexDomain {
    pub fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(self) -> bool {
        self.start >= self.end
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrivialDomain {
    pub s3_is_zero: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S2HardRuleKind {
    TermRange {
        min: u128,
        max: u128,
    },
    AlphaBalancedTermRange {
        min: u128,
        max: u128,
        alpha_milli: u32,
    },
    RelativeToEasy {
        easy_term_max: u128,
        width: u128,
        alpha_milli: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S2HardRule {
    pub kind: S2HardRuleKind,
}

impl S2HardRule {
    pub fn term_range(min: u128, max: u128) -> Self {
        Self {
            kind: S2HardRuleKind::TermRange { min, max },
        }
    }

    pub fn alpha_balanced_term_range(min: u128, max: u128, alpha: f64) -> Self {
        let alpha_milli = (alpha * 1000.0).round().clamp(0.0, u32::MAX as f64) as u32;
        Self {
            kind: S2HardRuleKind::AlphaBalancedTermRange {
                min,
                max,
                alpha_milli,
            },
        }
    }

    pub fn alpha_balanced_relative_to_easy(easy_term_max: u128, width: u128, alpha: f64) -> Self {
        let alpha_milli = (alpha * 1000.0).round().clamp(0.0, u32::MAX as f64) as u32;
        Self {
            kind: S2HardRuleKind::RelativeToEasy {
                easy_term_max,
                width: width.max(1),
                alpha_milli,
            },
        }
    }

    pub fn matches(self, term: u128) -> bool {
        let (min, max) = self.term_bounds();
        term >= min && term <= max
    }

    pub fn term_bounds(self) -> (u128, u128) {
        match self.kind {
            S2HardRuleKind::TermRange { min, max } => (min, max),
            S2HardRuleKind::AlphaBalancedTermRange { min, max, .. } => (min, max),
            S2HardRuleKind::RelativeToEasy {
                easy_term_max,
                width,
                ..
            } => {
                let min = easy_term_max.saturating_add(1);
                let max = easy_term_max.saturating_add(width.max(1));
                (min, max.max(min))
            }
        }
    }

    pub fn alpha(self) -> Option<f64> {
        match self.kind {
            S2HardRuleKind::AlphaBalancedTermRange { alpha_milli, .. } => {
                Some(alpha_milli as f64 / 1000.0)
            }
            S2HardRuleKind::RelativeToEasy { alpha_milli, .. } => Some(alpha_milli as f64 / 1000.0),
            _ => None,
        }
    }

    pub fn term_min(self) -> u128 {
        self.term_bounds().0
    }

    pub fn term_max(self) -> u128 {
        self.term_bounds().1
    }

    pub fn kind_name(self) -> &'static str {
        match self.kind {
            S2HardRuleKind::TermRange { .. } => "term_range",
            S2HardRuleKind::AlphaBalancedTermRange { .. } => "alpha_balanced_term_range",
            S2HardRuleKind::RelativeToEasy { .. } => "relative_to_easy",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S2EasyRuleKind {
    TermEquals {
        value: u128,
    },
    TermRange {
        min: u128,
        max: u128,
    },
    AlphaBalancedTermRange {
        min: u128,
        max: u128,
        alpha_milli: u32,
    },
    RelativeToHard {
        hard_term_min: u128,
        width: u128,
        alpha_milli: u32,
    },
    RelativeToHardWithFloor {
        hard_term_min: u128,
        width: u128,
        min_term_floor: u128,
        alpha_milli: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S2EasyRule {
    pub kind: S2EasyRuleKind,
}

impl S2EasyRule {
    fn relative_to_hard_bounds(
        hard_term_min: u128,
        width: u128,
        min_term_floor: u128,
    ) -> (u128, u128) {
        let max = hard_term_min.saturating_sub(1).max(1);
        let unclamped_min = max.saturating_sub(width.max(1) - 1);
        let min = unclamped_min.max(min_term_floor.max(1)).min(max);
        (min, max)
    }

    pub fn term_equals(value: u128) -> Self {
        Self {
            kind: S2EasyRuleKind::TermEquals { value },
        }
    }

    pub fn term_range(min: u128, max: u128) -> Self {
        Self {
            kind: S2EasyRuleKind::TermRange { min, max },
        }
    }

    pub fn alpha_balanced_term_range(min: u128, max: u128, alpha: f64) -> Self {
        let alpha_milli = (alpha * 1000.0).round().clamp(0.0, u32::MAX as f64) as u32;
        Self {
            kind: S2EasyRuleKind::AlphaBalancedTermRange {
                min,
                max,
                alpha_milli,
            },
        }
    }

    pub fn alpha_balanced_relative_to_hard(hard_term_min: u128, width: u128, alpha: f64) -> Self {
        let alpha_milli = (alpha * 1000.0).round().clamp(0.0, u32::MAX as f64) as u32;
        Self {
            kind: S2EasyRuleKind::RelativeToHard {
                hard_term_min,
                width: width.max(1),
                alpha_milli,
            },
        }
    }

    pub fn alpha_balanced_relative_to_hard_with_floor(
        hard_term_min: u128,
        width: u128,
        min_term_floor: u128,
        alpha: f64,
    ) -> Self {
        let alpha_milli = (alpha * 1000.0).round().clamp(0.0, u32::MAX as f64) as u32;
        Self {
            kind: S2EasyRuleKind::RelativeToHardWithFloor {
                hard_term_min,
                width: width.max(1),
                min_term_floor: min_term_floor.max(1),
                alpha_milli,
            },
        }
    }

    pub fn matches(self, term: u128) -> bool {
        match self.kind {
            S2EasyRuleKind::TermEquals { value } => term == value,
            S2EasyRuleKind::TermRange { min, max } => term >= min && term <= max,
            S2EasyRuleKind::AlphaBalancedTermRange { min, max, .. } => term >= min && term <= max,
            S2EasyRuleKind::RelativeToHard {
                hard_term_min,
                width,
                ..
            } => {
                let (min, max) = Self::relative_to_hard_bounds(hard_term_min, width, 1);
                term >= min && term <= max
            }
            S2EasyRuleKind::RelativeToHardWithFloor {
                hard_term_min,
                width,
                min_term_floor,
                ..
            } => {
                let (min, max) =
                    Self::relative_to_hard_bounds(hard_term_min, width, min_term_floor);
                term >= min && term <= max
            }
        }
    }

    pub fn representative_term_value(self) -> u128 {
        match self.kind {
            S2EasyRuleKind::TermEquals { value } => value,
            S2EasyRuleKind::TermRange { min, .. } => min,
            S2EasyRuleKind::AlphaBalancedTermRange { min, .. } => min,
            S2EasyRuleKind::RelativeToHard {
                hard_term_min,
                width,
                ..
            } => Self::relative_to_hard_bounds(hard_term_min, width, 1).0,
            S2EasyRuleKind::RelativeToHardWithFloor {
                hard_term_min,
                width,
                min_term_floor,
                ..
            } => Self::relative_to_hard_bounds(hard_term_min, width, min_term_floor).0,
        }
    }

    pub fn term_bounds(self) -> (u128, u128) {
        match self.kind {
            S2EasyRuleKind::TermEquals { value } => (value, value),
            S2EasyRuleKind::TermRange { min, max } => (min, max),
            S2EasyRuleKind::AlphaBalancedTermRange { min, max, .. } => (min, max),
            S2EasyRuleKind::RelativeToHard {
                hard_term_min,
                width,
                ..
            } => Self::relative_to_hard_bounds(hard_term_min, width, 1),
            S2EasyRuleKind::RelativeToHardWithFloor {
                hard_term_min,
                width,
                min_term_floor,
                ..
            } => Self::relative_to_hard_bounds(hard_term_min, width, min_term_floor),
        }
    }

    pub fn alpha(self) -> Option<f64> {
        match self.kind {
            S2EasyRuleKind::AlphaBalancedTermRange { alpha_milli, .. } => {
                Some(alpha_milli as f64 / 1000.0)
            }
            S2EasyRuleKind::RelativeToHard { alpha_milli, .. } => Some(alpha_milli as f64 / 1000.0),
            S2EasyRuleKind::RelativeToHardWithFloor { alpha_milli, .. } => {
                Some(alpha_milli as f64 / 1000.0)
            }
            _ => None,
        }
    }

    pub fn kind_name(self) -> &'static str {
        match self.kind {
            S2EasyRuleKind::TermEquals { .. } => "term_equals",
            S2EasyRuleKind::TermRange { .. } => "term_range",
            S2EasyRuleKind::AlphaBalancedTermRange { .. } => "alpha_balanced_term_range",
            S2EasyRuleKind::RelativeToHard { .. } => "relative_to_hard",
            S2EasyRuleKind::RelativeToHardWithFloor { .. } => "relative_to_hard_with_floor",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrontierSet {
    pub s1: S1Rule,
    pub s2_hard: S2HardRule,
    pub s2_easy: S2EasyRule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S1Rule {
    pub term_min_exclusive: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S1Domain {
    pub leaves: IndexDomain,
    pub rule: S1Rule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S2TrivialDomain {
    pub leaves: TrivialDomain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S2EasyDomain {
    pub leaves: IndexDomain,
    pub rule: S2EasyRule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S2HardDomain {
    pub leaves: IndexDomain,
    pub rule: S2HardRule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrDomains {
    pub active: IndexDomain,
    pub s1: S1Domain,
    pub s2_trivial: S2TrivialDomain,
    pub s2_easy: S2EasyDomain,
    pub s2_hard: S2HardDomain,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parameters {
    pub tuning: DrTuning,
    pub x: u128,
    pub y: u128,
    pub z: u128,
    pub z_usize: usize,
    pub a: usize,
    pub b: usize,
    pub hard_leaf_term_max: u128,
    pub easy_leaf_term_max: u128,
}

impl Parameters {
    pub const DEFAULT_HARD_LEAF_TERM_MAX: u128 = 2;
    pub const DEFAULT_EASY_LEAF_TERM_VALUE: u128 = 1;

    pub fn from_counts(x: u128, a: usize, b: usize) -> Self {
        let tuning = DrTuning::meissel_default();
        let y = icbrt(x);
        let z = isqrt(x);
        Self {
            tuning,
            x,
            y,
            z,
            z_usize: z as usize,
            a,
            b,
            hard_leaf_term_max: Self::DEFAULT_HARD_LEAF_TERM_MAX,
            easy_leaf_term_max: Self::DEFAULT_EASY_LEAF_TERM_VALUE,
        }
    }

    pub fn from_tables(x: u128, small: &[u64], large: &[u64]) -> Self {
        let tuning = DrTuning::meissel_default();
        let y = icbrt(x);
        let z = isqrt(x);
        let z_usize = z as usize;
        let a = pi_at(y, x, z_usize, small, large) as usize;
        let b = pi_at(z, x, z_usize, small, large) as usize;

        Self {
            tuning,
            x,
            y,
            z,
            z_usize,
            a,
            b,
            hard_leaf_term_max: Self::DEFAULT_HARD_LEAF_TERM_MAX,
            easy_leaf_term_max: Self::DEFAULT_EASY_LEAF_TERM_VALUE,
        }
    }

    pub fn with_hard_leaf_term_max(mut self, hard_leaf_term_max: u128) -> Self {
        self.hard_leaf_term_max = hard_leaf_term_max.max(2);
        self.easy_leaf_term_max = self.easy_leaf_term_max.min(self.hard_leaf_term_max);
        self
    }

    pub fn with_easy_leaf_term_max(mut self, easy_leaf_term_max: u128) -> Self {
        self.easy_leaf_term_max =
            easy_leaf_term_max.clamp(Self::DEFAULT_EASY_LEAF_TERM_VALUE, self.hard_leaf_term_max);
        self
    }

    pub fn frontier_set(self) -> FrontierSet {
        let hard_term_min = self.easy_leaf_term_max.saturating_add(1);
        FrontierSet {
            s1: S1Rule {
                term_min_exclusive: self.hard_leaf_term_max,
            },
            s2_hard: S2HardRule::alpha_balanced_term_range(
                hard_term_min,
                self.hard_leaf_term_max,
                self.tuning.alpha,
            ),
            s2_easy: S2EasyRule::alpha_balanced_term_range(
                Self::DEFAULT_EASY_LEAF_TERM_VALUE,
                self.easy_leaf_term_max,
                self.tuning.alpha,
            ),
        }
    }

    pub fn candidate_easy_frontier_set(self) -> FrontierSet {
        let candidate = DrTuning::candidate_easy_relative_to_hard();
        let easy_rule = candidate.build_rule(self.hard_leaf_term_max, self.tuning.alpha);
        let (_, easy_leaf_term_max) = easy_rule.term_bounds();

        FrontierSet {
            s1: S1Rule {
                term_min_exclusive: self.hard_leaf_term_max,
            },
            s2_hard: S2HardRule::alpha_balanced_term_range(
                easy_leaf_term_max.saturating_add(1),
                self.hard_leaf_term_max,
                self.tuning.alpha,
            ),
            s2_easy: easy_rule,
        }
    }

    pub fn candidate_easy_term_band_frontier_set(self) -> FrontierSet {
        let candidate = DrTuning::candidate_easy_term_band();
        let easy_rule = candidate.build_rule(self.tuning.alpha);
        let (_, easy_leaf_term_max) = easy_rule.term_bounds();

        FrontierSet {
            s1: S1Rule {
                term_min_exclusive: self.hard_leaf_term_max,
            },
            s2_hard: S2HardRule::alpha_balanced_term_range(
                easy_leaf_term_max.saturating_add(1),
                self.hard_leaf_term_max,
                self.tuning.alpha,
            ),
            s2_easy: easy_rule,
        }
    }

    pub fn phase_c_easy_term_band_frontier_set(self) -> FrontierSet {
        let candidate = DrTuning::phase_c_easy_term_band();
        let easy_rule = candidate.build_rule(self.tuning.alpha);
        let (_, easy_leaf_term_max) = easy_rule.term_bounds();

        FrontierSet {
            s1: S1Rule {
                term_min_exclusive: self.hard_leaf_term_max,
            },
            s2_hard: S2HardRule::alpha_balanced_term_range(
                easy_leaf_term_max.saturating_add(1),
                self.hard_leaf_term_max,
                self.tuning.alpha,
            ),
            s2_easy: easy_rule,
        }
    }

    pub fn phase_c_hard_term_band_frontier_set(self) -> FrontierSet {
        let easy_candidate = DrTuning::phase_c_easy_term_band();
        let hard_candidate = DrTuning::phase_c_hard_term_band();
        let easy_rule = easy_candidate.build_rule(self.tuning.alpha);
        let hard_rule = hard_candidate.build_rule(self.tuning.alpha);

        FrontierSet {
            s1: S1Rule {
                term_min_exclusive: hard_candidate.max_term,
            },
            s2_hard: hard_rule,
            s2_easy: easy_rule,
        }
    }

    pub fn phase_c_term_band_frontier_set(self) -> FrontierSet {
        self.phase_c_hard_term_band_frontier_set()
    }

    pub fn phase_c_frontier_set(self) -> FrontierSet {
        self.phase_c_linked_frontier_set()
    }

    pub fn phase_c_linked_frontier_set(self) -> FrontierSet {
        DrTuning::phase_c_linked_candidate().build_frontier_set(self.tuning.alpha)
    }

    pub fn phase_c_boundary_frontier_set(self) -> FrontierSet {
        DrTuning::phase_c_boundary_candidate().build_frontier_set(self.tuning.alpha)
    }

    pub fn phase_c_buffered_boundary_frontier_set(self) -> FrontierSet {
        DrTuning::phase_c_buffered_boundary_candidate().build_frontier_set(self.tuning.alpha)
    }

    pub fn active_domain(self) -> IndexDomain {
        IndexDomain {
            start: self.a,
            end: self.b,
        }
    }

    pub fn trivial_domain(self) -> TrivialDomain {
        let next_prime_lower_bound = self.y.saturating_add(1);
        let cube = next_prime_lower_bound
            .checked_mul(next_prime_lower_bound)
            .and_then(|sq| sq.checked_mul(next_prime_lower_bound));

        TrivialDomain {
            s3_is_zero: cube.is_none_or(|v| v > self.x),
        }
    }

    pub fn dr_domains_with_starts(self, hard_start: usize, easy_start: usize) -> DrDomains {
        self.dr_domains_from_frontiers(hard_start, easy_start, self.frontier_set())
    }

    pub fn dr_domains_from_frontiers(
        self,
        hard_start: usize,
        easy_start: usize,
        frontiers: FrontierSet,
    ) -> DrDomains {
        let active = self.active_domain();
        let hard_start = hard_start.clamp(active.start, active.end);
        let easy_start = easy_start.clamp(active.start, active.end);
        let ordinary = IndexDomain {
            start: active.start,
            end: hard_start.min(easy_start),
        };
        let hard = IndexDomain {
            start: hard_start.min(easy_start),
            end: easy_start,
        };
        let easy = IndexDomain {
            start: easy_start,
            end: active.end,
        };

        DrDomains {
            active,
            s1: S1Domain {
                leaves: ordinary,
                rule: frontiers.s1,
            },
            s2_trivial: S2TrivialDomain {
                leaves: self.trivial_domain(),
            },
            s2_easy: S2EasyDomain {
                leaves: easy,
                rule: frontiers.s2_easy,
            },
            s2_hard: S2HardDomain {
                leaves: hard,
                rule: frontiers.s2_hard,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DrTuning, EasyRelativeToHardCandidate, EasyTermBandCandidate, HardTermBandCandidate,
        Parameters, PhaseCBoundaryCandidate, PhaseCBoundaryQuotientGuardCandidate,
        PhaseCBoundaryRelativeQuotientBandCandidate,
        PhaseCBoundaryRelativeQuotientStepBandCandidate,
        PhaseCBoundaryRelativeQuotientStepBridgeCandidate, PhaseCBufferedBoundaryCandidate,
        PhaseCLinkedCandidate, PhaseCQuotientWindowCandidate, S2EasyRule,
    };

    #[test]
    fn candidate_easy_frontier_set_uses_reference_width_two() {
        let params = Parameters::from_counts(20_000_000, 58, 607).with_hard_leaf_term_max(3);
        let frontiers = params.candidate_easy_frontier_set();

        assert_eq!(DrTuning::CANDIDATE_EASY_RELATIVE_TO_HARD_WIDTH, 2);
        assert_eq!(DrTuning::CANDIDATE_EASY_RELATIVE_TO_HARD_MIN_TERM_FLOOR, 1);
        assert_eq!(frontiers.s2_easy.kind_name(), "relative_to_hard_with_floor");
        assert_eq!(frontiers.s2_easy.term_bounds(), (1, 2));
        assert_eq!(frontiers.s2_hard.term_bounds(), (3, 3));
    }

    #[test]
    fn relative_to_hard_with_floor_preserves_current_candidate_bounds() {
        let rule = S2EasyRule::alpha_balanced_relative_to_hard_with_floor(3, 2, 1, 1.0);

        assert_eq!(rule.kind_name(), "relative_to_hard_with_floor");
        assert_eq!(rule.term_bounds(), (1, 2));
        assert!(rule.matches(1));
        assert!(rule.matches(2));
        assert!(!rule.matches(3));
    }

    #[test]
    fn relative_to_hard_with_floor_clamps_to_feasible_range() {
        let rule = S2EasyRule::alpha_balanced_relative_to_hard_with_floor(3, 2, 5, 1.0);

        assert_eq!(rule.term_bounds(), (2, 2));
        assert!(!rule.matches(1));
        assert!(rule.matches(2));
    }

    #[test]
    fn easy_relative_to_hard_candidate_builds_the_expected_rule() {
        let candidate = EasyRelativeToHardCandidate {
            width: 2,
            min_term_floor: 1,
        };
        let rule = candidate.build_rule(3, 1.0);

        assert_eq!(rule.kind_name(), "relative_to_hard_with_floor");
        assert_eq!(rule.term_bounds(), (1, 2));
    }

    #[test]
    fn candidate_easy_term_band_frontier_set_uses_reference_band() {
        let params = Parameters::from_counts(20_000_000, 58, 607).with_hard_leaf_term_max(3);
        let frontiers = params.candidate_easy_term_band_frontier_set();

        assert_eq!(DrTuning::CANDIDATE_EASY_TERM_BAND_MIN, 1);
        assert_eq!(DrTuning::CANDIDATE_EASY_TERM_BAND_MAX, 2);
        assert_eq!(frontiers.s2_easy.kind_name(), "alpha_balanced_term_range");
        assert_eq!(frontiers.s2_easy.term_bounds(), (1, 2));
        assert_eq!(frontiers.s2_hard.term_bounds(), (3, 3));
    }

    #[test]
    fn easy_term_band_candidate_builds_the_expected_rule() {
        let candidate = EasyTermBandCandidate {
            min_term: 1,
            max_term: 2,
        };
        let rule = candidate.build_rule(1.0);

        assert_eq!(rule.kind_name(), "alpha_balanced_term_range");
        assert_eq!(rule.term_bounds(), (1, 2));
        assert!(rule.matches(1));
        assert!(rule.matches(2));
        assert!(!rule.matches(3));
    }

    #[test]
    fn phase_c_easy_term_band_frontier_set_uses_reference_band() {
        let params = Parameters::from_counts(20_000_000, 58, 607).with_hard_leaf_term_max(5);
        let frontiers = params.phase_c_easy_term_band_frontier_set();

        assert_eq!(DrTuning::PHASE_C_EASY_TERM_BAND_MIN, 1);
        assert_eq!(DrTuning::PHASE_C_EASY_TERM_BAND_MAX, 4);
        assert_eq!(frontiers.s2_easy.kind_name(), "alpha_balanced_term_range");
        assert_eq!(frontiers.s2_easy.term_bounds(), (1, 4));
        assert_eq!(frontiers.s2_hard.term_bounds(), (5, 5));
    }

    #[test]
    fn hard_term_band_candidate_builds_the_expected_rule() {
        let candidate = HardTermBandCandidate {
            min_term: 5,
            max_term: 6,
        };
        let rule = candidate.build_rule(1.0);

        assert_eq!(rule.kind_name(), "alpha_balanced_term_range");
        assert_eq!(rule.term_bounds(), (5, 6));
        assert!(rule.matches(5));
        assert!(rule.matches(6));
        assert!(!rule.matches(4));
    }

    #[test]
    fn phase_c_hard_term_band_frontier_set_uses_reference_band() {
        let params = Parameters::from_counts(20_000_000, 58, 607).with_hard_leaf_term_max(6);
        let frontiers = params.phase_c_hard_term_band_frontier_set();

        assert_eq!(DrTuning::PHASE_C_HARD_TERM_BAND_MIN, 5);
        assert_eq!(DrTuning::PHASE_C_HARD_TERM_BAND_MAX, 6);
        assert_eq!(frontiers.s2_easy.term_bounds(), (1, 4));
        assert_eq!(frontiers.s2_hard.term_bounds(), (5, 6));
        assert_eq!(frontiers.s1.term_min_exclusive, 6);
    }

    #[test]
    fn phase_c_linked_candidate_builds_the_expected_frontiers() {
        let candidate = PhaseCLinkedCandidate {
            easy_width: 5,
            easy_min_term_floor: 1,
            hard_width: 1,
        };
        let frontiers = candidate.build_frontier_set(1.0);

        assert_eq!(frontiers.s2_easy.kind_name(), "relative_to_hard_with_floor");
        assert_eq!(frontiers.s2_easy.term_bounds(), (1, 5));
        assert_eq!(frontiers.s2_hard.kind_name(), "relative_to_easy");
        assert_eq!(frontiers.s2_hard.term_bounds(), (6, 6));
        assert_eq!(frontiers.s1.term_min_exclusive, 6);
    }

    #[test]
    fn phase_c_linked_frontier_set_matches_promoted_reference() {
        let params = Parameters::from_counts(20_000_000, 58, 607).with_hard_leaf_term_max(6);
        let linked = params.phase_c_linked_frontier_set();
        let promoted = params.phase_c_frontier_set();
        let band = params.phase_c_term_band_frontier_set();

        assert_eq!(linked.s2_easy.term_bounds(), promoted.s2_easy.term_bounds());
        assert_eq!(linked.s2_hard.term_bounds(), promoted.s2_hard.term_bounds());
        assert_eq!(linked.s1.term_min_exclusive, promoted.s1.term_min_exclusive);
        assert_ne!(linked.s2_easy.term_bounds(), band.s2_easy.term_bounds());
    }

    #[test]
    fn phase_c_boundary_candidate_builds_the_expected_frontiers() {
        let candidate = PhaseCBoundaryCandidate {
            boundary_term: 5,
            easy_width: 5,
            hard_width: 1,
        };
        let frontiers = candidate.build_frontier_set(1.0);

        assert_eq!(frontiers.s2_easy.kind_name(), "alpha_balanced_term_range");
        assert_eq!(frontiers.s2_easy.term_bounds(), (1, 5));
        assert_eq!(frontiers.s2_hard.kind_name(), "alpha_balanced_term_range");
        assert_eq!(frontiers.s2_hard.term_bounds(), (6, 6));
        assert_eq!(frontiers.s1.term_min_exclusive, 6);
    }

    #[test]
    fn phase_c_boundary_frontier_set_matches_promoted_reference() {
        let params = Parameters::from_counts(20_000_000, 58, 607).with_hard_leaf_term_max(6);
        let boundary = params.phase_c_boundary_frontier_set();
        let promoted = params.phase_c_frontier_set();

        assert_eq!(
            boundary.s2_easy.term_bounds(),
            promoted.s2_easy.term_bounds()
        );
        assert_eq!(
            boundary.s2_hard.term_bounds(),
            promoted.s2_hard.term_bounds()
        );
        assert_eq!(
            boundary.s1.term_min_exclusive,
            promoted.s1.term_min_exclusive
        );
    }

    #[test]
    fn divergent_phase_c_boundary_candidate_builds_distinct_frontiers() {
        let candidate = PhaseCBoundaryCandidate {
            boundary_term: 4,
            easy_width: 4,
            hard_width: 2,
        };
        let frontiers = candidate.build_frontier_set(1.0);

        assert_eq!(frontiers.s2_easy.term_bounds(), (1, 4));
        assert_eq!(frontiers.s2_hard.term_bounds(), (5, 6));
        assert_eq!(frontiers.s1.term_min_exclusive, 6);
    }

    #[test]
    fn phase_c_buffered_boundary_candidate_builds_the_expected_frontiers() {
        let candidate = PhaseCBufferedBoundaryCandidate {
            boundary_term: 4,
            easy_width: 4,
            gap_width: 1,
            hard_width: 1,
        };
        let frontiers = candidate.build_frontier_set(1.0);

        assert_eq!(frontiers.s2_easy.term_bounds(), (1, 4));
        assert_eq!(frontiers.s2_hard.term_bounds(), (6, 6));
        assert_eq!(frontiers.s1.term_min_exclusive, 6);
    }

    #[test]
    fn phase_c_quotient_window_candidate_builds_the_expected_parameters() {
        let candidate = PhaseCQuotientWindowCandidate {
            easy_q_offset_max: 0,
            hard_q_width: 1,
        };

        assert_eq!(candidate.easy_q_offset_max, 0);
        assert_eq!(candidate.hard_q_width, 1);
    }

    #[test]
    fn phase_c_boundary_quotient_guard_candidate_builds_the_expected_parameters() {
        let candidate = PhaseCBoundaryQuotientGuardCandidate {
            boundary_term: 4,
            easy_width: 4,
            hard_width: 2,
            guard_q_offset: 0,
        };

        assert_eq!(candidate.boundary_term, 4);
        assert_eq!(candidate.easy_width, 4);
        assert_eq!(candidate.hard_width, 2);
        assert_eq!(candidate.guard_q_offset, 0);
    }

    #[test]
    fn phase_c_boundary_relative_quotient_band_candidate_builds_the_expected_parameters() {
        let candidate = PhaseCBoundaryRelativeQuotientBandCandidate {
            boundary_term: 4,
            easy_width: 4,
            hard_width: 2,
            easy_q_band_width: 0,
            hard_q_band_width: 0,
        };

        assert_eq!(candidate.boundary_term, 4);
        assert_eq!(candidate.easy_width, 4);
        assert_eq!(candidate.hard_width, 2);
        assert_eq!(candidate.easy_q_band_width, 0);
        assert_eq!(candidate.hard_q_band_width, 0);
    }

    #[test]
    fn phase_c_boundary_relative_quotient_step_band_candidate_builds_the_expected_parameters() {
        let candidate = PhaseCBoundaryRelativeQuotientStepBandCandidate {
            boundary_term: 4,
            easy_width: 4,
            hard_width: 2,
            easy_q_step_multiplier: 0,
            hard_q_step_multiplier: 0,
        };

        assert_eq!(candidate.boundary_term, 4);
        assert_eq!(candidate.easy_width, 4);
        assert_eq!(candidate.hard_width, 2);
        assert_eq!(candidate.easy_q_step_multiplier, 0);
        assert_eq!(candidate.hard_q_step_multiplier, 0);
    }

    #[test]
    fn phase_c_boundary_relative_quotient_step_bridge_candidate_builds_the_expected_parameters() {
        let candidate = PhaseCBoundaryRelativeQuotientStepBridgeCandidate {
            boundary_term: 4,
            easy_width: 4,
            hard_width: 2,
            easy_q_step_multiplier: 1,
            hard_q_step_multiplier: 1,
            bridge_width: 1,
        };

        assert_eq!(candidate.boundary_term, 4);
        assert_eq!(candidate.easy_width, 4);
        assert_eq!(candidate.hard_width, 2);
        assert_eq!(candidate.easy_q_step_multiplier, 1);
        assert_eq!(candidate.hard_q_step_multiplier, 1);
        assert_eq!(candidate.bridge_width, 1);
    }
}
