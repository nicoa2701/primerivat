pub mod easy;
pub mod hard;
pub mod ordinary;
pub mod trivial;
pub mod types;

use crate::math::isqrt;
use crate::parameters::{DrTuning, Parameters};
use crate::phi::{PhiBackend, default_phi_computation, phi_computation_with_backend};
use crate::primes::PrimeTable;
use crate::segment::primes_up_to;
use std::time::{Duration, Instant};
use types::{DomainSet, DrContext};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DrContributions {
    pub trivial: u128,
    pub easy: u128,
    pub ordinary: u128,
    pub hard: u128,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RulePreview {
    pub kind: &'static str,
    pub term_min: u128,
    pub term_max: u128,
    pub alpha: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrAnalysis {
    pub alpha: f64,
    pub phi_x_a: u128,
    pub a: usize,
    pub easy_candidate_family: Option<&'static str>,
    pub easy_candidate_width: Option<u128>,
    pub easy_candidate_floor: Option<u128>,
    pub easy_candidate_term_min: Option<u128>,
    pub easy_candidate_term_max: Option<u128>,
    pub s1_term_min_exclusive: u128,
    pub hard_leaf_term_max: u128,
    pub hard_leaf_term_min: u128,
    pub hard_rule_kind: &'static str,
    pub hard_rule_alpha: Option<f64>,
    pub easy_leaf_term_value: u128,
    pub easy_leaf_term_min: u128,
    pub easy_leaf_term_max: u128,
    pub easy_rule_kind: &'static str,
    pub easy_rule_alpha: Option<f64>,
    pub active_len: usize,
    pub s1_len: usize,
    pub s2_easy_len: usize,
    pub s2_hard_len: usize,
    pub s2_trivial_is_zero: bool,
    pub contributions: DrContributions,
    pub result: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EasySpecializedAnalysis {
    pub easy_len: usize,
    pub residual_len: usize,
    pub transition_len: usize,
    pub specialized_len: usize,
    pub easy_sum: u128,
    pub residual_sum: u128,
    pub transition_sum: u128,
    pub specialized_sum: u128,
    pub q_ref: Option<u128>,
    pub q_step: u128,
    pub first_specialized_q: Option<u128>,
    pub last_specialized_q: Option<u128>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardSpecializedAnalysis {
    pub hard_len: usize,
    pub residual_len: usize,
    pub transition_len: usize,
    pub specialized_len: usize,
    pub hard_sum: u128,
    pub residual_sum: u128,
    pub transition_sum: u128,
    pub specialized_sum: u128,
    pub q_ref: Option<u128>,
    pub q_step: u128,
    pub first_specialized_q: Option<u128>,
    pub last_specialized_q: Option<u128>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinarySpecializedAnalysis {
    pub ordinary_len: usize,
    pub residual_len: usize,
    pub pretransition_len: usize,
    pub transition_len: usize,
    pub specialized_len: usize,
    pub ordinary_sum: u128,
    pub residual_sum: u128,
    pub pretransition_sum: u128,
    pub transition_sum: u128,
    pub specialized_sum: u128,
    pub q_ref: Option<u128>,
    pub q_step: u128,
    pub first_specialized_q: Option<u128>,
    pub last_specialized_q: Option<u128>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryRelativeQuotientAnalysis {
    pub ordinary_len: usize,
    pub left_residual_len: usize,
    pub region_len: usize,
    pub right_residual_len: usize,
    pub ordinary_sum: u128,
    pub left_residual_sum: u128,
    pub region_sum: u128,
    pub right_residual_sum: u128,
    pub q_ref: Option<u128>,
    pub q_step: u128,
    pub first_region_q: Option<u128>,
    pub last_region_q: Option<u128>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryRelativeQuotientShoulderAnalysis {
    pub ordinary_len: usize,
    pub left_residual_len: usize,
    pub left_shoulder_len: usize,
    pub core_len: usize,
    pub right_shoulder_len: usize,
    pub right_residual_len: usize,
    pub ordinary_sum: u128,
    pub left_residual_sum: u128,
    pub left_shoulder_sum: u128,
    pub core_sum: u128,
    pub right_shoulder_sum: u128,
    pub right_residual_sum: u128,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryRelativeQuotientEnvelopeAnalysis {
    pub ordinary_len: usize,
    pub left_residual_len: usize,
    pub left_envelope_len: usize,
    pub core_len: usize,
    pub right_envelope_len: usize,
    pub right_residual_len: usize,
    pub ordinary_sum: u128,
    pub left_residual_sum: u128,
    pub left_envelope_sum: u128,
    pub core_sum: u128,
    pub right_envelope_sum: u128,
    pub right_residual_sum: u128,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryRelativeQuotientHierarchyAnalysis {
    pub ordinary_len: usize,
    pub left_residual_len: usize,
    pub left_outer_band_len: usize,
    pub left_near_band_len: usize,
    pub inner_core_len: usize,
    pub right_near_band_len: usize,
    pub right_outer_band_len: usize,
    pub right_residual_len: usize,
    pub ordinary_sum: u128,
    pub left_residual_sum: u128,
    pub left_outer_band_sum: u128,
    pub left_near_band_sum: u128,
    pub inner_core_sum: u128,
    pub right_near_band_sum: u128,
    pub right_outer_band_sum: u128,
    pub right_residual_sum: u128,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryRegionAssemblyAnalysis {
    pub ordinary_len: usize,
    pub left_outer_support_len: usize,
    pub left_adjacent_support_len: usize,
    pub central_assembly_len: usize,
    pub right_adjacent_support_len: usize,
    pub right_outer_support_len: usize,
    pub ordinary_sum: u128,
    pub left_outer_support_sum: u128,
    pub left_adjacent_support_sum: u128,
    pub central_assembly_sum: u128,
    pub right_adjacent_support_sum: u128,
    pub right_outer_support_sum: u128,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryQuasiLiteratureAnalysis {
    pub ordinary_len: usize,
    pub left_outer_work_len: usize,
    pub middle_work_len: usize,
    pub right_outer_work_len: usize,
    pub ordinary_sum: u128,
    pub left_outer_work_sum: u128,
    pub middle_work_sum: u128,
    pub right_outer_work_sum: u128,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryDrLikeAnalysis {
    pub ordinary_len: usize,
    pub left_outer_work_len: usize,
    pub left_transfer_work_len: usize,
    pub central_work_region_len: usize,
    pub right_transfer_work_len: usize,
    pub right_outer_work_len: usize,
    pub ordinary_sum: u128,
    pub left_outer_work_sum: u128,
    pub left_transfer_work_sum: u128,
    pub central_work_region_sum: u128,
    pub right_transfer_work_sum: u128,
    pub right_outer_work_sum: u128,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryRegionComparison {
    pub ordinary_len: usize,
    pub current_terminal_len: usize,
    pub relative_region_len: usize,
    pub current_terminal_sum: u128,
    pub relative_region_sum: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryPostPlateauProfile {
    pub ordinary_len: usize,
    pub terminal_len: usize,
    pub terminal_sum: u128,
    pub left_residual_len: usize,
    pub region_len: usize,
    pub right_residual_len: usize,
    pub left_residual_sum: u128,
    pub region_sum: u128,
    pub right_residual_sum: u128,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostPlateauTriptychAnalysis {
    pub easy_len: usize,
    pub easy_focus_len: usize,
    pub easy_focus_sum: u128,
    pub ordinary_len: usize,
    pub ordinary_terminal_len: usize,
    pub ordinary_terminal_sum: u128,
    pub ordinary_region_len: usize,
    pub ordinary_region_sum: u128,
    pub ordinary_assembly_core_len: usize,
    pub ordinary_assembly_core_sum: u128,
    pub ordinary_assembly_support_len: usize,
    pub ordinary_assembly_support_sum: u128,
    pub ordinary_quasi_literature_middle_len: usize,
    pub ordinary_quasi_literature_middle_sum: u128,
    pub ordinary_quasi_literature_outer_len: usize,
    pub ordinary_quasi_literature_outer_sum: u128,
    pub hard_len: usize,
    pub hard_focus_len: usize,
    pub hard_focus_sum: u128,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalysisComparison {
    pub current: DrAnalysis,
    pub experimental: DrAnalysis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrRuntimeProfile {
    pub result: u128,
    pub phi_x_a: u128,
    pub a: usize,
    pub phi_time: Duration,
    pub seed_primes_time: Duration,
    pub sqrt_primes_time: Duration,
    pub s2_time: Duration,
    pub total_time: Duration,
}

pub fn prepare_context(x: u128) -> DrContext<'static> {
    prepare_context_with_term_frontiers(
        x,
        Parameters::DEFAULT_HARD_LEAF_TERM_MAX,
        Parameters::DEFAULT_EASY_LEAF_TERM_VALUE,
    )
}

pub fn prepare_context_with_hard_leaf_term_max(
    x: u128,
    hard_leaf_term_max: u128,
) -> DrContext<'static> {
    prepare_context_with_term_frontiers(
        x,
        hard_leaf_term_max,
        Parameters::DEFAULT_EASY_LEAF_TERM_VALUE,
    )
}

pub fn prepare_context_with_term_frontiers(
    x: u128,
    hard_leaf_term_max: u128,
    easy_leaf_term_max: u128,
) -> DrContext<'static> {
    let computation = default_phi_computation(x);
    let params = Parameters::from_tables(x, &computation.small, &computation.large)
        .with_hard_leaf_term_max(hard_leaf_term_max)
        .with_easy_leaf_term_max(easy_leaf_term_max);
    let primes = PrimeTable::new(computation.primes);

    let small = Box::leak(computation.small.into_boxed_slice());
    let large = Box::leak(computation.large.into_boxed_slice());
    let primes = Box::leak(Box::new(primes));

    DrContext::new(params, computation.phi_x_a, primes, small, large)
}

pub fn skeleton_contributions(ctx: &DrContext<'_>) -> DrContributions {
    ctx.domain_set().contributions
}

impl DrContributions {
    pub fn total(self) -> u128 {
        self.trivial + self.easy + self.ordinary + self.hard
    }
}

pub fn preview_inactive_easy_variant(hard_term_min: u128, width: u128) -> RulePreview {
    let rule = crate::parameters::S2EasyRule::alpha_balanced_relative_to_hard(
        hard_term_min,
        width,
        crate::parameters::DrTuning::DEFAULT_ALPHA,
    );
    let (term_min, term_max) = rule.term_bounds();

    RulePreview {
        kind: rule.kind_name(),
        term_min,
        term_max,
        alpha: rule.alpha(),
    }
}

pub fn preview_inactive_hard_variant(easy_term_max: u128, width: u128) -> RulePreview {
    let rule = crate::parameters::S2HardRule::alpha_balanced_relative_to_easy(
        easy_term_max,
        width,
        crate::parameters::DrTuning::DEFAULT_ALPHA,
    );
    let (term_min, term_max) = rule.term_bounds();

    RulePreview {
        kind: rule.kind_name(),
        term_min,
        term_max,
        alpha: rule.alpha(),
    }
}

pub fn analyze_with_experimental_easy_relative_to_hard(
    x: u128,
    hard_leaf_term_max: u128,
    easy_width: u128,
) -> DrAnalysis {
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let easy_rule = crate::parameters::S2EasyRule::alpha_balanced_relative_to_hard(
        hard_leaf_term_max,
        easy_width,
        ctx.params.tuning.alpha,
    );
    let (_, easy_leaf_term_max) = easy_rule.term_bounds();
    let hard_rule = crate::parameters::S2HardRule::alpha_balanced_term_range(
        easy_leaf_term_max.saturating_add(1),
        hard_leaf_term_max,
        ctx.params.tuning.alpha,
    );
    let frontiers = crate::parameters::FrontierSet {
        s1: crate::parameters::S1Rule {
            term_min_exclusive: hard_leaf_term_max,
        },
        s2_hard: hard_rule,
        s2_easy: easy_rule,
    };
    let domain_set = ctx.domain_set_with_frontiers(frontiers);
    let domains = domain_set.domains;
    let contributions = domain_set.contributions;
    let result = domain_set.result;
    let (easy_leaf_term_min, easy_leaf_term_max) = easy_rule.term_bounds();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: None,
        easy_candidate_width: None,
        easy_candidate_floor: None,
        easy_candidate_term_min: None,
        easy_candidate_term_max: None,
        s1_term_min_exclusive: frontiers.s1.term_min_exclusive,
        hard_leaf_term_max: hard_rule.term_max(),
        hard_leaf_term_min: hard_rule.term_min(),
        hard_rule_kind: hard_rule.kind_name(),
        hard_rule_alpha: hard_rule.alpha(),
        easy_leaf_term_value: easy_rule.representative_term_value(),
        easy_leaf_term_min,
        easy_leaf_term_max,
        easy_rule_kind: easy_rule.kind_name(),
        easy_rule_alpha: easy_rule.alpha(),
        active_len: domains.active.len(),
        s1_len: domains.s1.leaves.len(),
        s2_easy_len: domains.s2_easy.leaves.len(),
        s2_hard_len: domains.s2_hard.leaves.len(),
        s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
        contributions,
        result,
    }
}

pub fn analyze_with_experimental_hard_relative_to_easy(
    x: u128,
    easy_leaf_term_max: u128,
    hard_width: u128,
) -> DrAnalysis {
    let hard_rule = crate::parameters::S2HardRule::alpha_balanced_relative_to_easy(
        easy_leaf_term_max,
        hard_width,
        crate::parameters::DrTuning::DEFAULT_ALPHA,
    );
    let hard_leaf_term_max = hard_rule.term_max();
    let ctx = prepare_context_with_term_frontiers(x, hard_leaf_term_max, easy_leaf_term_max);
    let easy_rule = crate::parameters::S2EasyRule::alpha_balanced_term_range(
        crate::parameters::Parameters::DEFAULT_EASY_LEAF_TERM_VALUE,
        easy_leaf_term_max,
        ctx.params.tuning.alpha,
    );
    let frontiers = crate::parameters::FrontierSet {
        s1: crate::parameters::S1Rule {
            term_min_exclusive: hard_leaf_term_max,
        },
        s2_hard: hard_rule,
        s2_easy: easy_rule,
    };
    let domain_set = ctx.domain_set_with_frontiers(frontiers);
    let domains = domain_set.domains;
    let contributions = domain_set.contributions;
    let result = domain_set.result;
    let (easy_leaf_term_min, easy_leaf_term_max) = easy_rule.term_bounds();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: None,
        easy_candidate_width: None,
        easy_candidate_floor: None,
        easy_candidate_term_min: None,
        easy_candidate_term_max: None,
        s1_term_min_exclusive: frontiers.s1.term_min_exclusive,
        hard_leaf_term_max: hard_rule.term_max(),
        hard_leaf_term_min: hard_rule.term_min(),
        hard_rule_kind: hard_rule.kind_name(),
        hard_rule_alpha: hard_rule.alpha(),
        easy_leaf_term_value: easy_rule.representative_term_value(),
        easy_leaf_term_min,
        easy_leaf_term_max,
        easy_rule_kind: easy_rule.kind_name(),
        easy_rule_alpha: easy_rule.alpha(),
        active_len: domains.active.len(),
        s1_len: domains.s1.leaves.len(),
        s2_easy_len: domains.s2_easy.leaves.len(),
        s2_hard_len: domains.s2_hard.leaves.len(),
        s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
        contributions,
        result,
    }
}

pub fn compare_current_vs_experimental_easy_relative_to_hard(
    x: u128,
    hard_leaf_term_max: u128,
    easy_leaf_term_max: u128,
    easy_width: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_term_frontiers(x, hard_leaf_term_max, easy_leaf_term_max),
        experimental: analyze_with_experimental_easy_relative_to_hard(
            x,
            hard_leaf_term_max,
            easy_width,
        ),
    }
}

pub fn compare_current_vs_experimental_hard_relative_to_easy(
    x: u128,
    hard_leaf_term_max: u128,
    easy_leaf_term_max: u128,
    hard_width: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_term_frontiers(x, hard_leaf_term_max, easy_leaf_term_max),
        experimental: analyze_with_experimental_hard_relative_to_easy(
            x,
            easy_leaf_term_max,
            hard_width,
        ),
    }
}

pub fn analyze_with_candidate_easy_relative_to_hard(
    x: u128,
    hard_leaf_term_max: u128,
) -> DrAnalysis {
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let candidate = DrTuning::candidate_easy_relative_to_hard();
    let frontiers = ctx.params.candidate_easy_frontier_set();
    let domain_set = ctx.domain_set_with_frontiers(frontiers);
    let domains = domain_set.domains;
    let contributions = domain_set.contributions;
    let result = domain_set.result;
    let easy_rule = frontiers.s2_easy;
    let hard_rule = frontiers.s2_hard;
    let (easy_leaf_term_min, easy_leaf_term_max) = easy_rule.term_bounds();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: Some("relative_to_hard"),
        easy_candidate_width: Some(candidate.width),
        easy_candidate_floor: Some(candidate.min_term_floor),
        easy_candidate_term_min: None,
        easy_candidate_term_max: None,
        s1_term_min_exclusive: frontiers.s1.term_min_exclusive,
        hard_leaf_term_max: hard_rule.term_max(),
        hard_leaf_term_min: hard_rule.term_min(),
        hard_rule_kind: hard_rule.kind_name(),
        hard_rule_alpha: hard_rule.alpha(),
        easy_leaf_term_value: easy_rule.representative_term_value(),
        easy_leaf_term_min,
        easy_leaf_term_max,
        easy_rule_kind: easy_rule.kind_name(),
        easy_rule_alpha: easy_rule.alpha(),
        active_len: domains.active.len(),
        s1_len: domains.s1.leaves.len(),
        s2_easy_len: domains.s2_easy.leaves.len(),
        s2_hard_len: domains.s2_hard.leaves.len(),
        s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
        contributions,
        result,
    }
}

pub fn analyze_with_candidate_easy_term_band(x: u128, hard_leaf_term_max: u128) -> DrAnalysis {
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let candidate = DrTuning::candidate_easy_term_band();
    let frontiers = ctx.params.candidate_easy_term_band_frontier_set();
    let domain_set = ctx.domain_set_with_frontiers(frontiers);
    let domains = domain_set.domains;
    let contributions = domain_set.contributions;
    let result = domain_set.result;
    let easy_rule = frontiers.s2_easy;
    let hard_rule = frontiers.s2_hard;
    let (easy_leaf_term_min, easy_leaf_term_max) = easy_rule.term_bounds();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: Some("term_band"),
        easy_candidate_width: None,
        easy_candidate_floor: None,
        easy_candidate_term_min: Some(candidate.min_term),
        easy_candidate_term_max: Some(candidate.max_term),
        s1_term_min_exclusive: frontiers.s1.term_min_exclusive,
        hard_leaf_term_max: hard_rule.term_max(),
        hard_leaf_term_min: hard_rule.term_min(),
        hard_rule_kind: hard_rule.kind_name(),
        hard_rule_alpha: hard_rule.alpha(),
        easy_leaf_term_value: easy_rule.representative_term_value(),
        easy_leaf_term_min,
        easy_leaf_term_max,
        easy_rule_kind: easy_rule.kind_name(),
        easy_rule_alpha: easy_rule.alpha(),
        active_len: domains.active.len(),
        s1_len: domains.s1.leaves.len(),
        s2_easy_len: domains.s2_easy.leaves.len(),
        s2_hard_len: domains.s2_hard.leaves.len(),
        s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
        contributions,
        result,
    }
}

pub fn analyze_with_phase_c_easy_term_band(x: u128, hard_leaf_term_max: u128) -> DrAnalysis {
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let candidate = DrTuning::phase_c_easy_term_band();
    let frontiers = ctx.params.phase_c_easy_term_band_frontier_set();
    let domain_set = ctx.domain_set_with_frontiers(frontiers);
    let domains = domain_set.domains;
    let contributions = domain_set.contributions;
    let result = domain_set.result;
    let easy_rule = frontiers.s2_easy;
    let hard_rule = frontiers.s2_hard;
    let (easy_leaf_term_min, easy_leaf_term_max) = easy_rule.term_bounds();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: Some("phase_c_term_band"),
        easy_candidate_width: None,
        easy_candidate_floor: None,
        easy_candidate_term_min: Some(candidate.min_term),
        easy_candidate_term_max: Some(candidate.max_term),
        s1_term_min_exclusive: frontiers.s1.term_min_exclusive,
        hard_leaf_term_max: hard_rule.term_max(),
        hard_leaf_term_min: hard_rule.term_min(),
        hard_rule_kind: hard_rule.kind_name(),
        hard_rule_alpha: hard_rule.alpha(),
        easy_leaf_term_value: easy_rule.representative_term_value(),
        easy_leaf_term_min,
        easy_leaf_term_max,
        easy_rule_kind: easy_rule.kind_name(),
        easy_rule_alpha: easy_rule.alpha(),
        active_len: domains.active.len(),
        s1_len: domains.s1.leaves.len(),
        s2_easy_len: domains.s2_easy.leaves.len(),
        s2_hard_len: domains.s2_hard.leaves.len(),
        s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
        contributions,
        result,
    }
}

pub fn analyze_with_phase_c_hard_term_band(x: u128) -> DrAnalysis {
    let hard_leaf_term_max = DrTuning::PHASE_C_HARD_TERM_BAND_MAX;
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let hard_candidate = DrTuning::phase_c_hard_term_band();
    let easy_candidate = DrTuning::phase_c_easy_term_band();
    let easy_rule = easy_candidate.build_rule(ctx.params.tuning.alpha);
    let hard_rule = hard_candidate.build_rule(ctx.params.tuning.alpha);
    let frontiers = crate::parameters::FrontierSet {
        s1: crate::parameters::S1Rule {
            term_min_exclusive: hard_candidate.max_term,
        },
        s2_hard: hard_rule,
        s2_easy: easy_rule,
    };
    let domain_set = ctx.domain_set_with_frontiers(frontiers);
    let domains = domain_set.domains;
    let contributions = domain_set.contributions;
    let result = domain_set.result;
    let (easy_leaf_term_min, easy_leaf_term_max) = easy_rule.term_bounds();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: Some("phase_c_hard_term_band"),
        easy_candidate_width: None,
        easy_candidate_floor: None,
        easy_candidate_term_min: Some(hard_candidate.min_term),
        easy_candidate_term_max: Some(hard_candidate.max_term),
        s1_term_min_exclusive: frontiers.s1.term_min_exclusive,
        hard_leaf_term_max: hard_rule.term_max(),
        hard_leaf_term_min: hard_rule.term_min(),
        hard_rule_kind: hard_rule.kind_name(),
        hard_rule_alpha: hard_rule.alpha(),
        easy_leaf_term_value: easy_rule.representative_term_value(),
        easy_leaf_term_min,
        easy_leaf_term_max,
        easy_rule_kind: easy_rule.kind_name(),
        easy_rule_alpha: easy_rule.alpha(),
        active_len: domains.active.len(),
        s1_len: domains.s1.leaves.len(),
        s2_easy_len: domains.s2_easy.leaves.len(),
        s2_hard_len: domains.s2_hard.leaves.len(),
        s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
        contributions,
        result,
    }
}

pub fn analyze_with_phase_c_term_band_package(x: u128) -> DrAnalysis {
    let hard_leaf_term_max = DrTuning::PHASE_C_HARD_TERM_BAND_MAX;
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let easy_candidate = DrTuning::phase_c_easy_term_band();
    let hard_candidate = DrTuning::phase_c_hard_term_band();
    let frontiers = ctx.params.phase_c_term_band_frontier_set();
    let domain_set = ctx.domain_set_with_frontiers(frontiers);
    let domains = domain_set.domains;
    let contributions = domain_set.contributions;
    let result = domain_set.result;
    let easy_rule = frontiers.s2_easy;
    let hard_rule = frontiers.s2_hard;
    let (easy_leaf_term_min, easy_leaf_term_max) = easy_rule.term_bounds();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: Some("phase_c_term_band_package"),
        easy_candidate_width: None,
        easy_candidate_floor: None,
        easy_candidate_term_min: Some(easy_candidate.min_term),
        easy_candidate_term_max: Some(hard_candidate.max_term),
        s1_term_min_exclusive: frontiers.s1.term_min_exclusive,
        hard_leaf_term_max: hard_rule.term_max(),
        hard_leaf_term_min: hard_rule.term_min(),
        hard_rule_kind: hard_rule.kind_name(),
        hard_rule_alpha: hard_rule.alpha(),
        easy_leaf_term_value: easy_rule.representative_term_value(),
        easy_leaf_term_min,
        easy_leaf_term_max,
        easy_rule_kind: easy_rule.kind_name(),
        easy_rule_alpha: easy_rule.alpha(),
        active_len: domains.active.len(),
        s1_len: domains.s1.leaves.len(),
        s2_easy_len: domains.s2_easy.leaves.len(),
        s2_hard_len: domains.s2_hard.leaves.len(),
        s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
        contributions,
        result,
    }
}

pub fn analyze_with_phase_c_package(x: u128) -> DrAnalysis {
    let hard_leaf_term_max = DrTuning::PHASE_C_LINKED_EASY_MIN_TERM_FLOOR
        .saturating_add(DrTuning::PHASE_C_LINKED_EASY_WIDTH - 1)
        .saturating_add(DrTuning::PHASE_C_LINKED_HARD_WIDTH);
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let candidate = DrTuning::phase_c_linked_candidate();
    let frontiers = ctx.params.phase_c_frontier_set();
    let domain_set = ctx.domain_set_with_frontiers(frontiers);
    let domains = domain_set.domains;
    let contributions = domain_set.contributions;
    let result = domain_set.result;
    let easy_rule = frontiers.s2_easy;
    let hard_rule = frontiers.s2_hard;
    let (easy_leaf_term_min, easy_leaf_term_max) = easy_rule.term_bounds();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: Some("phase_c_package"),
        easy_candidate_width: Some(candidate.easy_width),
        easy_candidate_floor: Some(candidate.easy_min_term_floor),
        easy_candidate_term_min: Some(easy_leaf_term_min),
        easy_candidate_term_max: Some(hard_rule.term_max()),
        s1_term_min_exclusive: frontiers.s1.term_min_exclusive,
        hard_leaf_term_max: hard_rule.term_max(),
        hard_leaf_term_min: hard_rule.term_min(),
        hard_rule_kind: hard_rule.kind_name(),
        hard_rule_alpha: hard_rule.alpha(),
        easy_leaf_term_value: easy_rule.representative_term_value(),
        easy_leaf_term_min,
        easy_leaf_term_max,
        easy_rule_kind: easy_rule.kind_name(),
        easy_rule_alpha: easy_rule.alpha(),
        active_len: domains.active.len(),
        s1_len: domains.s1.leaves.len(),
        s2_easy_len: domains.s2_easy.leaves.len(),
        s2_hard_len: domains.s2_hard.leaves.len(),
        s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
        contributions,
        result,
    }
}

pub fn analyze_with_phase_c_linked_package(x: u128) -> DrAnalysis {
    let hard_leaf_term_max = DrTuning::PHASE_C_LINKED_EASY_MIN_TERM_FLOOR
        .saturating_add(DrTuning::PHASE_C_LINKED_EASY_WIDTH - 1)
        .saturating_add(DrTuning::PHASE_C_LINKED_HARD_WIDTH);
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let candidate = DrTuning::phase_c_linked_candidate();
    let frontiers = ctx.params.phase_c_linked_frontier_set();
    analyze_with_phase_c_linked_frontiers(ctx, candidate, frontiers)
}

pub fn analyze_with_phase_c_boundary_package(x: u128) -> DrAnalysis {
    let candidate = DrTuning::phase_c_boundary_candidate();
    let hard_leaf_term_max = candidate.boundary_term.saturating_add(candidate.hard_width);
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let frontiers = ctx.params.phase_c_boundary_frontier_set();
    analyze_with_phase_c_boundary_frontiers(ctx, candidate, frontiers)
}

pub fn analyze_with_phase_c_buffered_boundary_package(x: u128) -> DrAnalysis {
    let candidate = DrTuning::phase_c_buffered_boundary_candidate();
    let hard_leaf_term_max = candidate
        .boundary_term
        .saturating_add(candidate.gap_width.max(1))
        .saturating_add(candidate.hard_width.max(1));
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let frontiers = ctx.params.phase_c_buffered_boundary_frontier_set();
    analyze_with_phase_c_buffered_boundary_frontiers(ctx, candidate, frontiers)
}

pub fn analyze_with_phase_c_quotient_window_package(x: u128) -> DrAnalysis {
    let candidate = DrTuning::phase_c_quotient_window_candidate();
    analyze_with_experimental_phase_c_quotient_window_package(
        x,
        candidate.easy_q_offset_max,
        candidate.hard_q_width,
    )
}

pub fn analyze_with_phase_c_boundary_quotient_guard_package(x: u128) -> DrAnalysis {
    let candidate = DrTuning::phase_c_boundary_quotient_guard_candidate();
    analyze_with_experimental_phase_c_boundary_quotient_guard_package(
        x,
        candidate.boundary_term,
        candidate.easy_width,
        candidate.hard_width,
        candidate.guard_q_offset,
    )
}

pub fn analyze_with_phase_c_boundary_relative_quotient_band_package(x: u128) -> DrAnalysis {
    let candidate = DrTuning::phase_c_boundary_relative_quotient_band_candidate();
    analyze_with_experimental_phase_c_boundary_relative_quotient_band_package(
        x,
        candidate.boundary_term,
        candidate.easy_width,
        candidate.hard_width,
        candidate.easy_q_band_width,
        candidate.hard_q_band_width,
    )
}

pub fn analyze_with_phase_c_boundary_relative_quotient_step_band_package(x: u128) -> DrAnalysis {
    let candidate = DrTuning::phase_c_boundary_relative_quotient_step_band_candidate();
    analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
        x,
        candidate.boundary_term,
        candidate.easy_width,
        candidate.hard_width,
        candidate.easy_q_step_multiplier,
        candidate.hard_q_step_multiplier,
    )
}

pub fn analyze_with_phase_c_boundary_relative_quotient_step_bridge_package(x: u128) -> DrAnalysis {
    let candidate = DrTuning::phase_c_boundary_relative_quotient_step_bridge_candidate();
    analyze_with_experimental_phase_c_boundary_relative_quotient_step_bridge_package(
        x,
        candidate.boundary_term,
        candidate.easy_width,
        candidate.hard_width,
        candidate.easy_q_step_multiplier,
        candidate.hard_q_step_multiplier,
        candidate.bridge_width,
    )
}

fn analyze_with_phase_c_boundary_frontiers(
    ctx: DrContext<'static>,
    candidate: crate::parameters::PhaseCBoundaryCandidate,
    frontiers: crate::parameters::FrontierSet,
) -> DrAnalysis {
    let domain_set = ctx.domain_set_with_frontiers(frontiers);
    let domains = domain_set.domains;
    let contributions = domain_set.contributions;
    let result = domain_set.result;
    let easy_rule = frontiers.s2_easy;
    let hard_rule = frontiers.s2_hard;
    let (easy_leaf_term_min, easy_leaf_term_max) = easy_rule.term_bounds();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: Some("phase_c_boundary_package"),
        easy_candidate_width: Some(candidate.easy_width),
        easy_candidate_floor: Some(candidate.boundary_term),
        easy_candidate_term_min: Some(easy_leaf_term_min),
        easy_candidate_term_max: Some(hard_rule.term_max()),
        s1_term_min_exclusive: frontiers.s1.term_min_exclusive,
        hard_leaf_term_max: hard_rule.term_max(),
        hard_leaf_term_min: hard_rule.term_min(),
        hard_rule_kind: hard_rule.kind_name(),
        hard_rule_alpha: hard_rule.alpha(),
        easy_leaf_term_value: easy_rule.representative_term_value(),
        easy_leaf_term_min,
        easy_leaf_term_max,
        easy_rule_kind: easy_rule.kind_name(),
        easy_rule_alpha: easy_rule.alpha(),
        active_len: domains.active.len(),
        s1_len: domains.s1.leaves.len(),
        s2_easy_len: domains.s2_easy.leaves.len(),
        s2_hard_len: domains.s2_hard.leaves.len(),
        s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
        contributions,
        result,
    }
}

fn analyze_with_phase_c_buffered_boundary_frontiers(
    ctx: DrContext<'static>,
    candidate: crate::parameters::PhaseCBufferedBoundaryCandidate,
    frontiers: crate::parameters::FrontierSet,
) -> DrAnalysis {
    let domain_set = ctx.domain_set_with_frontiers(frontiers);
    let domains = domain_set.domains;
    let contributions = domain_set.contributions;
    let result = domain_set.result;
    let easy_rule = frontiers.s2_easy;
    let hard_rule = frontiers.s2_hard;
    let (easy_leaf_term_min, easy_leaf_term_max) = easy_rule.term_bounds();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: Some("phase_c_buffered_boundary_package"),
        easy_candidate_width: Some(candidate.easy_width),
        easy_candidate_floor: Some(candidate.boundary_term),
        easy_candidate_term_min: Some(easy_leaf_term_min),
        easy_candidate_term_max: Some(hard_rule.term_max()),
        s1_term_min_exclusive: frontiers.s1.term_min_exclusive,
        hard_leaf_term_max: hard_rule.term_max(),
        hard_leaf_term_min: hard_rule.term_min(),
        hard_rule_kind: hard_rule.kind_name(),
        hard_rule_alpha: hard_rule.alpha(),
        easy_leaf_term_value: easy_rule.representative_term_value(),
        easy_leaf_term_min,
        easy_leaf_term_max,
        easy_rule_kind: easy_rule.kind_name(),
        easy_rule_alpha: easy_rule.alpha(),
        active_len: domains.active.len(),
        s1_len: domains.s1.leaves.len(),
        s2_easy_len: domains.s2_easy.leaves.len(),
        s2_hard_len: domains.s2_hard.leaves.len(),
        s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
        contributions,
        result,
    }
}

pub fn analyze_with_experimental_phase_c_boundary_package(
    x: u128,
    boundary_term: u128,
    easy_width: u128,
    hard_width: u128,
) -> DrAnalysis {
    let candidate = crate::parameters::PhaseCBoundaryCandidate {
        boundary_term,
        easy_width,
        hard_width,
    };
    let hard_leaf_term_max = candidate
        .boundary_term
        .saturating_add(candidate.hard_width.max(1));
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let frontiers = candidate.build_frontier_set(ctx.params.tuning.alpha);
    analyze_with_phase_c_boundary_frontiers(ctx, candidate, frontiers)
}

pub fn analyze_with_experimental_phase_c_buffered_boundary_package(
    x: u128,
    boundary_term: u128,
    easy_width: u128,
    gap_width: u128,
    hard_width: u128,
) -> DrAnalysis {
    let candidate = crate::parameters::PhaseCBufferedBoundaryCandidate {
        boundary_term,
        easy_width,
        gap_width,
        hard_width,
    };
    let hard_leaf_term_max = candidate
        .boundary_term
        .saturating_add(candidate.gap_width.max(1))
        .saturating_add(candidate.hard_width.max(1));
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let frontiers = candidate.build_frontier_set(ctx.params.tuning.alpha);
    analyze_with_phase_c_buffered_boundary_frontiers(ctx, candidate, frontiers)
}

pub fn analyze_with_experimental_phase_c_quotient_window_package(
    x: u128,
    easy_q_offset_max: u128,
    hard_q_width: u128,
) -> DrAnalysis {
    let ctx = prepare_context(x);
    analyze_with_phase_c_quotient_window_bounds(ctx, 0, easy_q_offset_max, hard_q_width)
}

pub fn analyze_with_experimental_phase_c_shifted_quotient_window_package(
    x: u128,
    easy_q_offset_min: u128,
    easy_q_offset_max: u128,
    hard_q_width: u128,
) -> DrAnalysis {
    let ctx = prepare_context(x);
    analyze_with_phase_c_quotient_window_bounds(
        ctx,
        easy_q_offset_min,
        easy_q_offset_max,
        hard_q_width,
    )
}

pub fn analyze_with_experimental_phase_c_boundary_quotient_guard_package(
    x: u128,
    boundary_term: u128,
    easy_width: u128,
    hard_width: u128,
    guard_q_offset: u128,
) -> DrAnalysis {
    let ctx = prepare_context(x);
    let boundary_candidate = crate::parameters::PhaseCBoundaryCandidate {
        boundary_term,
        easy_width,
        hard_width,
    };
    let frontiers = boundary_candidate.build_frontier_set(ctx.params.tuning.alpha);
    let base_domains = ctx.dr_domains_with_frontiers(frontiers);
    let threshold_q = ctx.z.saturating_add(guard_q_offset);

    let mut guard_start = base_domains.s2_hard.leaves.start;
    while guard_start < ctx.b {
        let Some(q) = ctx.quotient_at(guard_start) else {
            break;
        };
        if q <= threshold_q {
            break;
        }
        guard_start += 1;
    }

    let base_easy_start = base_domains.s2_easy.leaves.start;
    let filtered_easy_start = base_easy_start.max(guard_start);
    let filtered_hard_start = guard_start.min(base_easy_start);

    let sum_range = |start: usize, end: usize| -> u128 {
        (start..end)
            .map(|j| ctx.s2_term_at(j).expect("domain index must be valid"))
            .sum()
    };

    let contributions = DrContributions {
        trivial: if ctx.meissel_s3_is_trivial_zero() {
            0
        } else {
            trivial::trivial_leaves(&ctx)
        },
        ordinary: sum_range(ctx.a, filtered_hard_start),
        hard: sum_range(filtered_hard_start, filtered_easy_start),
        easy: sum_range(filtered_easy_start, ctx.b),
    };
    let result = ctx.phi_x_a + ctx.a as u128 - 1 - contributions.total();
    let (easy_leaf_term_min, easy_leaf_term_max) = frontiers.s2_easy.term_bounds();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: Some("phase_c_boundary_quotient_guard_package"),
        easy_candidate_width: Some(easy_width),
        easy_candidate_floor: Some(threshold_q),
        easy_candidate_term_min: Some(boundary_term),
        easy_candidate_term_max: Some(boundary_term.saturating_add(hard_width)),
        s1_term_min_exclusive: 0,
        hard_leaf_term_max: frontiers.s2_hard.term_max(),
        hard_leaf_term_min: frontiers.s2_hard.term_min(),
        hard_rule_kind: "boundary_with_quotient_guard",
        hard_rule_alpha: None,
        easy_leaf_term_value: 0,
        easy_leaf_term_min,
        easy_leaf_term_max,
        easy_rule_kind: "boundary_with_quotient_guard",
        easy_rule_alpha: None,
        active_len: ctx.b.saturating_sub(ctx.a),
        s1_len: filtered_hard_start.saturating_sub(ctx.a),
        s2_easy_len: ctx.b.saturating_sub(filtered_easy_start),
        s2_hard_len: filtered_easy_start.saturating_sub(filtered_hard_start),
        s2_trivial_is_zero: ctx.meissel_s3_is_trivial_zero(),
        contributions,
        result,
    }
}

pub fn analyze_with_experimental_phase_c_boundary_relative_quotient_band_package(
    x: u128,
    boundary_term: u128,
    easy_width: u128,
    hard_width: u128,
    easy_q_band_width: u128,
    hard_q_band_width: u128,
) -> DrAnalysis {
    let ctx = prepare_context(x);
    let boundary_candidate = crate::parameters::PhaseCBoundaryCandidate {
        boundary_term,
        easy_width,
        hard_width,
    };
    let frontiers = boundary_candidate.build_frontier_set(ctx.params.tuning.alpha);
    let base_domains = ctx.dr_domains_with_frontiers(frontiers);

    let base_easy_start = base_domains.s2_easy.leaves.start;
    let base_hard_start = base_domains.s2_hard.leaves.start;
    let q_easy_ref = if ctx.b > 0 {
        ctx.quotient_at(ctx.b - 1).unwrap_or(0)
    } else {
        0
    };
    let q_hard_ref = if base_domains.s2_hard.leaves.end > base_hard_start {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 1)
            .unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };

    let mut filtered_easy_start = ctx.b;
    while filtered_easy_start > base_easy_start {
        let index = filtered_easy_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= q_easy_ref.saturating_add(easy_q_band_width) {
            filtered_easy_start -= 1;
        } else {
            break;
        }
    }

    let mut filtered_hard_start = filtered_easy_start;
    while filtered_hard_start > base_hard_start {
        let index = filtered_hard_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= q_hard_ref.saturating_add(hard_q_band_width) {
            filtered_hard_start -= 1;
        } else {
            break;
        }
    }

    let sum_range = |start: usize, end: usize| -> u128 {
        (start..end)
            .map(|j| ctx.s2_term_at(j).expect("domain index must be valid"))
            .sum()
    };

    let contributions = DrContributions {
        trivial: if ctx.meissel_s3_is_trivial_zero() {
            0
        } else {
            trivial::trivial_leaves(&ctx)
        },
        ordinary: sum_range(ctx.a, filtered_hard_start),
        hard: sum_range(filtered_hard_start, filtered_easy_start),
        easy: sum_range(filtered_easy_start, ctx.b),
    };
    let result = ctx.phi_x_a + ctx.a as u128 - 1 - contributions.total();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: Some("phase_c_boundary_relative_quotient_band_package"),
        easy_candidate_width: Some(easy_width),
        easy_candidate_floor: Some(q_easy_ref),
        easy_candidate_term_min: Some(q_hard_ref),
        easy_candidate_term_max: Some(q_easy_ref),
        s1_term_min_exclusive: 0,
        hard_leaf_term_max: frontiers.s2_hard.term_max(),
        hard_leaf_term_min: frontiers.s2_hard.term_min(),
        hard_rule_kind: "boundary_with_relative_quotient_band",
        hard_rule_alpha: None,
        easy_leaf_term_value: 0,
        easy_leaf_term_min: frontiers.s2_easy.term_bounds().0,
        easy_leaf_term_max: frontiers.s2_easy.term_bounds().1,
        easy_rule_kind: "boundary_with_relative_quotient_band",
        easy_rule_alpha: None,
        active_len: ctx.b.saturating_sub(ctx.a),
        s1_len: filtered_hard_start.saturating_sub(ctx.a),
        s2_easy_len: ctx.b.saturating_sub(filtered_easy_start),
        s2_hard_len: filtered_easy_start.saturating_sub(filtered_hard_start),
        s2_trivial_is_zero: ctx.meissel_s3_is_trivial_zero(),
        contributions,
        result,
    }
}

pub fn analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
    x: u128,
    boundary_term: u128,
    easy_width: u128,
    hard_width: u128,
    easy_q_step_multiplier: u128,
    hard_q_step_multiplier: u128,
) -> DrAnalysis {
    let ctx = prepare_context(x);
    let boundary_candidate = crate::parameters::PhaseCBoundaryCandidate {
        boundary_term,
        easy_width,
        hard_width,
    };
    let frontiers = boundary_candidate.build_frontier_set(ctx.params.tuning.alpha);
    let base_domains = ctx.dr_domains_with_frontiers(frontiers);

    let base_easy_start = base_domains.s2_easy.leaves.start;
    let base_hard_start = base_domains.s2_hard.leaves.start;

    let q_easy_ref = if ctx.b > 0 {
        ctx.quotient_at(ctx.b - 1).unwrap_or(0)
    } else {
        0
    };
    let q_easy_prev = if ctx.b >= 2 {
        ctx.quotient_at(ctx.b - 2).unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let easy_q_step = q_easy_prev.saturating_sub(q_easy_ref);

    let q_hard_ref = if base_domains.s2_hard.leaves.end > base_hard_start {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 1)
            .unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let q_hard_prev = if base_domains.s2_hard.leaves.end >= base_hard_start + 2 {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 2)
            .unwrap_or(q_hard_ref)
    } else {
        q_hard_ref
    };
    let hard_q_step = q_hard_prev.saturating_sub(q_hard_ref);

    let easy_q_limit =
        q_easy_ref.saturating_add(easy_q_step.saturating_mul(easy_q_step_multiplier));
    let hard_q_limit =
        q_hard_ref.saturating_add(hard_q_step.saturating_mul(hard_q_step_multiplier));

    let mut filtered_easy_start = ctx.b;
    while filtered_easy_start > base_easy_start {
        let index = filtered_easy_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= easy_q_limit {
            filtered_easy_start -= 1;
        } else {
            break;
        }
    }

    let mut filtered_hard_start = filtered_easy_start;
    while filtered_hard_start > base_hard_start {
        let index = filtered_hard_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= hard_q_limit {
            filtered_hard_start -= 1;
        } else {
            break;
        }
    }

    let sum_range = |start: usize, end: usize| -> u128 {
        (start..end)
            .map(|j| ctx.s2_term_at(j).expect("domain index must be valid"))
            .sum()
    };

    let contributions = DrContributions {
        trivial: if ctx.meissel_s3_is_trivial_zero() {
            0
        } else {
            trivial::trivial_leaves(&ctx)
        },
        ordinary: sum_range(ctx.a, filtered_hard_start),
        hard: sum_range(filtered_hard_start, filtered_easy_start),
        easy: sum_range(filtered_easy_start, ctx.b),
    };
    let result = ctx.phi_x_a + ctx.a as u128 - 1 - contributions.total();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: Some("phase_c_boundary_relative_quotient_step_band_package"),
        easy_candidate_width: Some(easy_q_step_multiplier),
        easy_candidate_floor: Some(hard_q_step_multiplier),
        easy_candidate_term_min: Some(hard_q_limit),
        easy_candidate_term_max: Some(easy_q_limit),
        s1_term_min_exclusive: 0,
        hard_leaf_term_max: frontiers.s2_hard.term_max(),
        hard_leaf_term_min: frontiers.s2_hard.term_min(),
        hard_rule_kind: "boundary_with_relative_quotient_step_band",
        hard_rule_alpha: None,
        easy_leaf_term_value: 0,
        easy_leaf_term_min: frontiers.s2_easy.term_bounds().0,
        easy_leaf_term_max: frontiers.s2_easy.term_bounds().1,
        easy_rule_kind: "boundary_with_relative_quotient_step_band",
        easy_rule_alpha: None,
        active_len: ctx.b.saturating_sub(ctx.a),
        s1_len: filtered_hard_start.saturating_sub(ctx.a),
        s2_easy_len: ctx.b.saturating_sub(filtered_easy_start),
        s2_hard_len: filtered_easy_start.saturating_sub(filtered_hard_start),
        s2_trivial_is_zero: ctx.meissel_s3_is_trivial_zero(),
        contributions,
        result,
    }
}

pub fn analyze_with_experimental_phase_c_boundary_relative_quotient_step_bridge_package(
    x: u128,
    boundary_term: u128,
    easy_width: u128,
    hard_width: u128,
    easy_q_step_multiplier: u128,
    hard_q_step_multiplier: u128,
    bridge_width: u128,
) -> DrAnalysis {
    let ctx = prepare_context(x);
    let boundary_candidate = crate::parameters::PhaseCBoundaryCandidate {
        boundary_term,
        easy_width,
        hard_width,
    };
    let frontiers = boundary_candidate.build_frontier_set(ctx.params.tuning.alpha);
    let base_domains = ctx.dr_domains_with_frontiers(frontiers);

    let base_easy_start = base_domains.s2_easy.leaves.start;
    let base_hard_start = base_domains.s2_hard.leaves.start;

    let q_easy_ref = if ctx.b > 0 {
        ctx.quotient_at(ctx.b - 1).unwrap_or(0)
    } else {
        0
    };
    let q_easy_prev = if ctx.b >= 2 {
        ctx.quotient_at(ctx.b - 2).unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let easy_q_step = q_easy_prev.saturating_sub(q_easy_ref);

    let q_hard_ref = if base_domains.s2_hard.leaves.end > base_hard_start {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 1)
            .unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let q_hard_prev = if base_domains.s2_hard.leaves.end >= base_hard_start + 2 {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 2)
            .unwrap_or(q_hard_ref)
    } else {
        q_hard_ref
    };
    let hard_q_step = q_hard_prev.saturating_sub(q_hard_ref);

    let easy_q_limit =
        q_easy_ref.saturating_add(easy_q_step.saturating_mul(easy_q_step_multiplier));
    let hard_q_limit =
        q_hard_ref.saturating_add(hard_q_step.saturating_mul(hard_q_step_multiplier));
    let bridge_span = easy_q_step
        .max(hard_q_step)
        .max(1)
        .saturating_mul(bridge_width.max(1));
    let bridged_hard_q_limit = hard_q_limit.min(easy_q_limit.saturating_add(bridge_span));

    let mut filtered_easy_start = ctx.b;
    while filtered_easy_start > base_easy_start {
        let index = filtered_easy_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= easy_q_limit {
            filtered_easy_start -= 1;
        } else {
            break;
        }
    }

    let mut filtered_hard_start = filtered_easy_start;
    while filtered_hard_start > base_hard_start {
        let index = filtered_hard_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= bridged_hard_q_limit {
            filtered_hard_start -= 1;
        } else {
            break;
        }
    }

    let sum_range = |start: usize, end: usize| -> u128 {
        (start..end)
            .map(|j| ctx.s2_term_at(j).expect("domain index must be valid"))
            .sum()
    };

    let contributions = DrContributions {
        trivial: if ctx.meissel_s3_is_trivial_zero() {
            0
        } else {
            trivial::trivial_leaves(&ctx)
        },
        ordinary: sum_range(ctx.a, filtered_hard_start),
        hard: sum_range(filtered_hard_start, filtered_easy_start),
        easy: sum_range(filtered_easy_start, ctx.b),
    };
    let result = ctx.phi_x_a + ctx.a as u128 - 1 - contributions.total();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: Some("phase_c_boundary_relative_quotient_step_bridge_package"),
        easy_candidate_width: Some(easy_q_step_multiplier),
        easy_candidate_floor: Some(hard_q_step_multiplier),
        easy_candidate_term_min: Some(bridged_hard_q_limit),
        easy_candidate_term_max: Some(easy_q_limit),
        s1_term_min_exclusive: 0,
        hard_leaf_term_max: frontiers.s2_hard.term_max(),
        hard_leaf_term_min: frontiers.s2_hard.term_min(),
        hard_rule_kind: "boundary_with_relative_quotient_step_bridge",
        hard_rule_alpha: None,
        easy_leaf_term_value: 0,
        easy_leaf_term_min: frontiers.s2_easy.term_bounds().0,
        easy_leaf_term_max: frontiers.s2_easy.term_bounds().1,
        easy_rule_kind: "boundary_with_relative_quotient_step_bridge",
        easy_rule_alpha: None,
        active_len: ctx.b.saturating_sub(ctx.a),
        s1_len: filtered_hard_start.saturating_sub(ctx.a),
        s2_easy_len: ctx.b.saturating_sub(filtered_easy_start),
        s2_hard_len: filtered_easy_start.saturating_sub(filtered_hard_start),
        s2_trivial_is_zero: ctx.meissel_s3_is_trivial_zero(),
        contributions,
        result,
    }
}

fn analyze_with_phase_c_quotient_window_bounds(
    ctx: DrContext<'static>,
    easy_q_offset_min: u128,
    easy_q_offset_max: u128,
    hard_q_width: u128,
) -> DrAnalysis {
    let easy_q_min = ctx.z.saturating_add(easy_q_offset_min);
    let easy_q_max = ctx
        .z
        .saturating_add(easy_q_offset_max.max(easy_q_offset_min));
    let hard_q_min = easy_q_max.saturating_add(1);
    let hard_q_max = hard_q_min.saturating_add(hard_q_width.max(1) - 1);

    let mut easy_start = ctx.b;
    while easy_start > ctx.a {
        let index = easy_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q >= easy_q_min && q <= easy_q_max {
            easy_start -= 1;
        } else {
            break;
        }
    }

    let mut hard_start = easy_start;
    while hard_start > ctx.a {
        let index = hard_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q >= hard_q_min && q <= hard_q_max {
            hard_start -= 1;
        } else {
            break;
        }
    }

    let sum_range = |start: usize, end: usize| -> u128 {
        (start..end)
            .map(|j| ctx.s2_term_at(j).expect("domain index must be valid"))
            .sum()
    };

    let contributions = DrContributions {
        trivial: if ctx.meissel_s3_is_trivial_zero() {
            0
        } else {
            trivial::trivial_leaves(&ctx)
        },
        ordinary: sum_range(ctx.a, hard_start),
        hard: sum_range(hard_start, easy_start),
        easy: sum_range(easy_start, ctx.b),
    };
    let result = ctx.phi_x_a + ctx.a as u128 - 1 - contributions.total();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: Some("phase_c_quotient_window_package"),
        easy_candidate_width: Some(
            easy_q_offset_max.max(easy_q_offset_min) - easy_q_offset_min + 1,
        ),
        easy_candidate_floor: Some(easy_q_min),
        easy_candidate_term_min: Some(easy_q_min),
        easy_candidate_term_max: Some(hard_q_max),
        s1_term_min_exclusive: 0,
        hard_leaf_term_max: 0,
        hard_leaf_term_min: 0,
        hard_rule_kind: "quotient_window",
        hard_rule_alpha: None,
        easy_leaf_term_value: 0,
        easy_leaf_term_min: 0,
        easy_leaf_term_max: 0,
        easy_rule_kind: "quotient_window",
        easy_rule_alpha: None,
        active_len: ctx.b.saturating_sub(ctx.a),
        s1_len: hard_start.saturating_sub(ctx.a),
        s2_easy_len: ctx.b.saturating_sub(easy_start),
        s2_hard_len: easy_start.saturating_sub(hard_start),
        s2_trivial_is_zero: ctx.meissel_s3_is_trivial_zero(),
        contributions,
        result,
    }
}

fn analyze_with_phase_c_linked_frontiers(
    ctx: DrContext<'static>,
    candidate: crate::parameters::PhaseCLinkedCandidate,
    frontiers: crate::parameters::FrontierSet,
) -> DrAnalysis {
    let domain_set = ctx.domain_set_with_frontiers(frontiers);
    let domains = domain_set.domains;
    let contributions = domain_set.contributions;
    let result = domain_set.result;
    let easy_rule = frontiers.s2_easy;
    let hard_rule = frontiers.s2_hard;
    let (easy_leaf_term_min, easy_leaf_term_max) = easy_rule.term_bounds();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: Some("phase_c_linked_package"),
        easy_candidate_width: Some(candidate.easy_width),
        easy_candidate_floor: Some(candidate.easy_min_term_floor),
        easy_candidate_term_min: Some(easy_leaf_term_min),
        easy_candidate_term_max: Some(hard_rule.term_max()),
        s1_term_min_exclusive: frontiers.s1.term_min_exclusive,
        hard_leaf_term_max: hard_rule.term_max(),
        hard_leaf_term_min: hard_rule.term_min(),
        hard_rule_kind: hard_rule.kind_name(),
        hard_rule_alpha: hard_rule.alpha(),
        easy_leaf_term_value: easy_rule.representative_term_value(),
        easy_leaf_term_min,
        easy_leaf_term_max,
        easy_rule_kind: easy_rule.kind_name(),
        easy_rule_alpha: easy_rule.alpha(),
        active_len: domains.active.len(),
        s1_len: domains.s1.leaves.len(),
        s2_easy_len: domains.s2_easy.leaves.len(),
        s2_hard_len: domains.s2_hard.leaves.len(),
        s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
        contributions,
        result,
    }
}

pub fn analyze_with_experimental_phase_c_linked_package(
    x: u128,
    easy_width: u128,
    easy_floor: u128,
    hard_width: u128,
) -> DrAnalysis {
    let hard_leaf_term_max = easy_floor
        .max(1)
        .saturating_add(easy_width.max(1) - 1)
        .saturating_add(hard_width.max(1));
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let candidate = crate::parameters::PhaseCLinkedCandidate {
        easy_width,
        easy_min_term_floor: easy_floor,
        hard_width,
    };
    let frontiers = candidate.build_frontier_set(ctx.params.tuning.alpha);
    analyze_with_phase_c_linked_frontiers(ctx, candidate, frontiers)
}

pub fn analyze_with_experimental_easy_term_band(
    x: u128,
    hard_leaf_term_max: u128,
    min_term: u128,
    max_term: u128,
) -> DrAnalysis {
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let easy_rule = crate::parameters::S2EasyRule::alpha_balanced_term_range(
        min_term,
        max_term.max(min_term),
        ctx.params.tuning.alpha,
    );
    let (_, easy_leaf_term_max) = easy_rule.term_bounds();
    let hard_rule = crate::parameters::S2HardRule::alpha_balanced_term_range(
        easy_leaf_term_max.saturating_add(1),
        hard_leaf_term_max,
        ctx.params.tuning.alpha,
    );
    let frontiers = crate::parameters::FrontierSet {
        s1: crate::parameters::S1Rule {
            term_min_exclusive: hard_leaf_term_max,
        },
        s2_hard: hard_rule,
        s2_easy: easy_rule,
    };
    let domain_set = ctx.domain_set_with_frontiers(frontiers);
    let domains = domain_set.domains;
    let contributions = domain_set.contributions;
    let result = domain_set.result;
    let (easy_leaf_term_min, easy_leaf_term_max) = easy_rule.term_bounds();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: Some("term_band"),
        easy_candidate_width: None,
        easy_candidate_floor: None,
        easy_candidate_term_min: Some(min_term),
        easy_candidate_term_max: Some(max_term.max(min_term)),
        s1_term_min_exclusive: frontiers.s1.term_min_exclusive,
        hard_leaf_term_max: hard_rule.term_max(),
        hard_leaf_term_min: hard_rule.term_min(),
        hard_rule_kind: hard_rule.kind_name(),
        hard_rule_alpha: hard_rule.alpha(),
        easy_leaf_term_value: easy_rule.representative_term_value(),
        easy_leaf_term_min,
        easy_leaf_term_max,
        easy_rule_kind: easy_rule.kind_name(),
        easy_rule_alpha: easy_rule.alpha(),
        active_len: domains.active.len(),
        s1_len: domains.s1.leaves.len(),
        s2_easy_len: domains.s2_easy.leaves.len(),
        s2_hard_len: domains.s2_hard.leaves.len(),
        s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
        contributions,
        result,
    }
}

pub fn compare_current_vs_candidate_easy_relative_to_hard(
    x: u128,
    hard_leaf_term_max: u128,
    easy_leaf_term_max: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_term_frontiers(x, hard_leaf_term_max, easy_leaf_term_max),
        experimental: analyze_with_candidate_easy_relative_to_hard(x, hard_leaf_term_max),
    }
}

pub fn compare_current_vs_candidate_easy_relative_to_hard_with_floor(
    x: u128,
    hard_leaf_term_max: u128,
    easy_leaf_term_max: u128,
    min_term_floor: u128,
) -> AnalysisComparison {
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let candidate = crate::parameters::EasyRelativeToHardCandidate {
        width: crate::parameters::DrTuning::CANDIDATE_EASY_RELATIVE_TO_HARD_WIDTH,
        min_term_floor,
    };
    let easy_rule = candidate.build_rule(hard_leaf_term_max, ctx.params.tuning.alpha);
    let (_, easy_leaf_term_max_for_rule) = easy_rule.term_bounds();
    let hard_rule = crate::parameters::S2HardRule::alpha_balanced_term_range(
        easy_leaf_term_max_for_rule.saturating_add(1),
        hard_leaf_term_max,
        ctx.params.tuning.alpha,
    );
    let frontiers = crate::parameters::FrontierSet {
        s1: crate::parameters::S1Rule {
            term_min_exclusive: hard_leaf_term_max,
        },
        s2_hard: hard_rule,
        s2_easy: easy_rule,
    };
    let domain_set = ctx.domain_set_with_frontiers(frontiers);
    let domains = domain_set.domains;
    let contributions = domain_set.contributions;
    let result = domain_set.result;
    let (easy_leaf_term_min, easy_leaf_term_max_for_rule) = easy_rule.term_bounds();

    AnalysisComparison {
        current: analyze_with_term_frontiers(x, hard_leaf_term_max, easy_leaf_term_max),
        experimental: DrAnalysis {
            alpha: ctx.params.tuning.alpha,
            phi_x_a: ctx.phi_x_a,
            a: ctx.a,
            easy_candidate_family: Some("relative_to_hard"),
            easy_candidate_width: Some(candidate.width),
            easy_candidate_floor: Some(candidate.min_term_floor),
            easy_candidate_term_min: None,
            easy_candidate_term_max: None,
            s1_term_min_exclusive: frontiers.s1.term_min_exclusive,
            hard_leaf_term_max: hard_rule.term_max(),
            hard_leaf_term_min: hard_rule.term_min(),
            hard_rule_kind: hard_rule.kind_name(),
            hard_rule_alpha: hard_rule.alpha(),
            easy_leaf_term_value: easy_rule.representative_term_value(),
            easy_leaf_term_min,
            easy_leaf_term_max: easy_leaf_term_max_for_rule,
            easy_rule_kind: easy_rule.kind_name(),
            easy_rule_alpha: easy_rule.alpha(),
            active_len: domains.active.len(),
            s1_len: domains.s1.leaves.len(),
            s2_easy_len: domains.s2_easy.leaves.len(),
            s2_hard_len: domains.s2_hard.leaves.len(),
            s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
            contributions,
            result,
        },
    }
}

pub fn compare_current_vs_candidate_easy_term_band(
    x: u128,
    hard_leaf_term_max: u128,
    easy_leaf_term_max: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_term_frontiers(x, hard_leaf_term_max, easy_leaf_term_max),
        experimental: analyze_with_candidate_easy_term_band(x, hard_leaf_term_max),
    }
}

pub fn compare_current_vs_phase_c_easy_term_band(
    x: u128,
    hard_leaf_term_max: u128,
    easy_leaf_term_max: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_term_frontiers(x, hard_leaf_term_max, easy_leaf_term_max),
        experimental: analyze_with_phase_c_easy_term_band(x, hard_leaf_term_max),
    }
}

pub fn compare_candidate_easy_reference_vs_phase_c_term_band(
    x: u128,
    hard_leaf_term_max: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_candidate_easy_relative_to_hard(x, hard_leaf_term_max),
        experimental: analyze_with_phase_c_easy_term_band(x, hard_leaf_term_max + 1),
    }
}

pub fn compare_phase_c_easy_term_bands(
    x: u128,
    current_min_term: u128,
    current_max_term: u128,
    experimental_min_term: u128,
    experimental_max_term: u128,
) -> AnalysisComparison {
    let hard_leaf_term_max = current_max_term
        .max(experimental_max_term)
        .saturating_add(1);
    AnalysisComparison {
        current: analyze_with_experimental_easy_term_band(
            x,
            hard_leaf_term_max,
            current_min_term,
            current_max_term,
        ),
        experimental: analyze_with_experimental_easy_term_band(
            x,
            hard_leaf_term_max,
            experimental_min_term,
            experimental_max_term,
        ),
    }
}

pub fn compare_phase_c_hard_term_band_with_current(
    x: u128,
    hard_leaf_term_max: u128,
    easy_leaf_term_max: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_term_frontiers(x, hard_leaf_term_max, easy_leaf_term_max),
        experimental: analyze_with_phase_c_hard_term_band(x),
    }
}

pub fn compare_phase_c_hard_term_bands(
    x: u128,
    current_min_term: u128,
    current_max_term: u128,
    experimental_min_term: u128,
    experimental_max_term: u128,
) -> AnalysisComparison {
    let hard_leaf_term_max = current_max_term.max(experimental_max_term);
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);
    let easy_rule = DrTuning::phase_c_easy_term_band().build_rule(ctx.params.tuning.alpha);
    let current_hard = crate::parameters::S2HardRule::alpha_balanced_term_range(
        current_min_term,
        current_max_term,
        ctx.params.tuning.alpha,
    );
    let experimental_hard = crate::parameters::S2HardRule::alpha_balanced_term_range(
        experimental_min_term,
        experimental_max_term,
        ctx.params.tuning.alpha,
    );

    let make_analysis = |hard_rule: crate::parameters::S2HardRule| {
        let frontiers = crate::parameters::FrontierSet {
            s1: crate::parameters::S1Rule {
                term_min_exclusive: hard_rule.term_max(),
            },
            s2_hard: hard_rule,
            s2_easy: easy_rule,
        };
        let domain_set = ctx.domain_set_with_frontiers(frontiers);
        let domains = domain_set.domains;
        let contributions = domain_set.contributions;
        let result = domain_set.result;
        let (easy_leaf_term_min, easy_leaf_term_max) = easy_rule.term_bounds();

        DrAnalysis {
            alpha: ctx.params.tuning.alpha,
            phi_x_a: ctx.phi_x_a,
            a: ctx.a,
            easy_candidate_family: Some("phase_c_hard_term_band"),
            easy_candidate_width: None,
            easy_candidate_floor: None,
            easy_candidate_term_min: Some(hard_rule.term_min()),
            easy_candidate_term_max: Some(hard_rule.term_max()),
            s1_term_min_exclusive: frontiers.s1.term_min_exclusive,
            hard_leaf_term_max: hard_rule.term_max(),
            hard_leaf_term_min: hard_rule.term_min(),
            hard_rule_kind: hard_rule.kind_name(),
            hard_rule_alpha: hard_rule.alpha(),
            easy_leaf_term_value: easy_rule.representative_term_value(),
            easy_leaf_term_min,
            easy_leaf_term_max,
            easy_rule_kind: easy_rule.kind_name(),
            easy_rule_alpha: easy_rule.alpha(),
            active_len: domains.active.len(),
            s1_len: domains.s1.leaves.len(),
            s2_easy_len: domains.s2_easy.leaves.len(),
            s2_hard_len: domains.s2_hard.leaves.len(),
            s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
            contributions,
            result,
        }
    };

    AnalysisComparison {
        current: make_analysis(current_hard),
        experimental: make_analysis(experimental_hard),
    }
}

pub fn compare_current_vs_phase_c_package(x: u128) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze(x),
        experimental: analyze_with_phase_c_package(x),
    }
}

pub fn compare_phase_c_packages(
    x: u128,
    current_easy_min: u128,
    current_easy_max: u128,
    current_hard_min: u128,
    current_hard_max: u128,
    experimental_easy_min: u128,
    experimental_easy_max: u128,
    experimental_hard_min: u128,
    experimental_hard_max: u128,
) -> AnalysisComparison {
    let hard_leaf_term_max = current_hard_max.max(experimental_hard_max);
    let ctx = prepare_context_with_hard_leaf_term_max(x, hard_leaf_term_max);

    let make_analysis = |easy_min: u128, easy_max: u128, hard_min: u128, hard_max: u128| {
        let easy_rule = crate::parameters::S2EasyRule::alpha_balanced_term_range(
            easy_min,
            easy_max,
            ctx.params.tuning.alpha,
        );
        let hard_rule = crate::parameters::S2HardRule::alpha_balanced_term_range(
            hard_min,
            hard_max,
            ctx.params.tuning.alpha,
        );
        let frontiers = crate::parameters::FrontierSet {
            s1: crate::parameters::S1Rule {
                term_min_exclusive: hard_max,
            },
            s2_hard: hard_rule,
            s2_easy: easy_rule,
        };
        let domain_set = ctx.domain_set_with_frontiers(frontiers);
        let domains = domain_set.domains;
        let contributions = domain_set.contributions;
        let result = domain_set.result;

        DrAnalysis {
            alpha: ctx.params.tuning.alpha,
            phi_x_a: ctx.phi_x_a,
            a: ctx.a,
            easy_candidate_family: Some("phase_c_package"),
            easy_candidate_width: None,
            easy_candidate_floor: None,
            easy_candidate_term_min: Some(easy_min),
            easy_candidate_term_max: Some(hard_max),
            s1_term_min_exclusive: frontiers.s1.term_min_exclusive,
            hard_leaf_term_max: hard_rule.term_max(),
            hard_leaf_term_min: hard_rule.term_min(),
            hard_rule_kind: hard_rule.kind_name(),
            hard_rule_alpha: hard_rule.alpha(),
            easy_leaf_term_value: easy_rule.representative_term_value(),
            easy_leaf_term_min: easy_min,
            easy_leaf_term_max: easy_max,
            easy_rule_kind: easy_rule.kind_name(),
            easy_rule_alpha: easy_rule.alpha(),
            active_len: domains.active.len(),
            s1_len: domains.s1.leaves.len(),
            s2_easy_len: domains.s2_easy.leaves.len(),
            s2_hard_len: domains.s2_hard.leaves.len(),
            s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
            contributions,
            result,
        }
    };

    AnalysisComparison {
        current: make_analysis(
            current_easy_min,
            current_easy_max,
            current_hard_min,
            current_hard_max,
        ),
        experimental: make_analysis(
            experimental_easy_min,
            experimental_easy_max,
            experimental_hard_min,
            experimental_hard_max,
        ),
    }
}

pub fn compare_phase_c_package_vs_linked_package(x: u128) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_term_band_package(x),
        experimental: analyze_with_phase_c_linked_package(x),
    }
}

pub fn compare_phase_c_package_vs_boundary_package(x: u128) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental: analyze_with_phase_c_boundary_package(x),
    }
}

pub fn compare_phase_c_package_vs_boundary_candidate(x: u128) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental: analyze_with_experimental_phase_c_boundary_package(x, 4, 4, 2),
    }
}

pub fn compare_phase_c_package_vs_buffered_boundary_package(x: u128) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental: analyze_with_phase_c_buffered_boundary_package(x),
    }
}

pub fn compare_phase_c_package_vs_quotient_window_package(x: u128) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental: analyze_with_phase_c_quotient_window_package(x),
    }
}

pub fn compare_phase_c_package_vs_experimental_buffered_boundary_candidate(
    x: u128,
    boundary_term: u128,
    easy_width: u128,
    gap_width: u128,
    hard_width: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental: analyze_with_experimental_phase_c_buffered_boundary_package(
            x,
            boundary_term,
            easy_width,
            gap_width,
            hard_width,
        ),
    }
}

pub fn compare_phase_c_package_vs_experimental_quotient_window_candidate(
    x: u128,
    easy_q_offset_max: u128,
    hard_q_width: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental: analyze_with_experimental_phase_c_quotient_window_package(
            x,
            easy_q_offset_max,
            hard_q_width,
        ),
    }
}

pub fn compare_phase_c_package_vs_experimental_shifted_quotient_window_candidate(
    x: u128,
    easy_q_offset_min: u128,
    easy_q_offset_max: u128,
    hard_q_width: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental: analyze_with_experimental_phase_c_shifted_quotient_window_package(
            x,
            easy_q_offset_min,
            easy_q_offset_max,
            hard_q_width,
        ),
    }
}

pub fn compare_phase_c_package_vs_boundary_quotient_guard_package(x: u128) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental: analyze_with_phase_c_boundary_quotient_guard_package(x),
    }
}

pub fn compare_phase_c_package_vs_boundary_relative_quotient_band_package(
    x: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental: analyze_with_phase_c_boundary_relative_quotient_band_package(x),
    }
}

pub fn compare_phase_c_package_vs_boundary_relative_quotient_step_band_package(
    x: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental: analyze_with_phase_c_boundary_relative_quotient_step_band_package(x),
    }
}

pub fn compare_phase_c_package_vs_boundary_relative_quotient_step_bridge_package(
    x: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental: analyze_with_phase_c_boundary_relative_quotient_step_bridge_package(x),
    }
}

pub fn compare_phase_c_package_vs_experimental_boundary_quotient_guard_candidate(
    x: u128,
    boundary_term: u128,
    easy_width: u128,
    hard_width: u128,
    guard_q_offset: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental: analyze_with_experimental_phase_c_boundary_quotient_guard_package(
            x,
            boundary_term,
            easy_width,
            hard_width,
            guard_q_offset,
        ),
    }
}

pub fn compare_phase_c_package_vs_experimental_boundary_relative_quotient_band_candidate(
    x: u128,
    boundary_term: u128,
    easy_width: u128,
    hard_width: u128,
    easy_q_band_width: u128,
    hard_q_band_width: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental: analyze_with_experimental_phase_c_boundary_relative_quotient_band_package(
            x,
            boundary_term,
            easy_width,
            hard_width,
            easy_q_band_width,
            hard_q_band_width,
        ),
    }
}

pub fn compare_phase_c_package_vs_experimental_boundary_relative_quotient_step_band_candidate(
    x: u128,
    boundary_term: u128,
    easy_width: u128,
    hard_width: u128,
    easy_q_step_multiplier: u128,
    hard_q_step_multiplier: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental:
            analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                x,
                boundary_term,
                easy_width,
                hard_width,
                easy_q_step_multiplier,
                hard_q_step_multiplier,
            ),
    }
}

pub fn compare_phase_c_package_vs_experimental_boundary_relative_quotient_step_bridge_candidate(
    x: u128,
    boundary_term: u128,
    easy_width: u128,
    hard_width: u128,
    easy_q_step_multiplier: u128,
    hard_q_step_multiplier: u128,
    bridge_width: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental:
            analyze_with_experimental_phase_c_boundary_relative_quotient_step_bridge_package(
                x,
                boundary_term,
                easy_width,
                hard_width,
                easy_q_step_multiplier,
                hard_q_step_multiplier,
                bridge_width,
            ),
    }
}

pub fn compare_boundary_candidate_vs_experimental_boundary_relative_quotient_step_bridge_candidate(
    x: u128,
    current_boundary_term: u128,
    current_easy_width: u128,
    current_hard_width: u128,
    experimental_boundary_term: u128,
    experimental_easy_width: u128,
    experimental_hard_width: u128,
    experimental_easy_q_step_multiplier: u128,
    experimental_hard_q_step_multiplier: u128,
    experimental_bridge_width: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_experimental_phase_c_boundary_package(
            x,
            current_boundary_term,
            current_easy_width,
            current_hard_width,
        ),
        experimental:
            analyze_with_experimental_phase_c_boundary_relative_quotient_step_bridge_package(
                x,
                experimental_boundary_term,
                experimental_easy_width,
                experimental_hard_width,
                experimental_easy_q_step_multiplier,
                experimental_hard_q_step_multiplier,
                experimental_bridge_width,
            ),
    }
}

pub fn compare_step_band_vs_experimental_step_bridge_candidate(
    x: u128,
    current_boundary_term: u128,
    current_easy_width: u128,
    current_hard_width: u128,
    current_easy_q_step_multiplier: u128,
    current_hard_q_step_multiplier: u128,
    experimental_boundary_term: u128,
    experimental_easy_width: u128,
    experimental_hard_width: u128,
    experimental_easy_q_step_multiplier: u128,
    experimental_hard_q_step_multiplier: u128,
    experimental_bridge_width: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
            x,
            current_boundary_term,
            current_easy_width,
            current_hard_width,
            current_easy_q_step_multiplier,
            current_hard_q_step_multiplier,
        ),
        experimental:
            analyze_with_experimental_phase_c_boundary_relative_quotient_step_bridge_package(
                x,
                experimental_boundary_term,
                experimental_easy_width,
                experimental_hard_width,
                experimental_easy_q_step_multiplier,
                experimental_hard_q_step_multiplier,
                experimental_bridge_width,
            ),
    }
}

pub fn compare_boundary_candidate_vs_experimental_boundary_relative_quotient_step_band_candidate(
    x: u128,
    current_boundary_term: u128,
    current_easy_width: u128,
    current_hard_width: u128,
    experimental_boundary_term: u128,
    experimental_easy_width: u128,
    experimental_hard_width: u128,
    experimental_easy_q_step_multiplier: u128,
    experimental_hard_q_step_multiplier: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_experimental_phase_c_boundary_package(
            x,
            current_boundary_term,
            current_easy_width,
            current_hard_width,
        ),
        experimental:
            analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                x,
                experimental_boundary_term,
                experimental_easy_width,
                experimental_hard_width,
                experimental_easy_q_step_multiplier,
                experimental_hard_q_step_multiplier,
            ),
    }
}

pub fn compare_phase_c_package_vs_experimental_boundary_candidate(
    x: u128,
    boundary_term: u128,
    easy_width: u128,
    hard_width: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_package(x),
        experimental: analyze_with_experimental_phase_c_boundary_package(
            x,
            boundary_term,
            easy_width,
            hard_width,
        ),
    }
}

pub fn compare_phase_c_boundary_variants(
    x: u128,
    current: crate::parameters::PhaseCBoundaryCandidate,
    experimental: crate::parameters::PhaseCBoundaryCandidate,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_experimental_phase_c_boundary_package(
            x,
            current.boundary_term,
            current.easy_width,
            current.hard_width,
        ),
        experimental: analyze_with_experimental_phase_c_boundary_package(
            x,
            experimental.boundary_term,
            experimental.easy_width,
            experimental.hard_width,
        ),
    }
}

pub fn compare_phase_c_package_vs_linked_candidate(x: u128) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_phase_c_term_band_package(x),
        experimental: analyze_with_phase_c_package(x),
    }
}

pub fn compare_phase_c_linked_variants(
    x: u128,
    current: crate::parameters::PhaseCLinkedCandidate,
    experimental: crate::parameters::PhaseCLinkedCandidate,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_experimental_phase_c_linked_package(
            x,
            current.easy_width,
            current.easy_min_term_floor,
            current.hard_width,
        ),
        experimental: analyze_with_experimental_phase_c_linked_package(
            x,
            experimental.easy_width,
            experimental.easy_min_term_floor,
            experimental.hard_width,
        ),
    }
}

pub fn compare_current_vs_experimental_easy_term_band(
    x: u128,
    hard_leaf_term_max: u128,
    easy_leaf_term_max: u128,
    min_term: u128,
    max_term: u128,
) -> AnalysisComparison {
    AnalysisComparison {
        current: analyze_with_term_frontiers(x, hard_leaf_term_max, easy_leaf_term_max),
        experimental: analyze_with_experimental_easy_term_band(
            x,
            hard_leaf_term_max,
            min_term,
            max_term,
        ),
    }
}

pub fn analyze(x: u128) -> DrAnalysis {
    analyze_with_term_frontiers(
        x,
        Parameters::DEFAULT_HARD_LEAF_TERM_MAX,
        Parameters::DEFAULT_EASY_LEAF_TERM_VALUE,
    )
}

pub fn analyze_with_hard_leaf_term_max(x: u128, hard_leaf_term_max: u128) -> DrAnalysis {
    analyze_with_term_frontiers(
        x,
        hard_leaf_term_max,
        Parameters::DEFAULT_EASY_LEAF_TERM_VALUE,
    )
}

pub fn analyze_with_term_frontiers(
    x: u128,
    hard_leaf_term_max: u128,
    easy_leaf_term_max: u128,
) -> DrAnalysis {
    let ctx = prepare_context_with_term_frontiers(x, hard_leaf_term_max, easy_leaf_term_max);
    let domain_set: DomainSet = ctx.domain_set();
    let domains = domain_set.domains;
    let contributions = domain_set.contributions;
    let result = domain_set.result;
    let hard_rule = ctx.s2_hard_domain().rule;
    let easy_rule = ctx.s2_easy_domain().rule;
    let (easy_leaf_term_min, easy_leaf_term_max) = easy_rule.term_bounds();

    DrAnalysis {
        alpha: ctx.params.tuning.alpha,
        phi_x_a: ctx.phi_x_a,
        a: ctx.a,
        easy_candidate_family: None,
        easy_candidate_width: None,
        easy_candidate_floor: None,
        easy_candidate_term_min: None,
        easy_candidate_term_max: None,
        s1_term_min_exclusive: ctx.s1_domain().rule.term_min_exclusive,
        hard_leaf_term_max: ctx.params.hard_leaf_term_max,
        hard_leaf_term_min: hard_rule.term_min(),
        hard_rule_kind: hard_rule.kind_name(),
        hard_rule_alpha: hard_rule.alpha(),
        easy_leaf_term_value: easy_rule.representative_term_value(),
        easy_leaf_term_min,
        easy_leaf_term_max,
        easy_rule_kind: easy_rule.kind_name(),
        easy_rule_alpha: easy_rule.alpha(),
        active_len: domains.active.len(),
        s1_len: domains.s1.leaves.len(),
        s2_easy_len: domains.s2_easy.leaves.len(),
        s2_hard_len: domains.s2_hard.leaves.len(),
        s2_trivial_is_zero: domains.s2_trivial.leaves.s3_is_zero,
        contributions,
        result,
    }
}

pub fn analyze_easy_specialized(x: u128) -> EasySpecializedAnalysis {
    let ctx = prepare_context(x);
    let boundary_candidate = crate::parameters::PhaseCBoundaryCandidate {
        boundary_term: 4,
        easy_width: 4,
        hard_width: 2,
    };
    let frontiers = boundary_candidate.build_frontier_set(ctx.params.tuning.alpha);
    let base_domains = ctx.dr_domains_with_frontiers(frontiers);
    let base_easy_start = base_domains.s2_easy.leaves.start;

    let q_easy_ref = if ctx.b > 0 {
        ctx.quotient_at(ctx.b - 1).unwrap_or(0)
    } else {
        0
    };
    let q_easy_prev = if ctx.b >= 2 {
        ctx.quotient_at(ctx.b - 2).unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let q_step = q_easy_prev.saturating_sub(q_easy_ref).max(1);
    let q_limit = q_easy_ref.saturating_add(q_step);

    let mut easy_start = ctx.b;
    while easy_start > base_easy_start {
        let index = easy_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= q_limit {
            easy_start -= 1;
        } else {
            break;
        }
    }

    let easy = crate::parameters::IndexDomain {
        start: easy_start,
        end: ctx.b,
    };
    let window = easy::easy_specialized_window_in_range(&ctx, easy);

    EasySpecializedAnalysis {
        easy_len: easy.len(),
        residual_len: window.residual.len(),
        transition_len: window.transition.len(),
        specialized_len: window.specialized.len(),
        easy_sum: (easy.start..easy.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("easy-specialized index must be valid")
            })
            .sum(),
        residual_sum: (window.residual.start..window.residual.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("easy-specialized residual index must be valid")
            })
            .sum(),
        transition_sum: (window.transition.start..window.transition.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("easy-specialized transition index must be valid")
            })
            .sum(),
        specialized_sum: (window.specialized.start..window.specialized.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("easy-specialized specialized index must be valid")
            })
            .sum(),
        q_ref: window.q_ref,
        q_step,
        first_specialized_q: if window.specialized.is_empty() {
            None
        } else {
            ctx.quotient_at(window.specialized.start)
        },
        last_specialized_q: if window.specialized.is_empty() {
            None
        } else {
            ctx.quotient_at(window.specialized.end - 1)
        },
    }
}

pub fn analyze_hard_specialized(x: u128) -> HardSpecializedAnalysis {
    let ctx = prepare_context(x);
    let boundary_candidate = crate::parameters::PhaseCBoundaryCandidate {
        boundary_term: 4,
        easy_width: 4,
        hard_width: 2,
    };
    let frontiers = boundary_candidate.build_frontier_set(ctx.params.tuning.alpha);
    let base_domains = ctx.dr_domains_with_frontiers(frontiers);
    let base_easy_start = base_domains.s2_easy.leaves.start;
    let base_hard_start = base_domains.s2_hard.leaves.start;

    let q_easy_ref = if ctx.b > 0 {
        ctx.quotient_at(ctx.b - 1).unwrap_or(0)
    } else {
        0
    };
    let q_easy_prev = if ctx.b >= 2 {
        ctx.quotient_at(ctx.b - 2).unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let easy_q_step = q_easy_prev.saturating_sub(q_easy_ref);

    let q_hard_ref = if base_domains.s2_hard.leaves.end > base_hard_start {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 1)
            .unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let q_hard_prev = if base_domains.s2_hard.leaves.end >= base_hard_start + 2 {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 2)
            .unwrap_or(q_hard_ref)
    } else {
        q_hard_ref
    };
    let hard_q_step = q_hard_prev.saturating_sub(q_hard_ref).max(1);

    let easy_q_limit = q_easy_ref.saturating_add(easy_q_step);
    let hard_q_limit = q_hard_ref.saturating_add(hard_q_step);

    let mut filtered_easy_start = ctx.b;
    while filtered_easy_start > base_easy_start {
        let index = filtered_easy_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= easy_q_limit {
            filtered_easy_start -= 1;
        } else {
            break;
        }
    }

    let mut filtered_hard_start = filtered_easy_start;
    while filtered_hard_start > base_hard_start {
        let index = filtered_hard_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= hard_q_limit {
            filtered_hard_start -= 1;
        } else {
            break;
        }
    }

    let hard = crate::parameters::IndexDomain {
        start: filtered_hard_start,
        end: filtered_easy_start,
    };
    let window = hard::hard_specialized_window_in_range(&ctx, hard);

    HardSpecializedAnalysis {
        hard_len: hard.len(),
        residual_len: window.residual.len(),
        transition_len: window.transition.len(),
        specialized_len: window.specialized.len(),
        hard_sum: (hard.start..hard.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("hard-specialized index must be valid")
            })
            .sum(),
        residual_sum: (window.residual.start..window.residual.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("hard-specialized residual index must be valid")
            })
            .sum(),
        transition_sum: (window.transition.start..window.transition.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("hard-specialized transition index must be valid")
            })
            .sum(),
        specialized_sum: (window.specialized.start..window.specialized.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("hard-specialized specialized index must be valid")
            })
            .sum(),
        q_ref: window.q_ref,
        q_step: hard_q_step,
        first_specialized_q: if window.specialized.is_empty() {
            None
        } else {
            ctx.quotient_at(window.specialized.start)
        },
        last_specialized_q: if window.specialized.is_empty() {
            None
        } else {
            ctx.quotient_at(window.specialized.end - 1)
        },
    }
}

pub fn analyze_ordinary_specialized(x: u128) -> OrdinarySpecializedAnalysis {
    let ctx = prepare_context(x);
    let boundary_candidate = crate::parameters::PhaseCBoundaryCandidate {
        boundary_term: 4,
        easy_width: 4,
        hard_width: 2,
    };
    let frontiers = boundary_candidate.build_frontier_set(ctx.params.tuning.alpha);
    let base_domains = ctx.dr_domains_with_frontiers(frontiers);
    let base_easy_start = base_domains.s2_easy.leaves.start;
    let base_hard_start = base_domains.s2_hard.leaves.start;

    let q_easy_ref = if ctx.b > 0 {
        ctx.quotient_at(ctx.b - 1).unwrap_or(0)
    } else {
        0
    };
    let q_easy_prev = if ctx.b >= 2 {
        ctx.quotient_at(ctx.b - 2).unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let easy_q_step = q_easy_prev.saturating_sub(q_easy_ref);

    let q_hard_ref = if base_domains.s2_hard.leaves.end > base_hard_start {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 1)
            .unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let q_hard_prev = if base_domains.s2_hard.leaves.end >= base_hard_start + 2 {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 2)
            .unwrap_or(q_hard_ref)
    } else {
        q_hard_ref
    };
    let hard_q_step = q_hard_prev.saturating_sub(q_hard_ref).max(1);

    let easy_q_limit = q_easy_ref.saturating_add(easy_q_step);
    let hard_q_limit = q_hard_ref.saturating_add(hard_q_step);

    let mut filtered_easy_start = ctx.b;
    while filtered_easy_start > base_easy_start {
        let index = filtered_easy_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= easy_q_limit {
            filtered_easy_start -= 1;
        } else {
            break;
        }
    }

    let mut filtered_hard_start = filtered_easy_start;
    while filtered_hard_start > base_hard_start {
        let index = filtered_hard_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= hard_q_limit {
            filtered_hard_start -= 1;
        } else {
            break;
        }
    }

    let ordinary = crate::parameters::IndexDomain {
        start: base_domains.active.start,
        end: filtered_hard_start,
    };
    let region = ordinary::ordinary_specialized_region_in_range(&ctx, ordinary);

    OrdinarySpecializedAnalysis {
        ordinary_len: ordinary.len(),
        residual_len: region.residual.len(),
        pretransition_len: region.pretransition.len(),
        transition_len: region.transition.len(),
        specialized_len: region.specialized.len(),
        ordinary_sum: (ordinary.start..ordinary.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-specialized index must be valid")
            })
            .sum(),
        residual_sum: (region.residual.start..region.residual.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-specialized residual index must be valid")
            })
            .sum(),
        pretransition_sum: (region.pretransition.start..region.pretransition.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-specialized pretransition index must be valid")
            })
            .sum(),
        transition_sum: (region.transition.start..region.transition.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-specialized transition index must be valid")
            })
            .sum(),
        specialized_sum: (region.specialized.start..region.specialized.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-specialized specialized index must be valid")
            })
            .sum(),
        q_ref: region.q_ref,
        q_step: region.q_step,
        first_specialized_q: if region.specialized.is_empty() {
            None
        } else {
            ctx.quotient_at(region.specialized.start)
        },
        last_specialized_q: if region.specialized.is_empty() {
            None
        } else {
            ctx.quotient_at(region.specialized.end - 1)
        },
    }
}

pub fn analyze_ordinary_relative_quotient_region(x: u128) -> OrdinaryRelativeQuotientAnalysis {
    analyze_ordinary_relative_quotient_region_variant(x, 0, 1)
}

pub fn analyze_ordinary_relative_quotient_region_variant(
    x: u128,
    center_shift: isize,
    step_scale: u128,
) -> OrdinaryRelativeQuotientAnalysis {
    let ctx = prepare_context(x);
    let boundary_candidate = crate::parameters::PhaseCBoundaryCandidate {
        boundary_term: 4,
        easy_width: 4,
        hard_width: 2,
    };
    let frontiers = boundary_candidate.build_frontier_set(ctx.params.tuning.alpha);
    let base_domains = ctx.dr_domains_with_frontiers(frontiers);
    let base_easy_start = base_domains.s2_easy.leaves.start;
    let base_hard_start = base_domains.s2_hard.leaves.start;

    let q_easy_ref = if ctx.b > 0 {
        ctx.quotient_at(ctx.b - 1).unwrap_or(0)
    } else {
        0
    };
    let q_easy_prev = if ctx.b >= 2 {
        ctx.quotient_at(ctx.b - 2).unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let easy_q_step = q_easy_prev.saturating_sub(q_easy_ref);

    let q_hard_ref = if base_domains.s2_hard.leaves.end > base_hard_start {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 1)
            .unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let q_hard_prev = if base_domains.s2_hard.leaves.end >= base_hard_start + 2 {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 2)
            .unwrap_or(q_hard_ref)
    } else {
        q_hard_ref
    };
    let hard_q_step = q_hard_prev.saturating_sub(q_hard_ref).max(1);

    let easy_q_limit = q_easy_ref.saturating_add(easy_q_step);
    let hard_q_limit = q_hard_ref.saturating_add(hard_q_step);

    let mut filtered_easy_start = ctx.b;
    while filtered_easy_start > base_easy_start {
        let index = filtered_easy_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= easy_q_limit {
            filtered_easy_start -= 1;
        } else {
            break;
        }
    }

    let mut filtered_hard_start = filtered_easy_start;
    while filtered_hard_start > base_hard_start {
        let index = filtered_hard_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= hard_q_limit {
            filtered_hard_start -= 1;
        } else {
            break;
        }
    }

    let ordinary = crate::parameters::IndexDomain {
        start: base_domains.active.start,
        end: filtered_hard_start,
    };
    let region = ordinary::ordinary_relative_quotient_region_with_params_in_range(
        &ctx,
        ordinary,
        center_shift,
        step_scale,
    );

    OrdinaryRelativeQuotientAnalysis {
        ordinary_len: ordinary.len(),
        left_residual_len: region.left_residual.len(),
        region_len: region.region.len(),
        right_residual_len: region.right_residual.len(),
        ordinary_sum: (ordinary.start..ordinary.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-relative index must be valid")
            })
            .sum(),
        left_residual_sum: (region.left_residual.start..region.left_residual.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-relative left residual index must be valid")
            })
            .sum(),
        region_sum: (region.region.start..region.region.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-relative region index must be valid")
            })
            .sum(),
        right_residual_sum: (region.right_residual.start..region.right_residual.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-relative right residual index must be valid")
            })
            .sum(),
        q_ref: region.q_ref,
        q_step: region.q_step,
        first_region_q: if region.region.is_empty() {
            None
        } else {
            ctx.quotient_at(region.region.start)
        },
        last_region_q: if region.region.is_empty() {
            None
        } else {
            ctx.quotient_at(region.region.end - 1)
        },
    }
}

pub fn analyze_ordinary_relative_quotient_shoulder_region(
    x: u128,
) -> OrdinaryRelativeQuotientShoulderAnalysis {
    analyze_ordinary_relative_quotient_shoulder_region_variant(x, 0, 1, 2)
}

pub fn analyze_ordinary_relative_quotient_shoulder_region_variant(
    x: u128,
    center_shift: isize,
    core_step_scale: u128,
    shoulder_step_scale: u128,
) -> OrdinaryRelativeQuotientShoulderAnalysis {
    let ctx = prepare_context(x);
    let boundary_candidate = crate::parameters::PhaseCBoundaryCandidate {
        boundary_term: 4,
        easy_width: 4,
        hard_width: 2,
    };
    let frontiers = boundary_candidate.build_frontier_set(ctx.params.tuning.alpha);
    let base_domains = ctx.dr_domains_with_frontiers(frontiers);
    let base_easy_start = base_domains.s2_easy.leaves.start;
    let base_hard_start = base_domains.s2_hard.leaves.start;

    let q_easy_ref = if ctx.b > 0 {
        ctx.quotient_at(ctx.b - 1).unwrap_or(0)
    } else {
        0
    };
    let q_easy_prev = if ctx.b >= 2 {
        ctx.quotient_at(ctx.b - 2).unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let easy_q_step = q_easy_prev.saturating_sub(q_easy_ref);

    let q_hard_ref = if base_domains.s2_hard.leaves.end > base_hard_start {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 1)
            .unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let q_hard_prev = if base_domains.s2_hard.leaves.end >= base_hard_start + 2 {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 2)
            .unwrap_or(q_hard_ref)
    } else {
        q_hard_ref
    };
    let hard_q_step = q_hard_prev.saturating_sub(q_hard_ref).max(1);

    let easy_q_limit = q_easy_ref.saturating_add(easy_q_step);
    let hard_q_limit = q_hard_ref.saturating_add(hard_q_step);

    let mut filtered_easy_start = ctx.b;
    while filtered_easy_start > base_easy_start {
        let index = filtered_easy_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= easy_q_limit {
            filtered_easy_start -= 1;
        } else {
            break;
        }
    }

    let mut filtered_hard_start = filtered_easy_start;
    while filtered_hard_start > base_hard_start {
        let index = filtered_hard_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= hard_q_limit {
            filtered_hard_start -= 1;
        } else {
            break;
        }
    }

    let ordinary = crate::parameters::IndexDomain {
        start: base_domains.active.start,
        end: filtered_hard_start,
    };
    let region = ordinary::ordinary_relative_quotient_shoulder_region_with_params_in_range(
        &ctx,
        ordinary,
        center_shift,
        core_step_scale,
        shoulder_step_scale,
    );

    OrdinaryRelativeQuotientShoulderAnalysis {
        ordinary_len: ordinary.len(),
        left_residual_len: region.left_residual.len(),
        left_shoulder_len: region.left_shoulder.len(),
        core_len: region.core.len(),
        right_shoulder_len: region.right_shoulder.len(),
        right_residual_len: region.right_residual.len(),
        ordinary_sum: (ordinary.start..ordinary.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-shoulder index must be valid")
            })
            .sum(),
        left_residual_sum: (region.left_residual.start..region.left_residual.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-shoulder left residual index must be valid")
            })
            .sum(),
        left_shoulder_sum: (region.left_shoulder.start..region.left_shoulder.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-shoulder left shoulder index must be valid")
            })
            .sum(),
        core_sum: (region.core.start..region.core.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-shoulder core index must be valid")
            })
            .sum(),
        right_shoulder_sum: (region.right_shoulder.start..region.right_shoulder.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-shoulder right shoulder index must be valid")
            })
            .sum(),
        right_residual_sum: (region.right_residual.start..region.right_residual.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-shoulder right residual index must be valid")
            })
            .sum(),
        q_ref: region.q_ref,
        q_step: region.q_step,
    }
}

pub fn analyze_ordinary_relative_quotient_envelope_region(
    x: u128,
) -> OrdinaryRelativeQuotientEnvelopeAnalysis {
    analyze_ordinary_relative_quotient_envelope_region_variant(x, 0, 1, 3)
}

pub fn analyze_ordinary_relative_quotient_envelope_region_variant(
    x: u128,
    center_shift: isize,
    core_step_scale: u128,
    envelope_step_scale: u128,
) -> OrdinaryRelativeQuotientEnvelopeAnalysis {
    let ctx = prepare_context(x);
    let boundary_candidate = crate::parameters::PhaseCBoundaryCandidate {
        boundary_term: 4,
        easy_width: 4,
        hard_width: 2,
    };
    let frontiers = boundary_candidate.build_frontier_set(ctx.params.tuning.alpha);
    let base_domains = ctx.dr_domains_with_frontiers(frontiers);
    let base_easy_start = base_domains.s2_easy.leaves.start;
    let base_hard_start = base_domains.s2_hard.leaves.start;

    let q_easy_ref = if ctx.b > 0 {
        ctx.quotient_at(ctx.b - 1).unwrap_or(0)
    } else {
        0
    };
    let q_easy_prev = if ctx.b >= 2 {
        ctx.quotient_at(ctx.b - 2).unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let easy_q_step = q_easy_prev.saturating_sub(q_easy_ref);

    let q_hard_ref = if base_domains.s2_hard.leaves.end > base_hard_start {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 1)
            .unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let q_hard_prev = if base_domains.s2_hard.leaves.end >= base_hard_start + 2 {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 2)
            .unwrap_or(q_hard_ref)
    } else {
        q_hard_ref
    };
    let hard_q_step = q_hard_prev.saturating_sub(q_hard_ref).max(1);

    let easy_q_limit = q_easy_ref.saturating_add(easy_q_step);
    let hard_q_limit = q_hard_ref.saturating_add(hard_q_step);

    let mut filtered_easy_start = ctx.b;
    while filtered_easy_start > base_easy_start {
        let index = filtered_easy_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= easy_q_limit {
            filtered_easy_start -= 1;
        } else {
            break;
        }
    }

    let mut filtered_hard_start = filtered_easy_start;
    while filtered_hard_start > base_hard_start {
        let index = filtered_hard_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= hard_q_limit {
            filtered_hard_start -= 1;
        } else {
            break;
        }
    }

    let ordinary = crate::parameters::IndexDomain {
        start: base_domains.active.start,
        end: filtered_hard_start,
    };
    let region = ordinary::ordinary_relative_quotient_envelope_region_with_params_in_range(
        &ctx,
        ordinary,
        center_shift,
        core_step_scale,
        envelope_step_scale,
    );

    OrdinaryRelativeQuotientEnvelopeAnalysis {
        ordinary_len: ordinary.len(),
        left_residual_len: region.left_residual.len(),
        left_envelope_len: region.left_envelope.len(),
        core_len: region.core.len(),
        right_envelope_len: region.right_envelope.len(),
        right_residual_len: region.right_residual.len(),
        ordinary_sum: (ordinary.start..ordinary.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-envelope index must be valid")
            })
            .sum(),
        left_residual_sum: (region.left_residual.start..region.left_residual.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-envelope left residual index must be valid")
            })
            .sum(),
        left_envelope_sum: (region.left_envelope.start..region.left_envelope.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-envelope left envelope index must be valid")
            })
            .sum(),
        core_sum: (region.core.start..region.core.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-envelope core index must be valid")
            })
            .sum(),
        right_envelope_sum: (region.right_envelope.start..region.right_envelope.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-envelope right envelope index must be valid")
            })
            .sum(),
        right_residual_sum: (region.right_residual.start..region.right_residual.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-envelope right residual index must be valid")
            })
            .sum(),
        q_ref: region.q_ref,
        q_step: region.q_step,
    }
}

pub fn analyze_ordinary_relative_quotient_hierarchy_region(
    x: u128,
) -> OrdinaryRelativeQuotientHierarchyAnalysis {
    let ctx = prepare_context(x);
    let boundary_candidate = crate::parameters::PhaseCBoundaryCandidate {
        boundary_term: 4,
        easy_width: 4,
        hard_width: 2,
    };
    let frontiers = boundary_candidate.build_frontier_set(ctx.params.tuning.alpha);
    let base_domains = ctx.dr_domains_with_frontiers(frontiers);
    let base_easy_start = base_domains.s2_easy.leaves.start;
    let base_hard_start = base_domains.s2_hard.leaves.start;

    let q_easy_ref = if ctx.b > 0 {
        ctx.quotient_at(ctx.b - 1).unwrap_or(0)
    } else {
        0
    };
    let q_easy_prev = if ctx.b >= 2 {
        ctx.quotient_at(ctx.b - 2).unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let easy_q_step = q_easy_prev.saturating_sub(q_easy_ref);

    let q_hard_ref = if base_domains.s2_hard.leaves.end > base_hard_start {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 1)
            .unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let q_hard_prev = if base_domains.s2_hard.leaves.end >= base_hard_start + 2 {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 2)
            .unwrap_or(q_hard_ref)
    } else {
        q_hard_ref
    };
    let hard_q_step = q_hard_prev.saturating_sub(q_hard_ref).max(1);

    let easy_q_limit = q_easy_ref.saturating_add(easy_q_step);
    let hard_q_limit = q_hard_ref.saturating_add(hard_q_step);

    let mut filtered_easy_start = ctx.b;
    while filtered_easy_start > base_easy_start {
        let index = filtered_easy_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= easy_q_limit {
            filtered_easy_start -= 1;
        } else {
            break;
        }
    }

    let mut filtered_hard_start = filtered_easy_start;
    while filtered_hard_start > base_hard_start {
        let index = filtered_hard_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= hard_q_limit {
            filtered_hard_start -= 1;
        } else {
            break;
        }
    }

    let ordinary = crate::parameters::IndexDomain {
        start: base_domains.active.start,
        end: filtered_hard_start,
    };
    let region = ordinary::ordinary_relative_quotient_hierarchy_region_in_range(&ctx, ordinary);

    OrdinaryRelativeQuotientHierarchyAnalysis {
        ordinary_len: ordinary.len(),
        left_residual_len: region.left_residual.len(),
        left_outer_band_len: region.left_outer_band.len(),
        left_near_band_len: region.left_near_band.len(),
        inner_core_len: region.inner_core.len(),
        right_near_band_len: region.right_near_band.len(),
        right_outer_band_len: region.right_outer_band.len(),
        right_residual_len: region.right_residual.len(),
        ordinary_sum: (ordinary.start..ordinary.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-hierarchy index must be valid")
            })
            .sum(),
        left_residual_sum: (region.left_residual.start..region.left_residual.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("left residual index must be valid")
            })
            .sum(),
        left_outer_band_sum: (region.left_outer_band.start..region.left_outer_band.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("left outer band index must be valid")
            })
            .sum(),
        left_near_band_sum: (region.left_near_band.start..region.left_near_band.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("left near band index must be valid")
            })
            .sum(),
        inner_core_sum: (region.inner_core.start..region.inner_core.end)
            .map(|j| ctx.s2_term_at(j).expect("inner core index must be valid"))
            .sum(),
        right_near_band_sum: (region.right_near_band.start..region.right_near_band.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("right near band index must be valid")
            })
            .sum(),
        right_outer_band_sum: (region.right_outer_band.start..region.right_outer_band.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("right outer band index must be valid")
            })
            .sum(),
        right_residual_sum: (region.right_residual.start..region.right_residual.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("right residual index must be valid")
            })
            .sum(),
        q_ref: region.q_ref,
        q_step: region.q_step,
    }
}

pub fn analyze_ordinary_region_assembly(x: u128) -> OrdinaryRegionAssemblyAnalysis {
    let ctx = prepare_context(x);
    let boundary_candidate = crate::parameters::PhaseCBoundaryCandidate {
        boundary_term: 4,
        easy_width: 4,
        hard_width: 2,
    };
    let frontiers = boundary_candidate.build_frontier_set(ctx.params.tuning.alpha);
    let base_domains = ctx.dr_domains_with_frontiers(frontiers);
    let base_easy_start = base_domains.s2_easy.leaves.start;
    let base_hard_start = base_domains.s2_hard.leaves.start;

    let q_easy_ref = if ctx.b > 0 {
        ctx.quotient_at(ctx.b - 1).unwrap_or(0)
    } else {
        0
    };
    let q_easy_prev = if ctx.b >= 2 {
        ctx.quotient_at(ctx.b - 2).unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let easy_q_step = q_easy_prev.saturating_sub(q_easy_ref);

    let q_hard_ref = if base_domains.s2_hard.leaves.end > base_hard_start {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 1)
            .unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let q_hard_prev = if base_domains.s2_hard.leaves.end >= base_hard_start + 2 {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 2)
            .unwrap_or(q_hard_ref)
    } else {
        q_hard_ref
    };
    let hard_q_step = q_hard_prev.saturating_sub(q_hard_ref).max(1);

    let easy_q_limit = q_easy_ref.saturating_add(easy_q_step);
    let hard_q_limit = q_hard_ref.saturating_add(hard_q_step);

    let mut filtered_easy_start = ctx.b;
    while filtered_easy_start > base_easy_start {
        let index = filtered_easy_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= easy_q_limit {
            filtered_easy_start -= 1;
        } else {
            break;
        }
    }

    let mut filtered_hard_start = filtered_easy_start;
    while filtered_hard_start > base_hard_start {
        let index = filtered_hard_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= hard_q_limit {
            filtered_hard_start -= 1;
        } else {
            break;
        }
    }

    let ordinary = crate::parameters::IndexDomain {
        start: base_domains.active.start,
        end: filtered_hard_start,
    };
    let region = ordinary::ordinary_region_assembly_in_range(&ctx, ordinary);

    OrdinaryRegionAssemblyAnalysis {
        ordinary_len: ordinary.len(),
        left_outer_support_len: region.left_outer_support.len(),
        left_adjacent_support_len: region.left_adjacent_support.len(),
        central_assembly_len: region.central_assembly.len(),
        right_adjacent_support_len: region.right_adjacent_support.len(),
        right_outer_support_len: region.right_outer_support.len(),
        ordinary_sum: (ordinary.start..ordinary.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-assembly index must be valid")
            })
            .sum(),
        left_outer_support_sum: (region.left_outer_support.start..region.left_outer_support.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("left outer support index must be valid")
            })
            .sum(),
        left_adjacent_support_sum: (region.left_adjacent_support.start
            ..region.left_adjacent_support.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("left adjacent support index must be valid")
            })
            .sum(),
        central_assembly_sum: (region.central_assembly.start..region.central_assembly.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("central assembly index must be valid")
            })
            .sum(),
        right_adjacent_support_sum: (region.right_adjacent_support.start
            ..region.right_adjacent_support.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("right adjacent support index must be valid")
            })
            .sum(),
        right_outer_support_sum: (region.right_outer_support.start..region.right_outer_support.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("right outer support index must be valid")
            })
            .sum(),
        q_ref: region.q_ref,
        q_step: region.q_step,
    }
}

pub fn analyze_ordinary_quasi_literature_region(x: u128) -> OrdinaryQuasiLiteratureAnalysis {
    let ctx = prepare_context(x);
    let boundary_candidate = crate::parameters::PhaseCBoundaryCandidate {
        boundary_term: 4,
        easy_width: 4,
        hard_width: 2,
    };
    let frontiers = boundary_candidate.build_frontier_set(ctx.params.tuning.alpha);
    let base_domains = ctx.dr_domains_with_frontiers(frontiers);
    let base_easy_start = base_domains.s2_easy.leaves.start;
    let base_hard_start = base_domains.s2_hard.leaves.start;

    let q_easy_ref = if ctx.b > 0 {
        ctx.quotient_at(ctx.b - 1).unwrap_or(0)
    } else {
        0
    };
    let q_easy_prev = if ctx.b >= 2 {
        ctx.quotient_at(ctx.b - 2).unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let easy_q_step = q_easy_prev.saturating_sub(q_easy_ref);

    let q_hard_ref = if base_domains.s2_hard.leaves.end > base_hard_start {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 1)
            .unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let q_hard_prev = if base_domains.s2_hard.leaves.end >= base_hard_start + 2 {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 2)
            .unwrap_or(q_hard_ref)
    } else {
        q_hard_ref
    };
    let hard_q_step = q_hard_prev.saturating_sub(q_hard_ref).max(1);

    let easy_q_limit = q_easy_ref.saturating_add(easy_q_step);
    let hard_q_limit = q_hard_ref.saturating_add(hard_q_step);

    let mut filtered_easy_start = ctx.b;
    while filtered_easy_start > base_easy_start {
        let index = filtered_easy_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= easy_q_limit {
            filtered_easy_start -= 1;
        } else {
            break;
        }
    }

    let mut filtered_hard_start = filtered_easy_start;
    while filtered_hard_start > base_hard_start {
        let index = filtered_hard_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= hard_q_limit {
            filtered_hard_start -= 1;
        } else {
            break;
        }
    }

    let ordinary = crate::parameters::IndexDomain {
        start: base_domains.active.start,
        end: filtered_hard_start,
    };
    let region = ordinary::ordinary_quasi_literature_region_in_range(&ctx, ordinary);

    OrdinaryQuasiLiteratureAnalysis {
        ordinary_len: ordinary.len(),
        left_outer_work_len: region.left_outer_work.len(),
        middle_work_len: region.middle_work.len(),
        right_outer_work_len: region.right_outer_work.len(),
        ordinary_sum: (ordinary.start..ordinary.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-quasi index must be valid")
            })
            .sum(),
        left_outer_work_sum: (region.left_outer_work.start..region.left_outer_work.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("left outer work index must be valid")
            })
            .sum(),
        middle_work_sum: (region.middle_work.start..region.middle_work.end)
            .map(|j| ctx.s2_term_at(j).expect("middle work index must be valid"))
            .sum(),
        right_outer_work_sum: (region.right_outer_work.start..region.right_outer_work.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("right outer work index must be valid")
            })
            .sum(),
        q_ref: region.q_ref,
        q_step: region.q_step,
    }
}

pub fn analyze_ordinary_dr_like_region(x: u128) -> OrdinaryDrLikeAnalysis {
    let ctx = prepare_context(x);
    let boundary_candidate = crate::parameters::PhaseCBoundaryCandidate {
        boundary_term: 4,
        easy_width: 4,
        hard_width: 2,
    };
    let frontiers = boundary_candidate.build_frontier_set(ctx.params.tuning.alpha);
    let base_domains = ctx.dr_domains_with_frontiers(frontiers);
    let base_easy_start = base_domains.s2_easy.leaves.start;
    let base_hard_start = base_domains.s2_hard.leaves.start;

    let q_easy_ref = if ctx.b > 0 {
        ctx.quotient_at(ctx.b - 1).unwrap_or(0)
    } else {
        0
    };
    let q_easy_prev = if ctx.b >= 2 {
        ctx.quotient_at(ctx.b - 2).unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let easy_q_step = q_easy_prev.saturating_sub(q_easy_ref);

    let q_hard_ref = if base_domains.s2_hard.leaves.end > base_hard_start {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 1)
            .unwrap_or(q_easy_ref)
    } else {
        q_easy_ref
    };
    let q_hard_prev = if base_domains.s2_hard.leaves.end >= base_hard_start + 2 {
        ctx.quotient_at(base_domains.s2_hard.leaves.end - 2)
            .unwrap_or(q_hard_ref)
    } else {
        q_hard_ref
    };
    let hard_q_step = q_hard_prev.saturating_sub(q_hard_ref).max(1);

    let easy_q_limit = q_easy_ref.saturating_add(easy_q_step);
    let hard_q_limit = q_hard_ref.saturating_add(hard_q_step);

    let mut filtered_easy_start = ctx.b;
    while filtered_easy_start > base_easy_start {
        let index = filtered_easy_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= easy_q_limit {
            filtered_easy_start -= 1;
        } else {
            break;
        }
    }

    let mut filtered_hard_start = filtered_easy_start;
    while filtered_hard_start > base_hard_start {
        let index = filtered_hard_start - 1;
        let Some(q) = ctx.quotient_at(index) else {
            break;
        };
        if q <= hard_q_limit {
            filtered_hard_start -= 1;
        } else {
            break;
        }
    }

    let ordinary = crate::parameters::IndexDomain {
        start: base_domains.active.start,
        end: filtered_hard_start,
    };
    let region = ordinary::ordinary_dr_like_region_in_range(&ctx, ordinary);

    OrdinaryDrLikeAnalysis {
        ordinary_len: ordinary.len(),
        left_outer_work_len: region.left_outer_work.len(),
        left_transfer_work_len: region.left_transfer_work.len(),
        central_work_region_len: region.central_work_region.len(),
        right_transfer_work_len: region.right_transfer_work.len(),
        right_outer_work_len: region.right_outer_work.len(),
        ordinary_sum: (ordinary.start..ordinary.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("ordinary-dr-like index must be valid")
            })
            .sum(),
        left_outer_work_sum: (region.left_outer_work.start..region.left_outer_work.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("left outer work index must be valid")
            })
            .sum(),
        left_transfer_work_sum: (region.left_transfer_work.start..region.left_transfer_work.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("left transfer work index must be valid")
            })
            .sum(),
        central_work_region_sum: (region.central_work_region.start..region.central_work_region.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("central work region index must be valid")
            })
            .sum(),
        right_transfer_work_sum: (region.right_transfer_work.start..region.right_transfer_work.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("right transfer work index must be valid")
            })
            .sum(),
        right_outer_work_sum: (region.right_outer_work.start..region.right_outer_work.end)
            .map(|j| {
                ctx.s2_term_at(j)
                    .expect("right outer work index must be valid")
            })
            .sum(),
        q_ref: region.q_ref,
        q_step: region.q_step,
    }
}

pub fn compare_ordinary_specialized_vs_relative_quotient_region(
    x: u128,
) -> OrdinaryRegionComparison {
    let current = analyze_ordinary_specialized(x);
    let relative = analyze_ordinary_relative_quotient_region(x);

    OrdinaryRegionComparison {
        ordinary_len: current.ordinary_len,
        current_terminal_len: current.pretransition_len
            + current.transition_len
            + current.specialized_len,
        relative_region_len: relative.region_len,
        current_terminal_sum: current.pretransition_sum
            + current.transition_sum
            + current.specialized_sum,
        relative_region_sum: relative.region_sum,
    }
}

pub fn analyze_ordinary_post_plateau_profile(x: u128) -> OrdinaryPostPlateauProfile {
    let current = analyze_ordinary_specialized(x);
    let relative = analyze_ordinary_relative_quotient_region(x);

    OrdinaryPostPlateauProfile {
        ordinary_len: current.ordinary_len,
        terminal_len: current.pretransition_len + current.transition_len + current.specialized_len,
        terminal_sum: current.pretransition_sum + current.transition_sum + current.specialized_sum,
        left_residual_len: relative.left_residual_len,
        region_len: relative.region_len,
        right_residual_len: relative.right_residual_len,
        left_residual_sum: relative.left_residual_sum,
        region_sum: relative.region_sum,
        right_residual_sum: relative.right_residual_sum,
        q_ref: relative.q_ref,
        q_step: relative.q_step,
    }
}

pub fn analyze_post_plateau_triptych(x: u128) -> PostPlateauTriptychAnalysis {
    let easy = analyze_easy_specialized(x);
    let ordinary = analyze_ordinary_post_plateau_profile(x);
    let assembly = analyze_ordinary_region_assembly(x);
    let quasi_literature = analyze_ordinary_quasi_literature_region(x);
    let hard = analyze_hard_specialized(x);

    PostPlateauTriptychAnalysis {
        easy_len: easy.easy_len,
        easy_focus_len: easy.transition_len + easy.specialized_len,
        easy_focus_sum: easy.transition_sum + easy.specialized_sum,
        ordinary_len: ordinary.ordinary_len,
        ordinary_terminal_len: ordinary.terminal_len,
        ordinary_terminal_sum: ordinary.terminal_sum,
        ordinary_region_len: ordinary.region_len,
        ordinary_region_sum: ordinary.region_sum,
        ordinary_assembly_core_len: assembly.central_assembly_len,
        ordinary_assembly_core_sum: assembly.central_assembly_sum,
        ordinary_assembly_support_len: assembly.left_outer_support_len
            + assembly.left_adjacent_support_len
            + assembly.right_adjacent_support_len
            + assembly.right_outer_support_len,
        ordinary_assembly_support_sum: assembly.left_outer_support_sum
            + assembly.left_adjacent_support_sum
            + assembly.right_adjacent_support_sum
            + assembly.right_outer_support_sum,
        ordinary_quasi_literature_middle_len: quasi_literature.middle_work_len,
        ordinary_quasi_literature_middle_sum: quasi_literature.middle_work_sum,
        ordinary_quasi_literature_outer_len: quasi_literature.left_outer_work_len
            + quasi_literature.right_outer_work_len,
        ordinary_quasi_literature_outer_sum: quasi_literature.left_outer_work_sum
            + quasi_literature.right_outer_work_sum,
        hard_len: hard.hard_len,
        hard_focus_len: hard.transition_len + hard.specialized_len,
        hard_focus_sum: hard.transition_sum + hard.specialized_sum,
    }
}

pub fn prime_pi(x: u128, threads: usize) -> u128 {
    if x < 2 {
        return 0;
    }

    let _ = threads;
    analyze(x).result
}

/// Computes π(x) using the Deléglise-Rivat BIT-based hard leaf algorithm.
///
/// Replaces the serial O(n^(3/4)) hard leaf loop with the O(x^(2/3)/log²x)
/// BIT + segmented sieve sweep implemented in [`hard::hard_leaves_bit`].
/// All other components (φ, ordinary, easy) still use Lucy-Hedgehog tables.
///
/// This is the entry point wired into [`crate::deleglise_rivat`].
pub fn prime_pi_dr(x: u128) -> u128 {
    profile_prime_pi_dr(x).result
}

pub fn profile_prime_pi_dr(x: u128) -> DrRuntimeProfile {
    profile_prime_pi_dr_with_backend(x, PhiBackend::Lucy)
}

pub fn prime_pi_dr_with_backend(x: u128, backend: PhiBackend) -> u128 {
    profile_prime_pi_dr_with_backend(x, backend).result
}

pub fn profile_prime_pi_dr_with_backend(x: u128, backend: PhiBackend) -> DrRuntimeProfile {
    if x < 2 {
        return DrRuntimeProfile {
            result: 0,
            phi_x_a: 0,
            a: 0,
            phi_time: Duration::ZERO,
            seed_primes_time: Duration::ZERO,
            sqrt_primes_time: Duration::ZERO,
            s2_time: Duration::ZERO,
            total_time: Duration::ZERO,
        };
    }

    let total_start = Instant::now();
    let sqrt_x = isqrt(x) as u64;

    let t = Instant::now();
    let computation = match backend {
        PhiBackend::Lucy => default_phi_computation(x),
        _ => phi_computation_with_backend(x, backend),
    };
    let phi_time = t.elapsed();

    let seed_primes = &computation.primes[..computation.a];
    let seed_primes_time = Duration::ZERO;
    let a = computation.a;

    let t = Instant::now();
    let all_primes = primes_up_to(sqrt_x, seed_primes);
    let sqrt_primes_time = t.elapsed();

    let s2_primes = &all_primes[a..];

    if s2_primes.is_empty() {
        return DrRuntimeProfile {
            result: computation.phi_x_a + a as u128 - 1,
            phi_x_a: computation.phi_x_a,
            a,
            phi_time,
            seed_primes_time,
            sqrt_primes_time,
            s2_time: Duration::ZERO,
            total_time: total_start.elapsed(),
        };
    }

    let t = Instant::now();
    let s2 = hard::s2_bit(x, a, s2_primes, seed_primes);
    let s2_time = t.elapsed();

    DrRuntimeProfile {
        result: computation.phi_x_a + a as u128 - 1 - s2,
        phi_x_a: computation.phi_x_a,
        a,
        phi_time,
        seed_primes_time,
        sqrt_primes_time,
        s2_time,
        total_time: total_start.elapsed(),
    }
}

/// Computes π(x) using early-stop Lucy φ + segmented-sieve S₂ with prefix-popcount.
///
/// Two improvements over `prime_pi_dr`:
///
/// 1. **Early-stop Lucy** ([`lucy_phi_early_stop`]): the Lucy loop exits after
///    processing all primes ≤ ∛x (~1 229 for x = 10¹²) instead of continuing
///    to √x (~78 498).  φ(x, a) and the seed primes are already fully valid at
///    that point.
///
/// 2. **Prefix-popcount S₂** ([`hard::s2_popcount`]): replaces the 2 MiB BIT
///    (rebuilt from scratch every block, causing L3 cache thrashing) with a
///    32 KiB prefix-count table derived from the 64 KiB sieve bitset.  Both
///    structures fit in L1 cache across all block iterations.
pub fn prime_pi_dr_v2(x: u128) -> u128 {
    use crate::math::isqrt;
    use crate::segment::primes_up_to;
    use crate::sieve::lucy_phi_early_stop;

    if x < 2 {
        return 0;
    }

    let (phi_x_a, a, seed_primes) = lucy_phi_early_stop(x);

    let sqrt_x = isqrt(x) as u64;
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];

    if s2_primes.is_empty() {
        return phi_x_a + a as u128 - 1;
    }

    let s2 = hard::s2_popcount_par(x, a, s2_primes, &seed_primes);
    phi_x_a + a as u128 - 1 - s2
}

/// Step-by-step timing breakdown for `prime_pi_dr_v2`.
pub fn prime_pi_dr_v2_timed(x: u128) -> (u128, [std::time::Duration; 4]) {
    use crate::math::isqrt;
    use crate::segment::primes_up_to;
    use crate::sieve::lucy_phi_early_stop;
    use std::time::Instant;

    let mut times = [std::time::Duration::ZERO; 4];
    if x < 2 {
        return (0, times);
    }

    let t = Instant::now();
    let (phi_x_a, a, seed_primes) = lucy_phi_early_stop(x);
    times[0] = t.elapsed(); // Lucy early-stop

    let t = Instant::now();
    let sqrt_x = isqrt(x) as u64;
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];
    times[1] = t.elapsed(); // primes_up_to

    if s2_primes.is_empty() {
        return (phi_x_a + a as u128 - 1, times);
    }

    let t = Instant::now();
    let s2 = hard::s2_popcount_par(x, a, s2_primes, &seed_primes);
    times[2] = t.elapsed(); // s2_popcount_par

    (phi_x_a + a as u128 - 1 - s2, times)
}

/// True Deléglise-Rivat: computes π(x) in O(x^(2/3) / log²x) without building
/// the √x-sized Lucy `large[]` table.
///
/// # Algorithm
/// 1. Sieve to ∛x → `small_pi` (π table) + `seed_primes`.
/// 2. Extend to √x → `s2_primes`.
/// 3. Dry-run the Meissel φ recursion to collect all `n ∈ (∛x, x^(2/3)]`
///    whose π value will be needed at leaf nodes.
/// 4. Single combined BIT sweep over `[∛x, x^(2/3)]`:
///    - accumulates S₂ for `s2_primes`;
///    - answers φ-leaf π queries collected in step 3.
/// 5. Actual Meissel φ recursion using the precomputed large-π values.
/// 6. π(x) = φ(x,a) + a − 1 − S₂.
pub fn prime_pi_dr_meissel(x: u128) -> u128 {
    use crate::math::{icbrt, isqrt};
    use crate::phi::{build_phi_table, collect_phi_leaf_queries, phi_fast};
    use crate::segment::primes_up_to;
    use crate::sieve::sieve_to;
    use std::collections::{HashMap, HashSet};

    if x < 2 {
        return 0;
    }

    // ── 1. Small sieve up to ∛x + full phi table ────────────────────────────
    let cbrt_x = icbrt(x);
    let (small_pi, seed_primes) = sieve_to(cbrt_x as u64);
    let a = seed_primes.len();
    let phi_table = build_phi_table(cbrt_x as u64, &seed_primes);

    // ── 2. Primes ≤ √x for S₂ ───────────────────────────────────────────────
    let sqrt_x = isqrt(x) as u64;
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];

    // ── 3. Collect Meissel φ leaf π-queries (dry run) ───────────────────────
    let cbrt_x_u128 = cbrt_x as u128;
    let mut phi_query_set: HashSet<u128> = HashSet::new();
    collect_phi_leaf_queries(x, a, &seed_primes, cbrt_x_u128, &mut phi_query_set);
    let mut phi_queries: Vec<u128> = phi_query_set.into_iter().collect();
    phi_queries.sort_unstable();

    // ── 4. Combined BIT sweep ────────────────────────────────────────────────
    let (s2, large_pi) = hard::s2_and_phi_queries(x, a, s2_primes, &seed_primes, &phi_queries);

    // ── 5. Meissel φ(x, a) with phi_table for O(1) base cases ───────────────
    let mut memo: HashMap<(u128, usize), u128> = HashMap::new();
    let phi_x_a =
        phi_fast(x, a, &seed_primes, &phi_table, cbrt_x_u128, &small_pi, &large_pi, &mut memo);

    // ── 6. π(x) = φ(x,a) + a − 1 − S₂ ─────────────────────────────────────
    phi_x_a + a as u128 - 1 - s2
}

/// Profiling variant: returns (result, step timings).
pub fn prime_pi_dr_meissel_timed(x: u128) -> (u128, [std::time::Duration; 5]) {
    use crate::math::{icbrt, isqrt};
    use crate::phi::{build_phi_table, collect_phi_leaf_queries, phi_fast};
    use crate::segment::primes_up_to;
    use crate::sieve::sieve_to;
    use std::collections::{HashMap, HashSet};
    use std::time::Instant;

    let mut times = [std::time::Duration::ZERO; 5];

    if x < 2 {
        return (0, times);
    }

    let t0 = Instant::now();
    let cbrt_x = icbrt(x);
    let (small_pi, seed_primes) = sieve_to(cbrt_x as u64);
    let a = seed_primes.len();
    let phi_table = build_phi_table(cbrt_x as u64, &seed_primes);
    let sqrt_x = isqrt(x) as u64;
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];
    times[0] = t0.elapsed();

    let t1 = Instant::now();
    let cbrt_x_u128 = cbrt_x as u128;
    let mut phi_query_set: HashSet<u128> = HashSet::new();
    collect_phi_leaf_queries(x, a, &seed_primes, cbrt_x_u128, &mut phi_query_set);
    let phi_query_count = phi_query_set.len();
    let mut phi_queries: Vec<u128> = phi_query_set.into_iter().collect();
    phi_queries.sort_unstable();
    times[1] = t1.elapsed();

    eprintln!(
        "  [step2] phi_query_count = {phi_query_count}, a = {a}, sqrt_x = {sqrt_x}, cbrt_x = {cbrt_x}"
    );

    let t2 = Instant::now();
    let (s2, large_pi) = hard::s2_and_phi_queries(x, a, s2_primes, &seed_primes, &phi_queries);
    times[2] = t2.elapsed();

    let t3 = Instant::now();
    let mut memo: HashMap<(u128, usize), u128> = HashMap::new();
    let phi_x_a = phi_fast(
        x,
        a,
        &seed_primes,
        &phi_table,
        cbrt_x_u128,
        &small_pi,
        &large_pi,
        &mut memo,
    );
    times[3] = t3.elapsed();

    eprintln!("  [step4] memo entries = {}", memo.len());

    let result = phi_x_a + a as u128 - 1 - s2;
    times[4] = std::time::Duration::ZERO;

    (result, times)
}

// ─────────────────────────────────────────────────────────────────────────────
// prime_pi_dr_meissel_v2 — PhiCache Walisch
// ─────────────────────────────────────────────────────────────────────────────

/// π(x) via vrai Deléglise-Rivat avec cache dense Walisch pour φ(x,a).
///
/// Amélioration de `prime_pi_dr_meissel` : deux couches de cut-off pour φ :
/// - `phi_table` (n ≤ ∛x, a quelconque) — même que meissel_v1.
/// - `PhiCache`  (n ≤ x^(1/2.3), a ≤ 100) — nouveau : 20× plus de nœuds couverts.
///
/// Effets :
/// - `collect_phi_leaf_queries` collecte uniquement n > cache.max_x → 5-10× moins.
/// - mémo HashMap : seulement nœuds avec n > max_x ou (cbrt < n ≤ max_x et a > 100).
///
/// Algorithme :
/// 1. Crible jusqu'à ∛x → `seed_primes`, a = π(∛x) ; phi_table.
/// 2. Construit PhiCache (max_a=100, max_x=x^(1/2.3), ≤16 MB).
/// 3. Primes ≤ √x pour S₂.
/// 4. DFS pour collecter φ-leaf π-queries n > cache.max_x.
/// 5. Sweep BIT combiné S₂ + large_pi.
/// 6. φ(x, a) via `phi_cached` (PhiCache + phi_table + mémo réduit).
/// 7. π(x) = φ(x,a) + a − 1 − S₂.
pub fn prime_pi_dr_meissel_v2(x: u128) -> u128 {
    use crate::math::{icbrt, isqrt};
    use crate::phi::{build_phi_table, collect_phi_leaf_queries_cached, phi_cached, PhiMemoMap};
    use crate::phi_cache::PhiCache;
    use crate::segment::primes_up_to;
    use crate::sieve::sieve_to;
    use std::collections::HashSet;

    if x < 2 {
        return 0;
    }

    // ── 1. Crible jusqu'à ∛x + phi_table ────────────────────────────────────
    let cbrt_x = icbrt(x);
    let (_small_pi, seed_primes) = sieve_to(cbrt_x as u64);
    let a = seed_primes.len();
    let phi_table = build_phi_table(cbrt_x as u64, &seed_primes);
    let cbrt_x_u128 = cbrt_x as u128;

    // ── 2. PhiCache Walisch ──────────────────────────────────────────────────
    let cache = PhiCache::new(x, &seed_primes);
    let cache_cutoff = cache.max_x as u128;

    // ── 3. Primes ≤ √x pour S₂ ──────────────────────────────────────────────
    let sqrt_x = isqrt(x) as u64;
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];

    // ── 4. Collecte des φ-leaf π-queries n > cache.max_x ────────────────────
    // Cutoff = cache.max_x car phi_cached utilise cache.pi() pour n ≤ max_x.
    // Seuls les n > max_x nécessitent large_pi (sweep BIT).
    let mut phi_query_set: HashSet<u128> = HashSet::new();
    collect_phi_leaf_queries_cached(x, a, &seed_primes, cache_cutoff, &mut phi_query_set);
    let mut phi_queries: Vec<u128> = phi_query_set.into_iter().collect();
    phi_queries.sort_unstable();

    // ── 5. Sweep BIT combiné S₂ + large_pi ──────────────────────────────────
    let (s2, large_pi) = hard::s2_and_phi_queries(x, a, s2_primes, &seed_primes, &phi_queries);

    // ── 6. φ(x, a) via PhiCache + phi_table + mémo réduit ───────────────────
    let mut memo = PhiMemoMap::default();
    let phi_x_a =
        phi_cached(x, a, &seed_primes, &cache, &phi_table, cbrt_x_u128, &large_pi, &mut memo);

    // ── 7. π(x) = φ(x,a) + a − 1 − S₂ ─────────────────────────────────────
    phi_x_a + a as u128 - 1 - s2
}

/// Variante profilée de `prime_pi_dr_meissel_v2`.
/// Retourne (résultat, [step1, step2, step3, step4, step5]).
pub fn prime_pi_dr_meissel_v2_timed(x: u128) -> (u128, [std::time::Duration; 5]) {
    use crate::math::{icbrt, isqrt};
    use crate::phi::{build_phi_table, collect_phi_leaf_queries_cached, phi_cached, PhiMemoMap};
    use crate::phi_cache::PhiCache;
    use crate::segment::primes_up_to;
    use crate::sieve::sieve_to;
    use std::collections::HashSet;
    use std::time::Instant;

    let mut times = [std::time::Duration::ZERO; 5];

    if x < 2 {
        return (0, times);
    }

    let t0 = Instant::now();
    let cbrt_x = icbrt(x);
    let (_small_pi, seed_primes) = sieve_to(cbrt_x as u64);
    let a = seed_primes.len();
    let phi_table = build_phi_table(cbrt_x as u64, &seed_primes);
    let cbrt_x_u128 = cbrt_x as u128;
    let cache = PhiCache::new(x, &seed_primes);
    let cache_cutoff = cache.max_x as u128;
    let sqrt_x = isqrt(x) as u64;
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];
    times[0] = t0.elapsed();

    eprintln!(
        "  [step1] cbrt_x={cbrt_x}, a={a}, cache.max_x={}, sqrt_x={sqrt_x}",
        cache.max_x
    );

    let t1 = Instant::now();
    let mut phi_query_set: HashSet<u128> = HashSet::new();
    collect_phi_leaf_queries_cached(x, a, &seed_primes, cache_cutoff, &mut phi_query_set);
    let phi_query_count = phi_query_set.len();
    let mut phi_queries: Vec<u128> = phi_query_set.into_iter().collect();
    phi_queries.sort_unstable();
    times[1] = t1.elapsed();

    eprintln!("  [step2] phi_query_count={phi_query_count} (cache.max_x={}, cbrt_x={cbrt_x})", cache.max_x);

    let t2 = Instant::now();
    let (s2, large_pi) = hard::s2_and_phi_queries(x, a, s2_primes, &seed_primes, &phi_queries);
    times[2] = t2.elapsed();

    let t3 = Instant::now();
    let mut memo = PhiMemoMap::default();
    let phi_x_a =
        phi_cached(x, a, &seed_primes, &cache, &phi_table, cbrt_x_u128, &large_pi, &mut memo);
    times[3] = t3.elapsed();

    eprintln!("  [step4] memo.len()={}", memo.len());

    let result = phi_x_a + a as u128 - 1 - s2;
    times[4] = std::time::Duration::ZERO;

    (result, times)
}

/// Memory guard for `prime_pi_dr_v4`: maximum bytes for the flat phi memo.
///
/// At x = 10¹² the array is ~98 MB; at x = 10¹³ ~418 MB.
/// Falls back to `prime_pi_dr_v2` when the array would exceed this limit.
const V4_FLAT_MEMO_LIMIT: usize = 512 * 1024 * 1024; // 512 MiB

/// π(x) using flat-array φ memo + parallel prefix-popcount S₂.
///
/// This is "v4": replaces the HashMap memo in the Meissel φ recursion with a
/// flat `Vec<u64>` indexed by `d = ⌊x/n⌋`, eliminating 11 M HashMap lookups
/// at x = 10¹² and replacing them with direct array accesses (~5–50 ns each).
///
/// # Algorithm
/// 1. Sieve to ∛x → `seed_primes`, phi table for small n.
/// 2. Collect φ leaf queries using a flat-array visited bitset (fast DFS).
/// 3. Parallel prefix-popcount sweep (same as v2) → S₂ **and** large_pi.
/// 4. φ(x, a) via flat-array memo recursion.
/// 5. π(x) = φ(x, a) + a − 1 − S₂.
///
/// Falls back to `prime_pi_dr_v2` if the flat memo would exceed
/// [`V4_FLAT_MEMO_LIMIT`] bytes.
pub fn prime_pi_dr_v4(x: u128) -> u128 {
    use crate::math::{icbrt, isqrt};
    use crate::phi::{
        build_phi_table, collect_phi_leaf_queries_flat, phi_fast_flat,
    };
    use crate::segment::primes_up_to;
    use crate::sieve::sieve_to;
    use std::collections::{HashMap, HashSet};

    if x < 2 {
        return 0;
    }

    // ── 1. Sieve to ∛x ──────────────────────────────────────────────────────
    let cbrt_x = icbrt(x) as u128;
    let (small_pi, seed_primes) = sieve_to(cbrt_x as u64);
    let _ = small_pi; // not needed directly
    let a = seed_primes.len();
    let flat_stride = a + 1;

    // Memory guard: flat_memo size = cbrt_x * flat_stride * 8 bytes.
    let flat_memo_bytes = cbrt_x as usize * flat_stride * 8;
    if flat_memo_bytes > V4_FLAT_MEMO_LIMIT {
        return prime_pi_dr_v2(x);
    }

    let phi_table = build_phi_table(cbrt_x as u64, &seed_primes);

    // ── 2. Collect φ leaf queries (flat-array visited) ───────────────────────
    let mut phi_query_set: HashSet<u128> = HashSet::new();
    let mut visited_flat = vec![false; cbrt_x as usize * flat_stride];
    let mut visited_overflow: HashMap<(u128, usize), ()> = HashMap::new();
    collect_phi_leaf_queries_flat(
        x, a, 1, x, &seed_primes, cbrt_x,
        &mut phi_query_set,
        &mut visited_flat, flat_stride,
        &mut visited_overflow,
    );
    let mut phi_queries: Vec<u128> = phi_query_set.into_iter().collect();
    phi_queries.sort_unstable();

    // ── 3. Parallel S₂ + large_pi sweep ─────────────────────────────────────
    let sqrt_x = isqrt(x) as u64;
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];

    let (s2, large_pi) = if s2_primes.is_empty() {
        (0u128, HashMap::new())
    } else {
        hard::s2_popcount_par_with_phi(x, a, s2_primes, &seed_primes, &phi_queries)
    };

    // ── 4. φ(x, a) via flat-array memo ───────────────────────────────────────
    let mut flat_memo = vec![u64::MAX; cbrt_x as usize * flat_stride];
    let mut overflow_memo: HashMap<(u128, usize), u128> = HashMap::new();
    let phi_x_a = phi_fast_flat(
        x, a, 1, x, &seed_primes, &phi_table, cbrt_x, &large_pi,
        &mut flat_memo, flat_stride, &mut overflow_memo,
    );

    // ── 5. π(x) = φ(x,a) + a − 1 − S₂ ─────────────────────────────────────
    phi_x_a + a as u128 - 1 - s2
}

/// Step-by-step timing breakdown for `prime_pi_dr_v4`.
pub fn prime_pi_dr_v4_timed(x: u128) -> (u128, [std::time::Duration; 5]) {
    use crate::math::{icbrt, isqrt};
    use crate::phi::{
        build_phi_table, collect_phi_leaf_queries_flat, phi_fast_flat,
    };
    use crate::segment::primes_up_to;
    use crate::sieve::sieve_to;
    use std::collections::{HashMap, HashSet};
    use std::time::Instant;

    let mut times = [std::time::Duration::ZERO; 5];
    if x < 2 {
        return (0, times);
    }

    // ── 1. Sieve ─────────────────────────────────────────────────────────────
    let t = Instant::now();
    let cbrt_x = icbrt(x) as u128;
    let (small_pi, seed_primes) = sieve_to(cbrt_x as u64);
    let _ = small_pi;
    let a = seed_primes.len();
    let flat_stride = a + 1;
    let flat_memo_bytes = cbrt_x as usize * flat_stride * 8;
    if flat_memo_bytes > V4_FLAT_MEMO_LIMIT {
        eprintln!("  [v4] flat_memo would be {flat_memo_bytes} B — falling back to v2");
        let r = prime_pi_dr_v2(x);
        return (r, times);
    }
    let phi_table = build_phi_table(cbrt_x as u64, &seed_primes);
    let sqrt_x = isqrt(x) as u64;
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];
    times[0] = t.elapsed();

    // ── 2. Collect leaf queries ───────────────────────────────────────────────
    let t = Instant::now();
    let mut phi_query_set: HashSet<u128> = HashSet::new();
    let mut visited_flat = vec![false; cbrt_x as usize * flat_stride];
    let mut visited_overflow: HashMap<(u128, usize), ()> = HashMap::new();
    collect_phi_leaf_queries_flat(
        x, a, 1, x, &seed_primes, cbrt_x,
        &mut phi_query_set,
        &mut visited_flat, flat_stride,
        &mut visited_overflow,
    );
    let phi_query_count = phi_query_set.len();
    let overflow_visited = visited_overflow.len();
    let mut phi_queries: Vec<u128> = phi_query_set.into_iter().collect();
    phi_queries.sort_unstable();
    times[1] = t.elapsed();
    eprintln!(
        "  [v4/step2] phi_queries={phi_query_count}  overflow_visited={overflow_visited}  cbrt_x={cbrt_x}  a={a}"
    );

    // ── 3. Parallel S₂ + large_pi ───────────────────────────────────────────
    let t = Instant::now();
    let (s2, large_pi) = if s2_primes.is_empty() {
        (0u128, HashMap::new())
    } else {
        hard::s2_popcount_par_with_phi(x, a, s2_primes, &seed_primes, &phi_queries)
    };
    times[2] = t.elapsed();

    // ── 4. φ(x, a) flat memo ────────────────────────────────────────────────
    let t = Instant::now();
    let mut flat_memo = vec![u64::MAX; cbrt_x as usize * flat_stride];
    let mut overflow_memo: HashMap<(u128, usize), u128> = HashMap::new();
    let phi_x_a = phi_fast_flat(
        x, a, 1, x, &seed_primes, &phi_table, cbrt_x, &large_pi,
        &mut flat_memo, flat_stride, &mut overflow_memo,
    );
    times[3] = t.elapsed();
    eprintln!("  [v4/step4] overflow_memo={}", overflow_memo.len());

    // ── 5. Formula ───────────────────────────────────────────────────────────
    let result = phi_x_a + a as u128 - 1 - s2;
    times[4] = std::time::Duration::ZERO;
    (result, times)
}

/// Memory guard for `prime_pi_dr_v3`: maximum bytes allowed for `medium_pi`.
///
/// At x = 10¹² the array is ~400 MB; above ~2×10¹² it would exceed 512 MB.
/// When the array would be larger, we fall back to `prime_pi_dr_v2`.
const V3_MEDIUM_PI_LIMIT: usize = 512 * 1024 * 1024; // 512 MiB

/// Experimental: π(x) using dense medium_pi array for O(1) leaf π lookups.
///
/// # What this does
/// Replaces the `large_pi` HashMap in [`prime_pi_dr_meissel`] with a dense
/// `medium_pi[n − ∛x] = π(n)` array, built during a combined S₂+φ sieve
/// sweep from ∛x to x^(2/3) (≈2 extra SEG blocks vs the plain S₂ sweep).
/// Leaf queries are answered in O(1) (array index) instead of O(1) amortised
/// but 500 ns HashMap.
///
/// # Why it is NOT the default
/// The φ recursion (phi_fast_v3) visits ~2.25 M unique (n,a) interior nodes
/// at x = 10¹¹ and ~11 M at x = 10¹².  These all require memo HashMap lookups
/// at ~300–500 ns each.  The leaf-query saving (~70 K entries at x = 10¹¹,
/// ~700 K at x = 10¹²) is dwarfed by the interior-node memo cost.
///
/// Benchmark at x = 10¹²:
///   v2 (Lucy+pcnt)  699 ms   ← default
///   v3 (dense)      4.3 s    ← 6× slower
///
/// Fixing this requires an **iterative** φ computation that avoids the
/// recursive Legendre traversal altogether (Walisch-style sieve sweep).
///
/// Falls back to [`prime_pi_dr_v2`] when `x > ~2×10¹²` (medium_pi > 512 MiB).
pub fn prime_pi_dr_v3(x: u128) -> u128 {
    use crate::math::{icbrt, isqrt};
    use crate::phi::{build_phi_table, phi_fast_v3};
    use crate::segment::primes_up_to;
    use crate::sieve::sieve_to;
    use std::collections::HashMap;

    if x < 2 {
        return 0;
    }

    let cbrt_x = icbrt(x) as u64;
    let cbrt_u128 = cbrt_x as u128;

    // Estimate medium_pi size before allocating.
    let x_23_est = cbrt_u128.saturating_mul(cbrt_u128) + 4 * cbrt_u128;
    let medium_pi_bytes = (x_23_est - cbrt_u128 + 1) as usize * 4;
    if medium_pi_bytes > V3_MEDIUM_PI_LIMIT {
        return prime_pi_dr_v2(x);
    }

    // ── 1. Sieve to ∛x ───────────────────────────────────────────────────────
    let (_, seed_primes) = sieve_to(cbrt_x);
    let a = seed_primes.len();

    // ── 2. φ-table for O(1) base cases ───────────────────────────────────────
    let phi_table = build_phi_table(cbrt_x, &seed_primes);

    // ── 3. Primes in (∛x, √x] for S₂ ────────────────────────────────────────
    let sqrt_x = isqrt(x) as u64;
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];

    // ── 4. Combined S₂ + medium_pi sweep ────────────────────────────────────
    let (s2, medium_pi) = hard::s2_and_medium_pi(x, a, s2_primes, &seed_primes, cbrt_x);

    // ── 5. φ(x, a) via Meissel recursion ─────────────────────────────────────
    let mut memo: HashMap<(u128, usize), u128> = HashMap::new();
    let phi_x_a = phi_fast_v3(
        x,
        a,
        &seed_primes,
        &phi_table,
        cbrt_u128,
        &medium_pi,
        cbrt_u128, // medium_base = cbrt_x
        &mut memo,
    );

    // ── 6. π(x) = φ(x, a) + a − 1 − S₂ ─────────────────────────────────────
    phi_x_a + a as u128 - 1 - s2
}

/// Step-by-step timing breakdown for `prime_pi_dr_v3`.
pub fn prime_pi_dr_v3_timed(x: u128) -> (u128, [std::time::Duration; 5]) {
    use crate::math::{icbrt, isqrt};
    use crate::phi::{build_phi_table, phi_fast_v3};
    use crate::segment::primes_up_to;
    use crate::sieve::sieve_to;
    use std::collections::HashMap;
    use std::time::Instant;

    let mut times = [std::time::Duration::ZERO; 5];
    if x < 2 {
        return (0, times);
    }

    let cbrt_x = icbrt(x) as u64;
    let cbrt_u128 = cbrt_x as u128;
    let x_23_est = cbrt_u128.saturating_mul(cbrt_u128) + 4 * cbrt_u128;
    if (x_23_est - cbrt_u128 + 1) as usize * 4 > V3_MEDIUM_PI_LIMIT {
        let (result, v2_times) = prime_pi_dr_v2_timed(x);
        times[..4].copy_from_slice(&v2_times);
        return (result, times);
    }

    let t = Instant::now();
    let (_, seed_primes) = sieve_to(cbrt_x);
    let a = seed_primes.len();
    let phi_table = build_phi_table(cbrt_x, &seed_primes);
    let sqrt_x = isqrt(x) as u64;
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];
    times[0] = t.elapsed(); // sieve + phi_table + primes_up_to

    let t = Instant::now();
    let (s2, medium_pi) = hard::s2_and_medium_pi(x, a, s2_primes, &seed_primes, cbrt_x);
    times[1] = t.elapsed(); // combined sweep

    let t = Instant::now();
    let mut memo: HashMap<(u128, usize), u128> = HashMap::new();
    let phi_x_a = phi_fast_v3(
        x,
        a,
        &seed_primes,
        &phi_table,
        cbrt_u128,
        &medium_pi,
        cbrt_u128,
        &mut memo,
    );
    times[2] = t.elapsed(); // phi recursion
    eprintln!("  [v3] memo entries = {}, medium_pi size = {} MB",
              memo.len(), medium_pi.len() * 4 / (1024 * 1024));

    let result = phi_x_a + a as u128 - 1 - s2;
    (result, times)
}

// ─────────────────────────────────────────────────────────────────────────────
// prime_pi_dr_meissel_v3 — PiTable + phi_loop (sans mémo, sans collect_phi)
// ─────────────────────────────────────────────────────────────────────────────

/// π(x) par Meissel v3 : PiTable (√x) + phi_loop sans mémo.
///
/// Différences clés par rapport à `prime_pi_dr_meissel_v2` :
///
/// | Composant      | v2                          | v3                          |
/// |----------------|-----------------------------|-----------------------------|
/// | collect_phi    | DFS préalable (8.7 s à 1e13)| **supprimé**                |
/// | π(n) aux feuilles | HashMap large_pi (1.7 GB) | **PiTable O(1), ~12 MB**    |
/// | φ(x, a)        | phi_cached + mémo 42M        | **phi_loop sans mémo**      |
/// | S₂             | s2_and_phi_queries          | s2_popcount_par (inchangé)  |
///
/// Algorithme :
/// 1. Crible ∛x → `seed_primes`, a = π(∛x).
/// 2. `PiTable::new(√x)` — table dense π(n) pour n ≤ √x (~12 MB à 1e13).
/// 3. `PhiCache::new(x)` — cache dense pour n ≤ x^(1/2.3), a ≤ 100.
/// 4. S₂ = `s2_popcount_par` (sweep parallèle, pas besoin de collect_phi).
/// 5. φ(x, a) = `phi_loop` (formule télescopée + is_pix + PhiCache).
/// 6. π(x) = φ(x, a) + a − 1 − S₂.
pub fn prime_pi_dr_meissel_v3(x: u128) -> u128 {
    use crate::math::{icbrt, isqrt};
    use crate::phi::phi_loop;
    use crate::phi_cache::PhiCache;
    use crate::pi_table::PiTable;
    use crate::segment::primes_up_to;
    use crate::sieve::sieve_to;

    if x < 2 {
        return 0;
    }

    // ── 1. Crible ∛x ─────────────────────────────────────────────────────────
    let cbrt_x = icbrt(x);
    let (_small_pi, seed_primes) = sieve_to(cbrt_x as u64);
    let a = seed_primes.len();

    // ── 2. PiTable jusqu'à √x ────────────────────────────────────────────────
    let sqrt_x = isqrt(x) as u64;
    let pi_table = PiTable::new(sqrt_x);

    // ── 3. PhiCache pour les petits n ────────────────────────────────────────
    let cache = PhiCache::new(x, &seed_primes);

    // ── 4. S₂ par sweep parallèle (pas de collect_phi, pas de large_pi) ──────
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];
    let s2 = if s2_primes.is_empty() {
        0u128
    } else {
        hard::s2_popcount_par(x, a, s2_primes, &seed_primes)
    };

    // ── 5. φ(x, a) sans mémo ─────────────────────────────────────────────────
    let phi_x_a = phi_loop(x, a, &seed_primes, &cache, &pi_table);

    // ── 6. π(x) = φ(x, a) + a − 1 − S₂ ─────────────────────────────────────
    phi_x_a + a as u128 - 1 - s2
}

/// Variante profilée de `prime_pi_dr_meissel_v3`.
/// Retourne (résultat, [step1_sieve, step2_pi_table, step3_s2, step4_phi, _]).
pub fn prime_pi_dr_meissel_v3_timed(x: u128) -> (u128, [std::time::Duration; 5]) {
    use crate::math::{icbrt, isqrt};
    use crate::phi::phi_loop;
    use crate::phi_cache::PhiCache;
    use crate::pi_table::PiTable;
    use crate::segment::primes_up_to;
    use crate::sieve::sieve_to;
    use std::time::Instant;

    let mut times = [std::time::Duration::ZERO; 5];

    if x < 2 {
        return (0, times);
    }

    // ── step1 : crible ────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let cbrt_x = icbrt(x);
    let (_small_pi, seed_primes) = sieve_to(cbrt_x as u64);
    let a = seed_primes.len();
    let sqrt_x = isqrt(x) as u64;
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];
    times[0] = t0.elapsed();
    eprintln!("  [step1] cbrt_x={cbrt_x}, a={a}, sqrt_x={sqrt_x}, s2_primes={}", s2_primes.len());

    // ── step2 : PiTable + PhiCache ────────────────────────────────────────────
    let t1 = Instant::now();
    let pi_table = PiTable::new(sqrt_x);
    let cache = PhiCache::new(x, &seed_primes);
    times[1] = t1.elapsed();
    eprintln!("  [step2] pi_table.max_x={sqrt_x} (~{} MB), cache.max_x={} (~{} MB)",
        (sqrt_x as usize + 1) * 4 / (1024 * 1024),
        cache.max_x,
        (cache.max_x as usize + 1) * (cache.max_a + 1) * 12 / (1024 * 1024));

    // ── step3 : S₂ ───────────────────────────────────────────────────────────
    let t2 = Instant::now();
    let s2 = if s2_primes.is_empty() {
        0u128
    } else {
        hard::s2_popcount_par(x, a, s2_primes, &seed_primes)
    };
    times[2] = t2.elapsed();
    eprintln!("  [step3] s2={s2}");

    // ── step4 : φ(x, a) ──────────────────────────────────────────────────────
    let t3 = Instant::now();
    let phi_x_a = phi_loop(x, a, &seed_primes, &cache, &pi_table);
    times[3] = t3.elapsed();
    eprintln!("  [step4] phi_x_a={phi_x_a}");

    let result = phi_x_a + a as u128 - 1 - s2;
    (result, times)
}

// prime_pi_dr_meissel_v4 — S1 DFS + S2_hard running-sieve

/// Compute π(x) via the true Deléglise-Rivat decomposition:
///   π(x) = S1 + S2_hard + a − 1 − P2
///
/// where:
/// - S1      = Σ_{m squarefree, lpf(m)>p_c, m≤∛x} μ(m)·φ(x/m, c)   (φ_tiny leaves)
/// - S2_hard = running-sieve φ vector for hard leaves (p_b ≤ x^{1/6})
/// - P2      = Σ_{p∈(∛x,√x]} (π(x/p) − (π(p)−1))                    (= s2_popcount_par)
/// - a       = π(∛x)
///
/// Unlike v3 (which uses `phi_loop`), v4 never recurses for φ:
/// hard-leaf φ values are read from a running sieve vector in O(1) each.
pub fn prime_pi_dr_meissel_v4(x: u128) -> u128 {
    use crate::math::{icbrt, isqrt};
    use crate::phi::s1_ordinary;
    use crate::segment::primes_up_to;
    use crate::sieve::sieve_to;

    if x < 2 {
        return 0;
    }

    // Hardware-adaptive alpha: α=2.0 only when x≥3e16 AND L3<16Mo AND ≤8 cores.
    // Rationale: α=2.0 halves sieve-window count but doubles easy-region phi table.
    // On i5-9300H (8Mo L3, 4c) → gain; on i5-13450HX (20Mo L3, 10c) → regression.
    // Only α ∈ {1.0, 2.0} verified OK (intermediate values cause bugs, see bug_alpha2_fix.md).
    let alpha: f64 = crate::parameters::choose_alpha(x);

    // ── 1. Sieve up to y = alpha · ∛x ────────────────────────────────────────
    let cbrt_x = icbrt(x);
    let sqrt_x = isqrt(x) as u64;
    let y = ((cbrt_x as f64 * alpha) as u64).clamp(cbrt_x as u64, sqrt_x);
    let z = (x / y as u128) as u64;

    let (_small_pi, seed_primes) = sieve_to(y);
    let a = seed_primes.len();

    // ── 2. Hard-prime cutoff: b_max = π(√y) ──────────────────────────────────
    let sqrty = isqrt(y as u128) as u64;
    let b_max = seed_primes.partition_point(|&p| p <= sqrty);
    const C: usize = 5; // phi_tiny uses first c=5 primes

    // ── Guard: algorithm requires a > C; fall back to baseline for small x ───
    if a <= C {
        return crate::baseline::prime_pi(x);
    }

    // ── 3. S1: ordinary leaves DFS ───────────────────────────────────────────
    let s1 = s1_ordinary(x, y, C, &seed_primes);

    // ── 4+5. S2_hard + P2 in one combined sweep ───────────────────────────────
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];
    let (s2_hard, p2) = hard::s2_hard_sieve_par(x, y, z, C, b_max, a, &seed_primes, s2_primes);

    // ── 6. π(x) = φ(x, a) + a − 1 − P2  with φ(x,a) = S1 + S2_hard ─────────
    let phi_x_a = (s1 + s2_hard) as u128;
    phi_x_a + a as u128 - 1 - p2
}

/// Timed variant of [`prime_pi_dr_meissel_v4`].
/// Returns (result, [step1_sieve, step2_s1, step3_s2_hard, step4_p2, _]).
pub fn prime_pi_dr_meissel_v4_timed(x: u128) -> (u128, [std::time::Duration; 5]) {
    use crate::math::{icbrt, isqrt};
    use crate::phi::s1_ordinary;
    use crate::segment::primes_up_to;
    use crate::sieve::sieve_to;
    use std::time::Instant;

    let mut times = [std::time::Duration::ZERO; 5];

    if x < 2 {
        return (0, times);
    }

    const ALPHA: f64 = 1.0; // must match prime_pi_dr_meissel_v4

    // ── step1 : crible ────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let cbrt_x = icbrt(x);
    let sqrt_x = isqrt(x) as u64;
    let y = ((cbrt_x as f64 * ALPHA) as u64).clamp(cbrt_x as u64, sqrt_x);
    let z = (x / y as u128) as u64;
    let (_small_pi, seed_primes) = sieve_to(y);
    let a = seed_primes.len();
    let sqrty = isqrt(y as u128) as u64;
    let b_max = seed_primes.partition_point(|&p| p <= sqrty);
    let all_primes = primes_up_to(sqrt_x, &seed_primes);
    let s2_primes = &all_primes[a..];
    times[0] = t0.elapsed();

    // ── step2 : S1 (ordinary DFS) ─────────────────────────────────────────────
    let t1 = Instant::now();
    const C: usize = 5;
    let s1 = s1_ordinary(x, y, C, &seed_primes);
    times[1] = t1.elapsed();

    // ── step3+4 : S2_hard + P2 combined sweep ────────────────────────────────
    let t2 = Instant::now();
    let (s2_hard, p2) = hard::s2_hard_sieve_par(x, y, z, C, b_max, a, &seed_primes, s2_primes);
    times[2] = t2.elapsed();
    times[3] = std::time::Duration::ZERO; // P2 now included in step3

    let phi_x_a = (s1 + s2_hard) as u128;
    let result = phi_x_a + a as u128 - 1 - p2;
    (result, times)
}

#[cfg(test)]
mod tests {
    use super::{
        analyze, analyze_easy_specialized, analyze_hard_specialized,
        analyze_ordinary_dr_like_region, analyze_ordinary_post_plateau_profile,
        analyze_ordinary_quasi_literature_region, analyze_ordinary_region_assembly,
        analyze_ordinary_relative_quotient_envelope_region,
        analyze_ordinary_relative_quotient_envelope_region_variant,
        analyze_ordinary_relative_quotient_hierarchy_region,
        analyze_ordinary_relative_quotient_region,
        analyze_ordinary_relative_quotient_region_variant,
        analyze_ordinary_relative_quotient_shoulder_region,
        analyze_ordinary_relative_quotient_shoulder_region_variant, analyze_ordinary_specialized,
        analyze_post_plateau_triptych, analyze_with_candidate_easy_relative_to_hard,
        analyze_with_candidate_easy_term_band, analyze_with_experimental_easy_relative_to_hard,
        analyze_with_experimental_easy_term_band, analyze_with_experimental_hard_relative_to_easy,
        analyze_with_experimental_phase_c_boundary_package,
        analyze_with_experimental_phase_c_boundary_quotient_guard_package,
        analyze_with_experimental_phase_c_boundary_relative_quotient_band_package,
        analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package,
        analyze_with_experimental_phase_c_boundary_relative_quotient_step_bridge_package,
        analyze_with_experimental_phase_c_buffered_boundary_package,
        analyze_with_experimental_phase_c_linked_package,
        analyze_with_experimental_phase_c_quotient_window_package,
        analyze_with_experimental_phase_c_shifted_quotient_window_package,
        analyze_with_hard_leaf_term_max, analyze_with_phase_c_boundary_package,
        analyze_with_phase_c_boundary_quotient_guard_package,
        analyze_with_phase_c_boundary_relative_quotient_band_package,
        analyze_with_phase_c_boundary_relative_quotient_step_band_package,
        analyze_with_phase_c_boundary_relative_quotient_step_bridge_package,
        analyze_with_phase_c_buffered_boundary_package, analyze_with_phase_c_easy_term_band,
        analyze_with_phase_c_hard_term_band, analyze_with_phase_c_linked_package,
        analyze_with_phase_c_package, analyze_with_phase_c_quotient_window_package,
        analyze_with_phase_c_term_band_package, analyze_with_term_frontiers,
        compare_boundary_candidate_vs_experimental_boundary_relative_quotient_step_band_candidate,
        compare_candidate_easy_reference_vs_phase_c_term_band,
        compare_current_vs_candidate_easy_relative_to_hard,
        compare_current_vs_candidate_easy_relative_to_hard_with_floor,
        compare_current_vs_candidate_easy_term_band,
        compare_current_vs_experimental_easy_relative_to_hard,
        compare_current_vs_experimental_easy_term_band,
        compare_current_vs_experimental_hard_relative_to_easy,
        compare_current_vs_phase_c_easy_term_band, compare_current_vs_phase_c_package,
        compare_ordinary_specialized_vs_relative_quotient_region,
        compare_phase_c_boundary_variants, compare_phase_c_easy_term_bands,
        compare_phase_c_hard_term_band_with_current, compare_phase_c_hard_term_bands,
        compare_phase_c_linked_variants, compare_phase_c_package_vs_boundary_candidate,
        compare_phase_c_package_vs_boundary_package,
        compare_phase_c_package_vs_boundary_quotient_guard_package,
        compare_phase_c_package_vs_boundary_relative_quotient_band_package,
        compare_phase_c_package_vs_boundary_relative_quotient_step_band_package,
        compare_phase_c_package_vs_boundary_relative_quotient_step_bridge_package,
        compare_phase_c_package_vs_buffered_boundary_package,
        compare_phase_c_package_vs_experimental_boundary_candidate,
        compare_phase_c_package_vs_experimental_boundary_quotient_guard_candidate,
        compare_phase_c_package_vs_experimental_boundary_relative_quotient_band_candidate,
        compare_phase_c_package_vs_experimental_boundary_relative_quotient_step_band_candidate,
        compare_phase_c_package_vs_experimental_boundary_relative_quotient_step_bridge_candidate,
        compare_phase_c_package_vs_experimental_buffered_boundary_candidate,
        compare_phase_c_package_vs_experimental_quotient_window_candidate,
        compare_phase_c_package_vs_experimental_shifted_quotient_window_candidate,
        compare_phase_c_package_vs_linked_candidate, compare_phase_c_package_vs_linked_package,
        compare_phase_c_package_vs_quotient_window_package,
        compare_step_band_vs_experimental_step_bridge_candidate, prepare_context,
        preview_inactive_easy_variant, preview_inactive_hard_variant, prime_pi,
        prime_pi_dr_v2, prime_pi_dr_v3,
        skeleton_contributions,
    };

    #[test]
    fn prime_pi_dr_v3_matches_baseline() {
        for x in [
            2u128, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000,
        ] {
            let expected = crate::baseline::prime_pi(x);
            let got = prime_pi_dr_v3(x);
            assert_eq!(got, expected, "prime_pi_dr_v3({x}) = {got}, expected {expected}");
        }
    }

    #[test]
    fn prime_pi_dr_v3_matches_v2() {
        for x in [1_000u128, 50_000, 1_000_000, 5_000_000, 20_000_000] {
            let v2 = prime_pi_dr_v2(x);
            let v3 = prime_pi_dr_v3(x);
            assert_eq!(v3, v2, "v3({x}) = {v3}, v2 = {v2}");
        }
    }

    #[test]
    fn dr_skeleton_matches_baseline_results() {
        assert_eq!(prime_pi(10, 1), 4);
        assert_eq!(prime_pi(100, 2), 25);
        assert_eq!(prime_pi(1_000, 4), 168);
    }

    #[test]
    fn dr_prime_pi_matches_baseline_on_reference_values() {
        for &(x, threads) in &[(10_u128, 1_usize), (100, 2), (1_000, 4), (100_000, 4)] {
            assert_eq!(
                prime_pi(x, threads),
                crate::baseline::prime_pi_with_threads(x, threads)
            );
        }
    }

    #[test]
    fn skeleton_contributions_are_currently_non_negative_and_partial() {
        let ctx = prepare_context(1_000_000);
        let contributions = skeleton_contributions(&ctx);

        assert_eq!(contributions.trivial, 0);
        let easy_range = crate::dr::easy::easy_range(&ctx);
        assert_eq!(contributions.easy, easy_range.len() as u128);
        assert_eq!(
            contributions.ordinary + contributions.hard + contributions.easy,
            crate::baseline::s2::s2(
                ctx.x,
                ctx.a,
                ctx.z_usize,
                ctx.primes.as_slice(),
                ctx.small,
                ctx.large,
                1,
            )
        );
        assert_eq!(
            prime_pi(ctx.x, 1),
            crate::baseline::prime_pi_with_threads(ctx.x, 1)
        );
    }

    #[test]
    fn analysis_matches_baseline_and_domain_partition() {
        let analysis = analyze(1_000_000);
        assert_eq!(
            analysis.result,
            crate::baseline::prime_pi_with_threads(1_000_000, 1)
        );
        assert_eq!(
            analysis.active_len,
            analysis.s1_len + analysis.s2_hard_len + analysis.s2_easy_len
        );
        assert!(analysis.s2_trivial_is_zero);
    }

    #[test]
    fn configurable_hard_leaf_threshold_preserves_exact_result() {
        let a2 = analyze_with_hard_leaf_term_max(1_000_000, 2);
        let a3 = analyze_with_hard_leaf_term_max(1_000_000, 3);

        assert_eq!(a2.result, a3.result);
        assert_eq!(
            a2.result,
            crate::baseline::prime_pi_with_threads(1_000_000, 1)
        );
        assert!(a3.s2_hard_len >= a2.s2_hard_len);
        assert!(a3.s1_len <= a2.s1_len);
    }

    #[test]
    fn hard_rule_is_exposed_consistently_in_analysis() {
        let analysis = analyze_with_hard_leaf_term_max(1_000_000, 3);
        assert_eq!(analysis.alpha, 1.0);
        assert_eq!(analysis.s1_term_min_exclusive, 3);
        assert_eq!(analysis.hard_leaf_term_min, 2);
        assert_eq!(analysis.hard_leaf_term_max, 3);
        assert_eq!(analysis.hard_rule_kind, "alpha_balanced_term_range");
        assert_eq!(analysis.hard_rule_alpha, Some(1.0));
        assert_eq!(analysis.easy_candidate_family, None);
        assert_eq!(analysis.easy_candidate_width, None);
        assert_eq!(analysis.easy_candidate_floor, None);
        assert_eq!(analysis.easy_candidate_term_min, None);
        assert_eq!(analysis.easy_candidate_term_max, None);
        assert_eq!(analysis.easy_leaf_term_value, 1);
        assert_eq!(analysis.easy_leaf_term_min, 1);
        assert_eq!(analysis.easy_leaf_term_max, 1);
        assert_eq!(analysis.easy_rule_kind, "alpha_balanced_term_range");
        assert_eq!(analysis.easy_rule_alpha, Some(1.0));
    }

    #[test]
    fn configurable_easy_threshold_preserves_exact_result() {
        let default = analyze_with_term_frontiers(1_000_000, 3, 1);
        let widened = analyze_with_term_frontiers(1_000_000, 3, 2);

        assert_eq!(default.result, widened.result);
        assert_eq!(
            widened.result,
            crate::baseline::prime_pi_with_threads(1_000_000, 1)
        );
        assert!(widened.s2_easy_len >= default.s2_easy_len);
        assert!(widened.s2_hard_len <= default.s2_hard_len);
        assert_eq!(widened.easy_leaf_term_min, 1);
        assert_eq!(widened.easy_leaf_term_max, 2);
        assert_eq!(widened.hard_leaf_term_min, 3);
    }

    #[test]
    fn frontier_set_is_consistent_with_exposed_analysis_fields() {
        let ctx = prepare_context(1_000_000);
        let frontiers = ctx.frontier_set();
        let analysis = analyze(1_000_000);

        assert_eq!(
            frontiers.s1.term_min_exclusive,
            analysis.s1_term_min_exclusive
        );
        assert_eq!(frontiers.s2_hard.term_min(), analysis.hard_leaf_term_min);
        assert_eq!(frontiers.s2_hard.term_max(), analysis.hard_leaf_term_max);
        assert_eq!(
            frontiers.s2_easy.representative_term_value(),
            analysis.easy_leaf_term_value
        );
    }

    #[test]
    fn easy_specialized_analysis_recomposes_easy_contribution_on_small_sample_grid() {
        for &x in &[1_000_u128, 20_000, 100_000, 1_000_000, 20_000_000] {
            let easy =
                analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                    x, 4, 4, 2, 1, 1,
                );
            let specialized = analyze_easy_specialized(x);

            assert_eq!(
                specialized.residual_len + specialized.transition_len + specialized.specialized_len,
                specialized.easy_len
            );
            assert_eq!(
                specialized.residual_sum + specialized.transition_sum + specialized.specialized_sum,
                specialized.easy_sum
            );
            assert_eq!(specialized.easy_sum, easy.contributions.easy);
        }
    }

    #[test]
    fn hard_specialized_analysis_recomposes_hard_contribution_on_small_sample_grid() {
        for &x in &[1_000_u128, 20_000, 100_000, 1_000_000, 20_000_000] {
            let hard_analysis =
                analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                    x, 4, 4, 2, 1, 1,
                );
            let specialized = analyze_hard_specialized(x);

            assert_eq!(
                specialized.residual_len + specialized.transition_len + specialized.specialized_len,
                specialized.hard_len
            );
            assert_eq!(
                specialized.residual_sum + specialized.transition_sum + specialized.specialized_sum,
                specialized.hard_sum
            );
            assert_eq!(specialized.hard_sum, hard_analysis.contributions.hard);
        }
    }

    #[test]
    fn ordinary_specialized_analysis_recomposes_ordinary_contribution_on_small_sample_grid() {
        for &x in &[1_000_u128, 20_000, 100_000, 1_000_000, 20_000_000] {
            let ordinary_analysis =
                analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                    x, 4, 4, 2, 1, 1,
                );
            let specialized = analyze_ordinary_specialized(x);

            assert_eq!(
                specialized.residual_len
                    + specialized.pretransition_len
                    + specialized.transition_len
                    + specialized.specialized_len,
                specialized.ordinary_len
            );
            assert_eq!(
                specialized.residual_sum
                    + specialized.pretransition_sum
                    + specialized.transition_sum
                    + specialized.specialized_sum,
                specialized.ordinary_sum
            );
            assert_eq!(
                specialized.ordinary_sum,
                ordinary_analysis.contributions.ordinary
            );
        }
    }

    #[test]
    fn ordinary_relative_quotient_analysis_recomposes_ordinary_contribution_on_small_sample_grid() {
        for &x in &[1_000_u128, 20_000, 100_000, 1_000_000, 20_000_000] {
            let ordinary_analysis =
                analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                    x, 4, 4, 2, 1, 1,
                );
            let specialized = analyze_ordinary_relative_quotient_region(x);

            assert_eq!(
                specialized.left_residual_len
                    + specialized.region_len
                    + specialized.right_residual_len,
                specialized.ordinary_len
            );
            assert_eq!(
                specialized.left_residual_sum
                    + specialized.region_sum
                    + specialized.right_residual_sum,
                specialized.ordinary_sum
            );
            assert_eq!(
                specialized.ordinary_sum,
                ordinary_analysis.contributions.ordinary
            );
        }
    }

    #[test]
    fn ordinary_relative_quotient_variant_analysis_recomposes_ordinary_contribution() {
        for &x in &[20_000_u128, 100_000, 20_000_000] {
            let ordinary_analysis =
                analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                    x, 4, 4, 2, 1, 1,
                );
            let specialized = analyze_ordinary_relative_quotient_region_variant(x, 1, 2);

            assert_eq!(
                specialized.left_residual_sum
                    + specialized.region_sum
                    + specialized.right_residual_sum,
                specialized.ordinary_sum
            );
            assert_eq!(
                specialized.ordinary_sum,
                ordinary_analysis.contributions.ordinary
            );
        }
    }

    #[test]
    fn ordinary_region_comparison_is_consistent() {
        let comparison = compare_ordinary_specialized_vs_relative_quotient_region(20_000_000);
        assert_eq!(comparison.ordinary_len, 546);
        assert!(comparison.current_terminal_len >= 2);
        assert!(comparison.relative_region_len >= 3);
    }

    #[test]
    fn ordinary_post_plateau_profile_is_consistent() {
        let profile = analyze_ordinary_post_plateau_profile(20_000_000);
        assert_eq!(profile.ordinary_len, 546);
        assert!(profile.terminal_len >= 2);
        assert!(profile.region_len >= 3);
        assert_eq!(
            profile.left_residual_len + profile.region_len + profile.right_residual_len,
            profile.ordinary_len
        );
        assert_eq!(
            profile.left_residual_sum + profile.region_sum + profile.right_residual_sum,
            analyze_ordinary_relative_quotient_region(20_000_000).ordinary_sum
        );
    }

    #[test]
    fn post_plateau_triptych_analysis_is_consistent() {
        let analysis = analyze_post_plateau_triptych(20_000_000);
        assert_eq!(analysis.ordinary_len, 546);
        assert!(analysis.ordinary_region_len >= analysis.ordinary_terminal_len);
        assert!(analysis.ordinary_region_sum >= analysis.ordinary_terminal_sum);
        assert!(analysis.ordinary_assembly_support_len >= analysis.ordinary_region_len);
        assert!(
            analysis.ordinary_quasi_literature_middle_len >= analysis.ordinary_assembly_core_len
        );
        assert_eq!(
            analysis.ordinary_assembly_core_len + analysis.ordinary_assembly_support_len,
            analysis.ordinary_len
        );
        assert_eq!(
            analysis.ordinary_quasi_literature_middle_len
                + analysis.ordinary_quasi_literature_outer_len,
            analysis.ordinary_len
        );
        assert!(analysis.easy_focus_len <= analysis.easy_len);
        assert!(analysis.hard_focus_len <= analysis.hard_len);
    }

    #[test]
    fn ordinary_relative_quotient_shoulder_analysis_recomposes_ordinary_contribution() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let ordinary_analysis =
                analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                    x, 4, 4, 2, 1, 1,
                );
            let specialized = analyze_ordinary_relative_quotient_shoulder_region(x);
            assert_eq!(
                specialized.ordinary_sum,
                ordinary_analysis.contributions.ordinary
            );
            assert_eq!(
                specialized.left_residual_sum
                    + specialized.left_shoulder_sum
                    + specialized.core_sum
                    + specialized.right_shoulder_sum
                    + specialized.right_residual_sum,
                specialized.ordinary_sum
            );
        }
    }

    #[test]
    fn ordinary_relative_quotient_shoulder_variant_recomposes_ordinary_contribution() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let ordinary_analysis =
                analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                    x, 4, 4, 2, 1, 1,
                );
            let specialized =
                analyze_ordinary_relative_quotient_shoulder_region_variant(x, 1, 1, 3);
            assert_eq!(
                specialized.ordinary_sum,
                ordinary_analysis.contributions.ordinary
            );
            assert_eq!(
                specialized.left_residual_sum
                    + specialized.left_shoulder_sum
                    + specialized.core_sum
                    + specialized.right_shoulder_sum
                    + specialized.right_residual_sum,
                specialized.ordinary_sum
            );
        }
    }

    #[test]
    fn ordinary_relative_quotient_envelope_analysis_recomposes_ordinary_contribution() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let ordinary_analysis =
                analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                    x, 4, 4, 2, 1, 1,
                );
            let specialized = analyze_ordinary_relative_quotient_envelope_region(x);
            assert_eq!(
                specialized.ordinary_sum,
                ordinary_analysis.contributions.ordinary
            );
            assert_eq!(
                specialized.left_residual_sum
                    + specialized.left_envelope_sum
                    + specialized.core_sum
                    + specialized.right_envelope_sum
                    + specialized.right_residual_sum,
                specialized.ordinary_sum
            );
        }
    }

    #[test]
    fn ordinary_relative_quotient_envelope_variant_recomposes_ordinary_contribution() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let ordinary_analysis =
                analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                    x, 4, 4, 2, 1, 1,
                );
            let specialized =
                analyze_ordinary_relative_quotient_envelope_region_variant(x, 1, 1, 4);
            assert_eq!(
                specialized.ordinary_sum,
                ordinary_analysis.contributions.ordinary
            );
        }
    }

    #[test]
    fn ordinary_relative_quotient_hierarchy_analysis_recomposes_ordinary_contribution() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let ordinary_analysis =
                analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                    x, 4, 4, 2, 1, 1,
                );
            let specialized = analyze_ordinary_relative_quotient_hierarchy_region(x);
            assert_eq!(
                specialized.ordinary_sum,
                ordinary_analysis.contributions.ordinary
            );
            assert_eq!(
                specialized.left_residual_sum
                    + specialized.left_outer_band_sum
                    + specialized.left_near_band_sum
                    + specialized.inner_core_sum
                    + specialized.right_near_band_sum
                    + specialized.right_outer_band_sum
                    + specialized.right_residual_sum,
                specialized.ordinary_sum
            );
        }
    }

    #[test]
    fn ordinary_region_assembly_analysis_recomposes_ordinary_contribution() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let ordinary_analysis =
                analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                    x, 4, 4, 2, 1, 1,
                );
            let assembly = analyze_ordinary_region_assembly(x);
            assert_eq!(
                assembly.ordinary_sum,
                ordinary_analysis.contributions.ordinary
            );
            assert_eq!(
                assembly.left_outer_support_sum
                    + assembly.left_adjacent_support_sum
                    + assembly.central_assembly_sum
                    + assembly.right_adjacent_support_sum
                    + assembly.right_outer_support_sum,
                assembly.ordinary_sum
            );
        }
    }

    #[test]
    fn ordinary_quasi_literature_analysis_recomposes_ordinary_contribution() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let ordinary_analysis =
                analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                    x, 4, 4, 2, 1, 1,
                );
            let region = analyze_ordinary_quasi_literature_region(x);
            assert_eq!(
                region.ordinary_sum,
                ordinary_analysis.contributions.ordinary
            );
            assert_eq!(
                region.left_outer_work_sum + region.middle_work_sum + region.right_outer_work_sum,
                region.ordinary_sum
            );
        }
    }

    #[test]
    fn ordinary_dr_like_analysis_recomposes_ordinary_contribution() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let ordinary_analysis =
                analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                    x, 4, 4, 2, 1, 1,
                );
            let region = analyze_ordinary_dr_like_region(x);
            assert_eq!(
                region.ordinary_sum,
                ordinary_analysis.contributions.ordinary
            );
            assert_eq!(
                region.left_outer_work_sum
                    + region.left_transfer_work_sum
                    + region.central_work_region_sum
                    + region.right_transfer_work_sum
                    + region.right_outer_work_sum,
                region.ordinary_sum
            );
        }
    }

    #[test]
    fn domain_set_matches_analysis_output() {
        let ctx = prepare_context(1_000_000);
        let domain_set = ctx.domain_set();
        let analysis = analyze(1_000_000);

        assert_eq!(domain_set.result, analysis.result);
        assert_eq!(domain_set.contributions, analysis.contributions);
        assert_eq!(domain_set.domains.active.len(), analysis.active_len);
        assert_eq!(domain_set.domains.s1.leaves.len(), analysis.s1_len);
        assert_eq!(
            domain_set.domains.s2_hard.leaves.len(),
            analysis.s2_hard_len
        );
        assert_eq!(
            domain_set.domains.s2_easy.leaves.len(),
            analysis.s2_easy_len
        );
    }

    #[test]
    fn inactive_easy_variant_preview_is_explicit() {
        let preview = preview_inactive_easy_variant(4, 2);

        assert_eq!(preview.kind, "relative_to_hard");
        assert_eq!(preview.term_min, 2);
        assert_eq!(preview.term_max, 3);
        assert_eq!(preview.alpha, Some(1.0));
    }

    #[test]
    fn inactive_hard_variant_preview_is_explicit() {
        let preview = preview_inactive_hard_variant(2, 3);

        assert_eq!(preview.kind, "relative_to_easy");
        assert_eq!(preview.term_min, 3);
        assert_eq!(preview.term_max, 5);
        assert_eq!(preview.alpha, Some(1.0));
    }

    #[test]
    fn experimental_easy_relative_to_hard_is_exact_on_concrete_sample() {
        let experimental = analyze_with_experimental_easy_relative_to_hard(20_000_000, 3, 2);
        let current = analyze_with_term_frontiers(20_000_000, 3, 2);

        assert_eq!(experimental.result, current.result);
        assert_eq!(
            experimental.result,
            crate::baseline::prime_pi_with_threads(20_000_000, 1)
        );
        assert_eq!(experimental.easy_rule_kind, "relative_to_hard");
        assert_eq!(experimental.easy_leaf_term_min, 1);
        assert_eq!(experimental.easy_leaf_term_max, 2);
    }

    #[test]
    fn experimental_easy_term_band_is_exact_on_concrete_sample() {
        let experimental = analyze_with_experimental_easy_term_band(20_000_000, 3, 2, 2);
        let current = analyze_with_term_frontiers(20_000_000, 3, 1);

        assert_eq!(experimental.result, current.result);
        assert_eq!(
            experimental.result,
            crate::baseline::prime_pi_with_threads(20_000_000, 1)
        );
        assert_eq!(experimental.easy_candidate_family, Some("term_band"));
        assert_eq!(experimental.easy_leaf_term_min, 2);
        assert_eq!(experimental.easy_leaf_term_max, 2);
    }

    #[test]
    fn experimental_hard_relative_to_easy_is_exact_on_concrete_sample() {
        let experimental = analyze_with_experimental_hard_relative_to_easy(20_000_000, 2, 1);
        let current = analyze_with_term_frontiers(20_000_000, 3, 2);

        assert_eq!(experimental.result, current.result);
        assert_eq!(
            experimental.result,
            crate::baseline::prime_pi_with_threads(20_000_000, 1)
        );
        assert_eq!(experimental.hard_rule_kind, "relative_to_easy");
        assert_eq!(experimental.hard_leaf_term_min, 3);
        assert_eq!(experimental.hard_leaf_term_max, 3);
    }

    #[test]
    fn comparison_helper_tracks_easy_experimental_delta() {
        let comparison = compare_current_vs_experimental_easy_relative_to_hard(20_000_000, 3, 1, 2);

        assert_eq!(comparison.current.result, comparison.experimental.result);
        assert_eq!(
            comparison.current.easy_rule_kind,
            "alpha_balanced_term_range"
        );
        assert_eq!(comparison.experimental.easy_rule_kind, "relative_to_hard");
        assert!(comparison.experimental.s2_easy_len >= comparison.current.s2_easy_len);
    }

    #[test]
    fn comparison_helper_tracks_easy_term_band_delta() {
        let comparison = compare_current_vs_experimental_easy_term_band(20_000_000, 3, 1, 2, 2);

        assert_eq!(comparison.current.result, comparison.experimental.result);
        assert_eq!(
            comparison.experimental.easy_candidate_family,
            Some("term_band")
        );
        assert_eq!(comparison.experimental.easy_candidate_term_min, Some(2));
        assert_eq!(comparison.experimental.easy_candidate_term_max, Some(2));
    }

    #[test]
    fn comparison_helper_tracks_hard_experimental_delta() {
        let comparison = compare_current_vs_experimental_hard_relative_to_easy(20_000_000, 3, 2, 1);

        assert_eq!(comparison.current.result, comparison.experimental.result);
        assert_eq!(
            comparison.current.hard_rule_kind,
            "alpha_balanced_term_range"
        );
        assert_eq!(comparison.experimental.hard_rule_kind, "relative_to_easy");
        assert_eq!(
            comparison.current.s2_hard_len,
            comparison.experimental.s2_hard_len
        );
    }

    #[test]
    fn candidate_easy_relative_to_hard_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_candidate_easy_relative_to_hard(x, 3);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(analysis.easy_candidate_family, Some("relative_to_hard"));
            assert_eq!(analysis.easy_rule_kind, "relative_to_hard_with_floor");
            assert_eq!(analysis.easy_candidate_width, Some(2));
            assert_eq!(analysis.easy_candidate_floor, Some(1));
        }
    }

    #[test]
    fn candidate_easy_term_band_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_candidate_easy_term_band(x, 3);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(analysis.easy_candidate_family, Some("term_band"));
            assert_eq!(analysis.easy_rule_kind, "alpha_balanced_term_range");
            assert_eq!(analysis.easy_candidate_term_min, Some(1));
            assert_eq!(analysis.easy_candidate_term_max, Some(2));
        }
    }

    #[test]
    fn phase_c_easy_term_band_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_phase_c_easy_term_band(x, 5);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(analysis.easy_candidate_family, Some("phase_c_term_band"));
            assert_eq!(analysis.easy_rule_kind, "alpha_balanced_term_range");
            assert_eq!(analysis.easy_candidate_term_min, Some(1));
            assert_eq!(analysis.easy_candidate_term_max, Some(4));
        }
    }

    #[test]
    fn candidate_easy_comparison_helper_is_exact_on_small_sample_grid() {
        for &(x, easy_leaf_term_max) in &[(1_000_u128, 1_u128), (100_000, 1), (20_000_000, 1)] {
            let comparison =
                compare_current_vs_candidate_easy_relative_to_hard(x, 3, easy_leaf_term_max);
            assert_eq!(comparison.current.result, comparison.experimental.result);
        }
    }

    #[test]
    fn candidate_easy_term_band_comparison_helper_is_exact_on_small_sample_grid() {
        for &(x, easy_leaf_term_max) in &[(1_000_u128, 1_u128), (100_000, 1), (20_000_000, 1)] {
            let comparison = compare_current_vs_candidate_easy_term_band(x, 3, easy_leaf_term_max);
            assert_eq!(comparison.current.result, comparison.experimental.result);
        }
    }

    #[test]
    fn phase_c_easy_term_band_comparison_helper_is_exact_on_small_sample_grid() {
        for &(x, easy_leaf_term_max) in &[(1_000_u128, 1_u128), (100_000, 1), (20_000_000, 1)] {
            let comparison = compare_current_vs_phase_c_easy_term_band(x, 5, easy_leaf_term_max);
            assert_eq!(comparison.current.result, comparison.experimental.result);
        }
    }

    #[test]
    fn reference_candidate_vs_phase_c_term_band_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_candidate_easy_reference_vs_phase_c_term_band(x, 3);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("relative_to_hard")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_term_band")
            );
        }
    }

    #[test]
    fn phase_c_term_band_variants_compare_exactly_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_phase_c_easy_term_bands(x, 1, 4, 1, 3);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(comparison.current.easy_candidate_family, Some("term_band"));
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("term_band")
            );
            assert_eq!(comparison.current.easy_candidate_term_max, Some(4));
            assert_eq!(comparison.experimental.easy_candidate_term_max, Some(3));
        }
    }

    #[test]
    fn phase_c_hard_term_band_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_phase_c_hard_term_band(x);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_hard_term_band")
            );
            assert_eq!(analysis.hard_leaf_term_min, 5);
            assert_eq!(analysis.hard_leaf_term_max, 6);
        }
    }

    #[test]
    fn phase_c_hard_term_band_comparison_with_current_is_exact() {
        for &(x, hard_leaf_term_max, easy_leaf_term_max) in &[
            (1_000_u128, 6_u128, 4_u128),
            (100_000, 6, 4),
            (20_000_000, 6, 4),
        ] {
            let comparison = compare_phase_c_hard_term_band_with_current(
                x,
                hard_leaf_term_max,
                easy_leaf_term_max,
            );
            assert_eq!(comparison.current.result, comparison.experimental.result);
        }
    }

    #[test]
    fn phase_c_hard_term_band_variants_compare_exactly_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_phase_c_hard_term_bands(x, 5, 6, 5, 7);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_hard_term_band")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_hard_term_band")
            );
            assert_eq!(comparison.current.hard_leaf_term_max, 6);
            assert_eq!(comparison.experimental.hard_leaf_term_max, 7);
        }
    }

    #[test]
    fn phase_c_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_phase_c_package(x);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(analysis.easy_candidate_family, Some("phase_c_package"));
            assert_eq!(analysis.easy_leaf_term_min, 1);
            assert_eq!(analysis.easy_leaf_term_max, 5);
            assert_eq!(analysis.hard_leaf_term_min, 6);
            assert_eq!(analysis.hard_leaf_term_max, 6);
        }
    }

    #[test]
    fn phase_c_package_comparison_with_current_is_exact() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_current_vs_phase_c_package(x);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_package")
            );
        }
    }

    #[test]
    fn phase_c_term_band_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_phase_c_term_band_package(x);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_term_band_package")
            );
            assert_eq!(analysis.easy_leaf_term_min, 1);
            assert_eq!(analysis.easy_leaf_term_max, 4);
            assert_eq!(analysis.hard_leaf_term_min, 5);
            assert_eq!(analysis.hard_leaf_term_max, 6);
        }
    }

    #[test]
    fn phase_c_boundary_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_phase_c_boundary_package(x);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_boundary_package")
            );
            assert_eq!(analysis.easy_leaf_term_min, 1);
            assert_eq!(analysis.easy_leaf_term_max, 5);
            assert_eq!(analysis.hard_leaf_term_min, 6);
            assert_eq!(analysis.hard_leaf_term_max, 6);
            assert_eq!(analysis.easy_candidate_floor, Some(5));
        }
    }

    #[test]
    fn phase_c_boundary_quotient_guard_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_phase_c_boundary_quotient_guard_package(x);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_boundary_quotient_guard_package")
            );
        }
    }

    #[test]
    fn phase_c_boundary_relative_quotient_band_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_phase_c_boundary_relative_quotient_band_package(x);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_boundary_relative_quotient_band_package")
            );
        }
    }

    #[test]
    fn phase_c_boundary_relative_quotient_step_band_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_phase_c_boundary_relative_quotient_step_band_package(x);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_boundary_relative_quotient_step_band_package")
            );
        }
    }

    #[test]
    fn phase_c_boundary_relative_quotient_step_bridge_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_phase_c_boundary_relative_quotient_step_bridge_package(x);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_boundary_relative_quotient_step_bridge_package")
            );
        }
    }

    #[test]
    fn phase_c_buffered_boundary_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_phase_c_buffered_boundary_package(x);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_buffered_boundary_package")
            );
            assert_eq!(analysis.easy_leaf_term_min, 1);
            assert_eq!(analysis.easy_leaf_term_max, 4);
            assert_eq!(analysis.hard_leaf_term_min, 6);
            assert_eq!(analysis.hard_leaf_term_max, 6);
        }
    }

    #[test]
    fn phase_c_quotient_window_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_phase_c_quotient_window_package(x);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_quotient_window_package")
            );
        }
    }

    #[test]
    fn phase_c_linked_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_phase_c_linked_package(x);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_linked_package")
            );
            assert_eq!(analysis.easy_leaf_term_min, 1);
            assert_eq!(analysis.easy_leaf_term_max, 5);
            assert_eq!(analysis.hard_leaf_term_min, 6);
            assert_eq!(analysis.hard_leaf_term_max, 6);
        }
    }

    #[test]
    fn phase_c_package_vs_linked_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_phase_c_package_vs_linked_package(x);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_term_band_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_linked_package")
            );
        }
    }

    #[test]
    fn phase_c_package_vs_linked_candidate_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_phase_c_package_vs_linked_candidate(x);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_term_band_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_package")
            );
            assert_eq!(comparison.current.easy_leaf_term_max, 4);
            assert_eq!(comparison.experimental.easy_leaf_term_max, 5);
            assert_eq!(comparison.experimental.hard_leaf_term_min, 6);
            assert_eq!(comparison.experimental.hard_leaf_term_max, 6);
        }
    }

    #[test]
    fn phase_c_package_vs_boundary_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_phase_c_package_vs_boundary_package(x);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_boundary_package")
            );
            assert_eq!(comparison.current.easy_leaf_term_max, 5);
            assert_eq!(comparison.experimental.easy_leaf_term_max, 5);
        }
    }

    #[test]
    fn phase_c_package_vs_boundary_quotient_guard_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_phase_c_package_vs_boundary_quotient_guard_package(x);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_boundary_quotient_guard_package")
            );
        }
    }

    #[test]
    fn phase_c_package_vs_boundary_relative_quotient_band_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_phase_c_package_vs_boundary_relative_quotient_band_package(x);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_boundary_relative_quotient_band_package")
            );
        }
    }

    #[test]
    fn phase_c_package_vs_boundary_relative_quotient_step_band_package_is_exact_on_small_sample_grid()
     {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison =
                compare_phase_c_package_vs_boundary_relative_quotient_step_band_package(x);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_boundary_relative_quotient_step_band_package")
            );
        }
    }

    #[test]
    fn phase_c_package_vs_boundary_relative_quotient_step_bridge_package_is_exact_on_small_sample_grid()
     {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison =
                compare_phase_c_package_vs_boundary_relative_quotient_step_bridge_package(x);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_boundary_relative_quotient_step_bridge_package")
            );
        }
    }

    #[test]
    fn phase_c_package_vs_buffered_boundary_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_phase_c_package_vs_buffered_boundary_package(x);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_buffered_boundary_package")
            );
            assert_eq!(comparison.experimental.easy_leaf_term_max, 4);
            assert_eq!(comparison.experimental.hard_leaf_term_min, 6);
            assert_eq!(comparison.experimental.hard_leaf_term_max, 6);
        }
    }

    #[test]
    fn phase_c_package_vs_quotient_window_package_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_phase_c_package_vs_quotient_window_package(x);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_quotient_window_package")
            );
        }
    }

    #[test]
    fn phase_c_package_vs_boundary_candidate_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_phase_c_package_vs_boundary_candidate(x);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_boundary_package")
            );
            assert_eq!(comparison.current.easy_leaf_term_max, 5);
            assert_eq!(comparison.experimental.easy_leaf_term_max, 4);
            assert_eq!(comparison.current.hard_leaf_term_min, 6);
            assert_eq!(comparison.experimental.hard_leaf_term_min, 5);
            assert_eq!(comparison.experimental.hard_leaf_term_max, 6);
        }
    }

    #[test]
    fn phase_c_package_vs_local_boundary_neighbor_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_phase_c_package_vs_experimental_boundary_candidate(x, 4, 5, 1);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_boundary_package")
            );
            assert_eq!(comparison.experimental.easy_leaf_term_min, 1);
            assert_eq!(comparison.experimental.easy_leaf_term_max, 4);
            assert_eq!(comparison.experimental.hard_leaf_term_min, 5);
            assert_eq!(comparison.experimental.hard_leaf_term_max, 5);
        }
    }

    #[test]
    fn experimental_phase_c_boundary_variant_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_experimental_phase_c_boundary_package(x, 4, 4, 2);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_boundary_package")
            );
            assert_eq!(analysis.easy_leaf_term_min, 1);
            assert_eq!(analysis.easy_leaf_term_max, 4);
            assert_eq!(analysis.hard_leaf_term_min, 5);
            assert_eq!(analysis.hard_leaf_term_max, 6);
            assert_eq!(analysis.easy_candidate_floor, Some(4));
        }
    }

    #[test]
    fn experimental_phase_c_boundary_quotient_guard_variant_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis =
                analyze_with_experimental_phase_c_boundary_quotient_guard_package(x, 4, 4, 2, 1);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_boundary_quotient_guard_package")
            );
        }
    }

    #[test]
    fn experimental_phase_c_boundary_relative_quotient_band_variant_is_exact_on_small_sample_grid()
    {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis =
                analyze_with_experimental_phase_c_boundary_relative_quotient_band_package(
                    x, 4, 4, 2, 0, 0,
                );
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_boundary_relative_quotient_band_package")
            );
        }
    }

    #[test]
    fn experimental_phase_c_boundary_relative_quotient_step_band_variant_is_exact_on_small_sample_grid()
     {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis =
                analyze_with_experimental_phase_c_boundary_relative_quotient_step_band_package(
                    x, 4, 4, 2, 0, 0,
                );
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_boundary_relative_quotient_step_band_package")
            );
        }
    }

    #[test]
    fn experimental_phase_c_boundary_relative_quotient_step_bridge_variant_is_exact_on_small_sample_grid()
     {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis =
                analyze_with_experimental_phase_c_boundary_relative_quotient_step_bridge_package(
                    x, 4, 4, 2, 1, 1, 1,
                );
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_boundary_relative_quotient_step_bridge_package")
            );
        }
    }

    #[test]
    fn phase_c_package_vs_experimental_boundary_quotient_guard_candidate_is_exact_on_small_sample_grid()
     {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison =
                compare_phase_c_package_vs_experimental_boundary_quotient_guard_candidate(
                    x, 4, 4, 2, 1,
                );
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_boundary_quotient_guard_package")
            );
        }
    }

    #[test]
    fn phase_c_package_vs_experimental_boundary_relative_quotient_band_candidate_is_exact_on_small_sample_grid()
     {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison =
                compare_phase_c_package_vs_experimental_boundary_relative_quotient_band_candidate(
                    x, 4, 4, 2, 0, 0,
                );
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_boundary_relative_quotient_band_package")
            );
        }
    }

    #[test]
    fn phase_c_package_vs_experimental_boundary_relative_quotient_step_band_candidate_is_exact_on_small_sample_grid()
     {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison =
                compare_phase_c_package_vs_experimental_boundary_relative_quotient_step_band_candidate(
                    x, 4, 4, 2, 0, 0,
                );
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_boundary_relative_quotient_step_band_package")
            );
        }
    }

    #[test]
    fn phase_c_package_vs_experimental_boundary_relative_quotient_step_bridge_candidate_is_exact_on_small_sample_grid()
     {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison =
                compare_phase_c_package_vs_experimental_boundary_relative_quotient_step_bridge_candidate(
                    x, 4, 4, 2, 1, 1, 1,
                );
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_boundary_relative_quotient_step_bridge_package")
            );
        }
    }

    #[test]
    fn boundary_candidate_vs_relative_quotient_step_band_candidate_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison =
                compare_boundary_candidate_vs_experimental_boundary_relative_quotient_step_band_candidate(
                    x, 4, 4, 2, 4, 4, 2, 1, 1,
                );
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_boundary_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_boundary_relative_quotient_step_band_package")
            );
        }
    }

    #[test]
    fn step_band_vs_step_bridge_candidate_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_step_band_vs_experimental_step_bridge_candidate(
                x, 4, 4, 2, 1, 1, 4, 4, 2, 1, 1, 1,
            );
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_boundary_relative_quotient_step_band_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_boundary_relative_quotient_step_bridge_package")
            );
        }
    }

    #[test]
    fn experimental_phase_c_buffered_boundary_variant_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis =
                analyze_with_experimental_phase_c_buffered_boundary_package(x, 4, 4, 1, 2);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_buffered_boundary_package")
            );
            assert_eq!(analysis.easy_leaf_term_min, 1);
            assert_eq!(analysis.easy_leaf_term_max, 4);
            assert_eq!(analysis.hard_leaf_term_min, 6);
            assert_eq!(analysis.hard_leaf_term_max, 7);
        }
    }

    #[test]
    fn experimental_phase_c_quotient_window_variant_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_experimental_phase_c_quotient_window_package(x, 1, 2);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_quotient_window_package")
            );
        }
    }

    #[test]
    fn experimental_phase_c_shifted_quotient_window_variant_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis =
                analyze_with_experimental_phase_c_shifted_quotient_window_package(x, 1, 1, 2);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_quotient_window_package")
            );
        }
    }

    #[test]
    fn phase_c_package_vs_experimental_buffered_neighbor_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison =
                compare_phase_c_package_vs_experimental_buffered_boundary_candidate(x, 4, 4, 1, 2);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_buffered_boundary_package")
            );
        }
    }

    #[test]
    fn phase_c_package_vs_experimental_quotient_neighbor_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison =
                compare_phase_c_package_vs_experimental_quotient_window_candidate(x, 1, 2);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_quotient_window_package")
            );
        }
    }

    #[test]
    fn phase_c_package_vs_experimental_shifted_quotient_neighbor_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison =
                compare_phase_c_package_vs_experimental_shifted_quotient_window_candidate(
                    x, 1, 1, 2,
                );
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_quotient_window_package")
            );
        }
    }

    #[test]
    fn phase_c_boundary_variants_compare_exactly_on_small_sample_grid() {
        let current = crate::parameters::DrTuning::phase_c_boundary_candidate();
        let experimental = crate::parameters::PhaseCBoundaryCandidate {
            boundary_term: 4,
            easy_width: 4,
            hard_width: 2,
        };

        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_phase_c_boundary_variants(x, current, experimental);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_boundary_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_boundary_package")
            );
            assert_eq!(comparison.current.easy_leaf_term_max, 5);
            assert_eq!(comparison.experimental.easy_leaf_term_max, 4);
            assert_eq!(comparison.experimental.hard_leaf_term_min, 5);
            assert_eq!(comparison.experimental.hard_leaf_term_max, 6);
        }
    }

    #[test]
    fn experimental_phase_c_linked_variant_is_exact_on_small_sample_grid() {
        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let analysis = analyze_with_experimental_phase_c_linked_package(x, 5, 1, 2);
            assert_eq!(
                analysis.result,
                crate::baseline::prime_pi_with_threads(x, 1)
            );
            assert_eq!(
                analysis.easy_candidate_family,
                Some("phase_c_linked_package")
            );
            assert_eq!(analysis.easy_leaf_term_min, 1);
            assert_eq!(analysis.easy_leaf_term_max, 5);
            assert_eq!(analysis.hard_leaf_term_min, 6);
            assert_eq!(analysis.hard_leaf_term_max, 7);
        }
    }

    #[test]
    fn phase_c_linked_variants_compare_exactly_on_small_sample_grid() {
        let current = crate::parameters::DrTuning::phase_c_linked_candidate();
        let experimental = crate::parameters::PhaseCLinkedCandidate {
            easy_width: 5,
            easy_min_term_floor: 1,
            hard_width: 2,
        };

        for &x in &[1_000_u128, 100_000, 20_000_000] {
            let comparison = compare_phase_c_linked_variants(x, current, experimental);
            assert_eq!(comparison.current.result, comparison.experimental.result);
            assert_eq!(
                comparison.current.easy_candidate_family,
                Some("phase_c_linked_package")
            );
            assert_eq!(
                comparison.experimental.easy_candidate_family,
                Some("phase_c_linked_package")
            );
            assert_eq!(comparison.current.easy_leaf_term_max, 5);
            assert_eq!(comparison.experimental.easy_leaf_term_max, 5);
        }
    }

    #[test]
    fn nearby_phase_c_linked_candidates_stay_exact_on_small_sample_grid() {
        let candidates = [
            crate::parameters::PhaseCLinkedCandidate {
                easy_width: 5,
                easy_min_term_floor: 1,
                hard_width: 1,
            },
            crate::parameters::PhaseCLinkedCandidate {
                easy_width: 5,
                easy_min_term_floor: 1,
                hard_width: 2,
            },
            crate::parameters::PhaseCLinkedCandidate {
                easy_width: 5,
                easy_min_term_floor: 1,
                hard_width: 3,
            },
            crate::parameters::PhaseCLinkedCandidate {
                easy_width: 5,
                easy_min_term_floor: 2,
                hard_width: 2,
            },
            crate::parameters::PhaseCLinkedCandidate {
                easy_width: 6,
                easy_min_term_floor: 1,
                hard_width: 2,
            },
        ];

        for candidate in candidates {
            for &x in &[1_000_u128, 100_000, 20_000_000] {
                let analysis = analyze_with_experimental_phase_c_linked_package(
                    x,
                    candidate.easy_width,
                    candidate.easy_min_term_floor,
                    candidate.hard_width,
                );
                assert_eq!(
                    analysis.result,
                    crate::baseline::prime_pi_with_threads(x, 1)
                );
                assert_eq!(
                    analysis.easy_candidate_family,
                    Some("phase_c_linked_package")
                );
            }
        }
    }

    #[test]
    fn candidate_easy_with_floor_comparison_helper_is_exact() {
        let comparison =
            compare_current_vs_candidate_easy_relative_to_hard_with_floor(20_000_000, 3, 1, 1);

        assert_eq!(comparison.current.result, comparison.experimental.result);
        assert_eq!(
            comparison.experimental.easy_rule_kind,
            "relative_to_hard_with_floor"
        );
        assert_eq!(comparison.experimental.easy_candidate_width, Some(2));
        assert_eq!(comparison.experimental.easy_candidate_floor, Some(1));
    }

    #[test]
    fn prime_pi_dr_meissel_matches_baseline() {
        use super::prime_pi_dr_meissel;
        use crate::baseline::prime_pi;
        for x in [
            2u128,
            10,
            100,
            1_000,
            10_000,
            100_000,
            1_000_000,
            10_000_000,
            100_000_000,
            1_000_000_000,
            10_000_000_000,
            100_000_000_000,
            1_000_000_000_000,
        ] {
            assert_eq!(prime_pi_dr_meissel(x), prime_pi(x), "mismatch at x = {x}");
        }
    }
}
