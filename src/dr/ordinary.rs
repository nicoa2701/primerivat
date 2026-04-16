use super::types::DrContext;
use crate::parameters::IndexDomain;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinarySpecializedRegion {
    pub residual: IndexDomain,
    pub pretransition: IndexDomain,
    pub transition: IndexDomain,
    pub specialized: IndexDomain,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryRelativeQuotientRegion {
    pub left_residual: IndexDomain,
    pub region: IndexDomain,
    pub right_residual: IndexDomain,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryRelativeQuotientShoulderRegion {
    pub left_residual: IndexDomain,
    pub left_shoulder: IndexDomain,
    pub core: IndexDomain,
    pub right_shoulder: IndexDomain,
    pub right_residual: IndexDomain,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryRelativeQuotientEnvelopeRegion {
    pub left_residual: IndexDomain,
    pub left_envelope: IndexDomain,
    pub core: IndexDomain,
    pub right_envelope: IndexDomain,
    pub right_residual: IndexDomain,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryRelativeQuotientHierarchyRegion {
    pub left_residual: IndexDomain,
    pub left_outer_band: IndexDomain,
    pub left_near_band: IndexDomain,
    pub inner_core: IndexDomain,
    pub right_near_band: IndexDomain,
    pub right_outer_band: IndexDomain,
    pub right_residual: IndexDomain,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryRegionAssembly {
    pub left_outer_support: IndexDomain,
    pub left_adjacent_support: IndexDomain,
    pub central_assembly: IndexDomain,
    pub right_adjacent_support: IndexDomain,
    pub right_outer_support: IndexDomain,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryQuasiLiteratureRegion {
    pub left_outer_work: IndexDomain,
    pub middle_work: IndexDomain,
    pub right_outer_work: IndexDomain,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdinaryDrLikeRegion {
    pub left_outer_work: IndexDomain,
    pub left_transfer_work: IndexDomain,
    pub central_work_region: IndexDomain,
    pub right_transfer_work: IndexDomain,
    pub right_outer_work: IndexDomain,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

/// Current ordinary window used by the DR skeleton.
///
/// The active S2 window is partitioned into:
/// - a lower `ordinary` prefix
/// - a middle `hard` segment
/// - an upper `easy` suffix where every term is exactly 1
///
/// `ordinary` is the lower half of the non-easy prefix.
pub fn ordinary_range(ctx: &DrContext<'_>) -> IndexDomain {
    ctx.ordinary_domain()
}

/// First non-trivial contribution computed inside `src/dr/`.
///
/// Numerically, this matches the serial form of the current `S2` summand over
/// the active prime window:
///
///   Σ_{j=a}^{b-1} (π(x / p_j) - j)
///
/// with `j` using the crate's 0-based indexing convention.
pub fn ordinary_leaf_sum(ctx: &DrContext<'_>) -> u128 {
    let range = ordinary_range(ctx);
    let mut sum = 0u128;

    for j in range.start..range.end {
        sum += ctx
            .s2_term_at(j)
            .expect("ordinary range must stay within the prime table");
    }

    sum
}

pub fn ordinary_specialized_region(ctx: &DrContext<'_>) -> OrdinarySpecializedRegion {
    ordinary_specialized_region_in_range(ctx, ordinary_range(ctx))
}

pub fn ordinary_relative_quotient_region(ctx: &DrContext<'_>) -> OrdinaryRelativeQuotientRegion {
    ordinary_relative_quotient_region_with_params_in_range(ctx, ordinary_range(ctx), 0, 1)
}

pub fn ordinary_relative_quotient_shoulder_region(
    ctx: &DrContext<'_>,
) -> OrdinaryRelativeQuotientShoulderRegion {
    ordinary_relative_quotient_shoulder_region_with_params_in_range(
        ctx,
        ordinary_range(ctx),
        0,
        1,
        2,
    )
}

pub fn ordinary_relative_quotient_envelope_region(
    ctx: &DrContext<'_>,
) -> OrdinaryRelativeQuotientEnvelopeRegion {
    ordinary_relative_quotient_envelope_region_in_range(ctx, ordinary_range(ctx))
}

pub fn ordinary_relative_quotient_hierarchy_region(
    ctx: &DrContext<'_>,
) -> OrdinaryRelativeQuotientHierarchyRegion {
    ordinary_relative_quotient_hierarchy_region_in_range(ctx, ordinary_range(ctx))
}

pub fn ordinary_region_assembly(ctx: &DrContext<'_>) -> OrdinaryRegionAssembly {
    ordinary_region_assembly_in_range(ctx, ordinary_range(ctx))
}

pub fn ordinary_quasi_literature_region(ctx: &DrContext<'_>) -> OrdinaryQuasiLiteratureRegion {
    ordinary_quasi_literature_region_in_range(ctx, ordinary_range(ctx))
}

pub fn ordinary_dr_like_region(ctx: &DrContext<'_>) -> OrdinaryDrLikeRegion {
    ordinary_dr_like_region_in_range(ctx, ordinary_range(ctx))
}

pub fn ordinary_specialized_region_in_range(
    ctx: &DrContext<'_>,
    ordinary: IndexDomain,
) -> OrdinarySpecializedRegion {
    if ordinary.is_empty() {
        return OrdinarySpecializedRegion {
            residual: ordinary,
            pretransition: ordinary,
            transition: ordinary,
            specialized: ordinary,
            q_ref: None,
            q_step: 1,
        };
    }

    let last = ordinary.end - 1;
    let q_ref = ctx.quotient_at(last);
    let q_prev = last
        .checked_sub(1)
        .filter(|index| *index >= ordinary.start)
        .and_then(|index| ctx.quotient_at(index))
        .or(q_ref);
    let q_step = q_prev
        .zip(q_ref)
        .map(|(prev, current)| prev.saturating_sub(current).max(1))
        .unwrap_or(1);

    let mut specialized_start = ordinary.end;
    while specialized_start > ordinary.start {
        let index = specialized_start - 1;
        let Some(q_j) = ctx.quotient_at(index) else {
            break;
        };
        if q_ref.is_some_and(|q| q_j <= q) {
            specialized_start -= 1;
        } else {
            break;
        }
    }

    let q_transition_limit = q_ref.map(|q| q.saturating_add(q_step));
    let mut transition_start = specialized_start;
    while transition_start > ordinary.start {
        let index = transition_start - 1;
        let Some(q_j) = ctx.quotient_at(index) else {
            break;
        };
        if q_ref.is_some_and(|q| q_j > q) && q_transition_limit.is_some_and(|limit| q_j <= limit) {
            transition_start -= 1;
        } else {
            break;
        }
    }

    let q_pretransition_limit = q_ref.map(|q| q.saturating_add(q_step.saturating_mul(2)));
    let mut pretransition_start = transition_start;
    while pretransition_start > ordinary.start {
        let index = pretransition_start - 1;
        let Some(q_j) = ctx.quotient_at(index) else {
            break;
        };
        if q_transition_limit.is_some_and(|lower| q_j > lower)
            && q_pretransition_limit.is_some_and(|upper| q_j <= upper)
        {
            pretransition_start -= 1;
        } else {
            break;
        }
    }

    OrdinarySpecializedRegion {
        residual: IndexDomain {
            start: ordinary.start,
            end: pretransition_start,
        },
        pretransition: IndexDomain {
            start: pretransition_start,
            end: transition_start,
        },
        transition: IndexDomain {
            start: transition_start,
            end: specialized_start,
        },
        specialized: IndexDomain {
            start: specialized_start,
            end: ordinary.end,
        },
        q_ref,
        q_step,
    }
}

pub fn ordinary_relative_quotient_region_in_range(
    ctx: &DrContext<'_>,
    ordinary: IndexDomain,
) -> OrdinaryRelativeQuotientRegion {
    ordinary_relative_quotient_region_with_params_in_range(ctx, ordinary, 0, 1)
}

pub fn ordinary_relative_quotient_region_with_params_in_range(
    ctx: &DrContext<'_>,
    ordinary: IndexDomain,
    center_shift: isize,
    step_scale: u128,
) -> OrdinaryRelativeQuotientRegion {
    if ordinary.is_empty() {
        return OrdinaryRelativeQuotientRegion {
            left_residual: ordinary,
            region: ordinary,
            right_residual: ordinary,
            q_ref: None,
            q_step: 1,
        };
    }

    let len = ordinary.len();
    let base_mid = ordinary.start + len / 2;
    let shifted_mid = if center_shift >= 0 {
        base_mid.saturating_add(center_shift as usize)
    } else {
        base_mid.saturating_sub(center_shift.unsigned_abs())
    };
    let mid = shifted_mid.clamp(ordinary.start, ordinary.end - 1);
    let q_ref = ctx.quotient_at(mid);
    let q_left = mid
        .checked_sub(1)
        .filter(|index| *index >= ordinary.start)
        .and_then(|index| ctx.quotient_at(index))
        .unwrap_or_else(|| q_ref.unwrap_or(0));
    let q_right = if mid + 1 < ordinary.end {
        ctx.quotient_at(mid + 1)
            .unwrap_or_else(|| q_ref.unwrap_or(0))
    } else {
        q_ref.unwrap_or(0)
    };
    let q_ref_value = q_ref.unwrap_or(0);
    let q_step = q_left
        .abs_diff(q_ref_value)
        .max(q_ref_value.abs_diff(q_right))
        .max(1);
    let q_limit = q_step.saturating_mul(step_scale.max(1));

    let mut region_start = mid;
    while region_start > ordinary.start {
        let index = region_start - 1;
        let Some(q_j) = ctx.quotient_at(index) else {
            break;
        };
        if q_j.abs_diff(q_ref_value) <= q_limit {
            region_start -= 1;
        } else {
            break;
        }
    }

    let mut region_end = mid + 1;
    while region_end < ordinary.end {
        let Some(q_j) = ctx.quotient_at(region_end) else {
            break;
        };
        if q_j.abs_diff(q_ref_value) <= q_limit {
            region_end += 1;
        } else {
            break;
        }
    }

    OrdinaryRelativeQuotientRegion {
        left_residual: IndexDomain {
            start: ordinary.start,
            end: region_start,
        },
        region: IndexDomain {
            start: region_start,
            end: region_end,
        },
        right_residual: IndexDomain {
            start: region_end,
            end: ordinary.end,
        },
        q_ref,
        q_step,
    }
}

pub fn ordinary_relative_quotient_shoulder_region_in_range(
    ctx: &DrContext<'_>,
    ordinary: IndexDomain,
) -> OrdinaryRelativeQuotientShoulderRegion {
    ordinary_relative_quotient_shoulder_region_with_params_in_range(ctx, ordinary, 0, 1, 2)
}

pub fn ordinary_relative_quotient_shoulder_region_with_params_in_range(
    ctx: &DrContext<'_>,
    ordinary: IndexDomain,
    center_shift: isize,
    core_step_scale: u128,
    shoulder_step_scale: u128,
) -> OrdinaryRelativeQuotientShoulderRegion {
    if ordinary.is_empty() {
        return OrdinaryRelativeQuotientShoulderRegion {
            left_residual: ordinary,
            left_shoulder: ordinary,
            core: ordinary,
            right_shoulder: ordinary,
            right_residual: ordinary,
            q_ref: None,
            q_step: 1,
        };
    }

    let core_scale = core_step_scale.max(1);
    let outer_scale = shoulder_step_scale.max(core_scale);
    let core = ordinary_relative_quotient_region_with_params_in_range(
        ctx,
        ordinary,
        center_shift,
        core_scale,
    );
    let outer = ordinary_relative_quotient_region_with_params_in_range(
        ctx,
        ordinary,
        center_shift,
        outer_scale,
    );

    OrdinaryRelativeQuotientShoulderRegion {
        left_residual: IndexDomain {
            start: ordinary.start,
            end: outer.region.start,
        },
        left_shoulder: IndexDomain {
            start: outer.region.start,
            end: core.region.start,
        },
        core: core.region,
        right_shoulder: IndexDomain {
            start: core.region.end,
            end: outer.region.end,
        },
        right_residual: IndexDomain {
            start: outer.region.end,
            end: ordinary.end,
        },
        q_ref: core.q_ref,
        q_step: core.q_step,
    }
}

pub fn ordinary_relative_quotient_envelope_region_in_range(
    ctx: &DrContext<'_>,
    ordinary: IndexDomain,
) -> OrdinaryRelativeQuotientEnvelopeRegion {
    ordinary_relative_quotient_envelope_region_with_params_in_range(ctx, ordinary, 0, 1, 3)
}

pub fn ordinary_relative_quotient_envelope_region_with_params_in_range(
    ctx: &DrContext<'_>,
    ordinary: IndexDomain,
    center_shift: isize,
    core_step_scale: u128,
    envelope_step_scale: u128,
) -> OrdinaryRelativeQuotientEnvelopeRegion {
    if ordinary.is_empty() {
        return OrdinaryRelativeQuotientEnvelopeRegion {
            left_residual: ordinary,
            left_envelope: ordinary,
            core: ordinary,
            right_envelope: ordinary,
            right_residual: ordinary,
            q_ref: None,
            q_step: 1,
        };
    }

    let core_scale = core_step_scale.max(1);
    let outer_scale = envelope_step_scale.max(core_scale);
    let core = ordinary_relative_quotient_region_with_params_in_range(
        ctx,
        ordinary,
        center_shift,
        core_scale,
    );
    let outer = ordinary_relative_quotient_region_with_params_in_range(
        ctx,
        ordinary,
        center_shift,
        outer_scale,
    );

    OrdinaryRelativeQuotientEnvelopeRegion {
        left_residual: IndexDomain {
            start: ordinary.start,
            end: outer.region.start,
        },
        left_envelope: IndexDomain {
            start: outer.region.start,
            end: core.region.start,
        },
        core: core.region,
        right_envelope: IndexDomain {
            start: core.region.end,
            end: outer.region.end,
        },
        right_residual: IndexDomain {
            start: outer.region.end,
            end: ordinary.end,
        },
        q_ref: core.q_ref,
        q_step: core.q_step,
    }
}

pub fn ordinary_relative_quotient_hierarchy_region_in_range(
    ctx: &DrContext<'_>,
    ordinary: IndexDomain,
) -> OrdinaryRelativeQuotientHierarchyRegion {
    if ordinary.is_empty() {
        return OrdinaryRelativeQuotientHierarchyRegion {
            left_residual: ordinary,
            left_outer_band: ordinary,
            left_near_band: ordinary,
            inner_core: ordinary,
            right_near_band: ordinary,
            right_outer_band: ordinary,
            right_residual: ordinary,
            q_ref: None,
            q_step: 1,
        };
    }

    let shoulder = ordinary_relative_quotient_shoulder_region_in_range(ctx, ordinary);
    let envelope = ordinary_relative_quotient_envelope_region_in_range(ctx, ordinary);

    OrdinaryRelativeQuotientHierarchyRegion {
        left_residual: envelope.left_residual,
        left_outer_band: IndexDomain {
            start: envelope.left_envelope.start,
            end: shoulder.left_shoulder.start,
        },
        left_near_band: shoulder.left_shoulder,
        inner_core: shoulder.core,
        right_near_band: shoulder.right_shoulder,
        right_outer_band: IndexDomain {
            start: shoulder.right_shoulder.end,
            end: envelope.right_envelope.end,
        },
        right_residual: envelope.right_residual,
        q_ref: shoulder.q_ref,
        q_step: shoulder.q_step,
    }
}

pub fn ordinary_region_assembly_in_range(
    ctx: &DrContext<'_>,
    ordinary: IndexDomain,
) -> OrdinaryRegionAssembly {
    if ordinary.is_empty() {
        return OrdinaryRegionAssembly {
            left_outer_support: ordinary,
            left_adjacent_support: ordinary,
            central_assembly: ordinary,
            right_adjacent_support: ordinary,
            right_outer_support: ordinary,
            q_ref: None,
            q_step: 1,
        };
    }

    let hierarchy = ordinary_relative_quotient_hierarchy_region_in_range(ctx, ordinary);

    OrdinaryRegionAssembly {
        left_outer_support: IndexDomain {
            start: hierarchy.left_residual.start,
            end: hierarchy.left_outer_band.end,
        },
        left_adjacent_support: hierarchy.left_near_band,
        central_assembly: hierarchy.inner_core,
        right_adjacent_support: hierarchy.right_near_band,
        right_outer_support: IndexDomain {
            start: hierarchy.right_outer_band.start,
            end: hierarchy.right_residual.end,
        },
        q_ref: hierarchy.q_ref,
        q_step: hierarchy.q_step,
    }
}

pub fn ordinary_quasi_literature_region_in_range(
    ctx: &DrContext<'_>,
    ordinary: IndexDomain,
) -> OrdinaryQuasiLiteratureRegion {
    if ordinary.is_empty() {
        return OrdinaryQuasiLiteratureRegion {
            left_outer_work: ordinary,
            middle_work: ordinary,
            right_outer_work: ordinary,
            q_ref: None,
            q_step: 1,
        };
    }

    let assembly = ordinary_region_assembly_in_range(ctx, ordinary);
    let middle_start = assembly
        .left_adjacent_support
        .start
        .min(assembly.central_assembly.start);
    let middle_end = assembly
        .right_adjacent_support
        .end
        .max(assembly.central_assembly.end);

    OrdinaryQuasiLiteratureRegion {
        left_outer_work: assembly.left_outer_support,
        middle_work: IndexDomain {
            start: middle_start,
            end: middle_end,
        },
        right_outer_work: assembly.right_outer_support,
        q_ref: assembly.q_ref,
        q_step: assembly.q_step,
    }
}

pub fn ordinary_dr_like_region_in_range(
    ctx: &DrContext<'_>,
    ordinary: IndexDomain,
) -> OrdinaryDrLikeRegion {
    if ordinary.is_empty() {
        return OrdinaryDrLikeRegion {
            left_outer_work: ordinary,
            left_transfer_work: ordinary,
            central_work_region: ordinary,
            right_transfer_work: ordinary,
            right_outer_work: ordinary,
            q_ref: None,
            q_step: 1,
        };
    }

    let assembly = ordinary_region_assembly_in_range(ctx, ordinary);
    let quasi = ordinary_quasi_literature_region_in_range(ctx, ordinary);

    let central_start = assembly.central_assembly.start;
    let central_end = assembly.central_assembly.end;
    let expanded_start = central_start.saturating_sub(1).max(quasi.middle_work.start);
    let expanded_end = (central_end + 1).min(quasi.middle_work.end);

    OrdinaryDrLikeRegion {
        left_outer_work: quasi.left_outer_work,
        left_transfer_work: IndexDomain {
            start: quasi.middle_work.start,
            end: expanded_start,
        },
        central_work_region: IndexDomain {
            start: expanded_start,
            end: expanded_end,
        },
        right_transfer_work: IndexDomain {
            start: expanded_end,
            end: quasi.middle_work.end,
        },
        right_outer_work: quasi.right_outer_work,
        q_ref: quasi.q_ref,
        q_step: quasi.q_step,
    }
}

pub fn ordinary_specialized_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_specialized_region(ctx);
    (region.specialized.start..region.specialized.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary specialized domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_transition_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_specialized_region(ctx);
    (region.transition.start..region.transition.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary transition domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_pretransition_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_specialized_region(ctx);
    (region.pretransition.start..region.pretransition.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary pretransition domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_residual_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_specialized_region(ctx);
    (region.residual.start..region.residual.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary residual domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_relative_region_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_relative_quotient_region(ctx);
    (region.region.start..region.region.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary relative quotient region must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_left_residual_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_relative_quotient_region(ctx);
    (region.left_residual.start..region.left_residual.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary left residual domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_right_residual_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_relative_quotient_region(ctx);
    (region.right_residual.start..region.right_residual.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary right residual domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_dr_like_left_outer_work_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_dr_like_region(ctx);
    (region.left_outer_work.start..region.left_outer_work.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary left outer work domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_dr_like_left_transfer_work_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_dr_like_region(ctx);
    (region.left_transfer_work.start..region.left_transfer_work.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary left transfer work domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_dr_like_central_work_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_dr_like_region(ctx);
    (region.central_work_region.start..region.central_work_region.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary central work region must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_dr_like_right_transfer_work_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_dr_like_region(ctx);
    (region.right_transfer_work.start..region.right_transfer_work.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary right transfer work domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_dr_like_right_outer_work_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_dr_like_region(ctx);
    (region.right_outer_work.start..region.right_outer_work.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary right outer work domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_left_shoulder_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_relative_quotient_shoulder_region(ctx);
    (region.left_shoulder.start..region.left_shoulder.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary left shoulder domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_core_region_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_relative_quotient_shoulder_region(ctx);
    (region.core.start..region.core.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary core region domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_right_shoulder_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_relative_quotient_shoulder_region(ctx);
    (region.right_shoulder.start..region.right_shoulder.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary right shoulder domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_left_envelope_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_relative_quotient_envelope_region(ctx);
    (region.left_envelope.start..region.left_envelope.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary left envelope domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_envelope_core_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_relative_quotient_envelope_region(ctx);
    (region.core.start..region.core.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary envelope core domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_right_envelope_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_relative_quotient_envelope_region(ctx);
    (region.right_envelope.start..region.right_envelope.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary right envelope domain must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_left_outer_band_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_relative_quotient_hierarchy_region(ctx);
    (region.left_outer_band.start..region.left_outer_band.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary left outer band must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_left_near_band_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_relative_quotient_hierarchy_region(ctx);
    (region.left_near_band.start..region.left_near_band.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary left near band must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_inner_core_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_relative_quotient_hierarchy_region(ctx);
    (region.inner_core.start..region.inner_core.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary inner core must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_right_near_band_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_relative_quotient_hierarchy_region(ctx);
    (region.right_near_band.start..region.right_near_band.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary right near band must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_right_outer_band_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_relative_quotient_hierarchy_region(ctx);
    (region.right_outer_band.start..region.right_outer_band.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary right outer band must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_left_outer_support_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_region_assembly(ctx);
    (region.left_outer_support.start..region.left_outer_support.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary left outer support must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_left_adjacent_support_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_region_assembly(ctx);
    (region.left_adjacent_support.start..region.left_adjacent_support.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary left adjacent support must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_central_assembly_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_region_assembly(ctx);
    (region.central_assembly.start..region.central_assembly.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary central assembly must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_right_adjacent_support_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_region_assembly(ctx);
    (region.right_adjacent_support.start..region.right_adjacent_support.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary right adjacent support must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_right_outer_support_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_region_assembly(ctx);
    (region.right_outer_support.start..region.right_outer_support.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary right outer support must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_left_outer_work_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_quasi_literature_region(ctx);
    (region.left_outer_work.start..region.left_outer_work.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary left outer work must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_middle_work_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_quasi_literature_region(ctx);
    (region.middle_work.start..region.middle_work.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary middle work must stay within the prime table")
        })
        .sum()
}

pub fn ordinary_right_outer_work_leaves(ctx: &DrContext<'_>) -> u128 {
    let region = ordinary_quasi_literature_region(ctx);
    (region.right_outer_work.start..region.right_outer_work.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("ordinary right outer work must stay within the prime table")
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use crate::{dr::prepare_context, parameters::IndexDomain};

    use super::{
        ordinary_central_assembly_leaves, ordinary_core_region_leaves,
        ordinary_dr_like_central_work_leaves, ordinary_dr_like_left_outer_work_leaves,
        ordinary_dr_like_left_transfer_work_leaves, ordinary_dr_like_region,
        ordinary_dr_like_region_in_range, ordinary_dr_like_right_outer_work_leaves,
        ordinary_dr_like_right_transfer_work_leaves, ordinary_envelope_core_leaves,
        ordinary_inner_core_leaves, ordinary_leaf_sum, ordinary_left_adjacent_support_leaves,
        ordinary_left_envelope_leaves, ordinary_left_near_band_leaves,
        ordinary_left_outer_band_leaves, ordinary_left_outer_support_leaves,
        ordinary_left_outer_work_leaves, ordinary_left_residual_leaves,
        ordinary_left_shoulder_leaves, ordinary_middle_work_leaves, ordinary_pretransition_leaves,
        ordinary_quasi_literature_region, ordinary_quasi_literature_region_in_range,
        ordinary_range, ordinary_region_assembly, ordinary_region_assembly_in_range,
        ordinary_relative_quotient_envelope_region,
        ordinary_relative_quotient_envelope_region_in_range,
        ordinary_relative_quotient_envelope_region_with_params_in_range,
        ordinary_relative_quotient_hierarchy_region,
        ordinary_relative_quotient_hierarchy_region_in_range, ordinary_relative_quotient_region,
        ordinary_relative_quotient_region_in_range,
        ordinary_relative_quotient_region_with_params_in_range,
        ordinary_relative_quotient_shoulder_region,
        ordinary_relative_quotient_shoulder_region_in_range,
        ordinary_relative_quotient_shoulder_region_with_params_in_range,
        ordinary_relative_region_leaves, ordinary_residual_leaves,
        ordinary_right_adjacent_support_leaves, ordinary_right_envelope_leaves,
        ordinary_right_near_band_leaves, ordinary_right_outer_band_leaves,
        ordinary_right_outer_support_leaves, ordinary_right_outer_work_leaves,
        ordinary_right_residual_leaves, ordinary_right_shoulder_leaves,
        ordinary_specialized_leaves, ordinary_specialized_region,
        ordinary_specialized_region_in_range, ordinary_transition_leaves,
    };

    #[test]
    fn ordinary_range_matches_active_prime_window() {
        let ctx = prepare_context(1_000_000);
        let range = ordinary_range(&ctx);
        let hard = crate::dr::hard::hard_range(&ctx);
        let rule = ctx.s1_domain().rule;

        assert_eq!(range.start, ctx.active_domain().start);
        assert_eq!(range.end, hard.start);
        for j in range.start..range.end {
            let term = ctx.s2_term_at(j).expect("ordinary index must be valid");
            assert!(term > rule.term_min_exclusive);
        }
    }

    #[test]
    fn ordinary_sum_is_bounded_by_baseline_s2_in_serial_form() {
        let ctx = prepare_context(1_000_000);
        let ordinary = ordinary_leaf_sum(&ctx);
        let baseline_s2 = crate::baseline::s2::s2(
            ctx.x,
            ctx.a,
            ctx.z_usize,
            ctx.primes.as_slice(),
            ctx.small,
            ctx.large,
            1,
        );

        assert!(ordinary <= baseline_s2);
    }

    #[test]
    fn ordinary_specialized_region_stays_within_the_ordinary_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = ordinary_range(&ctx);
        let region = ordinary_specialized_region(&ctx);

        assert_eq!(region.residual.start, ordinary.start);
        assert_eq!(region.residual.end, region.pretransition.start);
        assert_eq!(region.pretransition.end, region.transition.start);
        assert_eq!(region.transition.end, region.specialized.start);
        assert_eq!(region.specialized.end, ordinary.end);
    }

    #[test]
    fn ordinary_specialized_region_uses_the_terminal_quotient_plateau() {
        let ctx = prepare_context(20_000_000);
        let region = ordinary_specialized_region(&ctx);

        if !region.specialized.is_empty() {
            let q_ref = region
                .q_ref
                .expect("non-empty ordinary specialized region must expose q_ref");
            for j in region.specialized.start..region.specialized.end {
                let q_j = ctx
                    .quotient_at(j)
                    .expect("ordinary specialized quotient must exist");
                assert!(q_j <= q_ref);
            }
        }
    }

    #[test]
    fn ordinary_transition_region_stays_between_residual_and_specialized() {
        let ctx = prepare_context(20_000_000);
        let region = ordinary_specialized_region(&ctx);

        for j in region.transition.start..region.transition.end {
            let q_j = ctx
                .quotient_at(j)
                .expect("ordinary transition quotient must exist");
            let q_ref = region.q_ref.expect("transition region must expose q_ref");
            assert!(q_j > q_ref);
            assert!(q_j <= q_ref.saturating_add(region.q_step));
        }
    }

    #[test]
    fn ordinary_pretransition_region_stays_before_transition() {
        let ctx = prepare_context(20_000_000);
        let region = ordinary_specialized_region(&ctx);

        for j in region.pretransition.start..region.pretransition.end {
            let q_j = ctx
                .quotient_at(j)
                .expect("ordinary pretransition quotient must exist");
            let q_ref = region
                .q_ref
                .expect("pretransition region must expose q_ref");
            let lower = q_ref.saturating_add(region.q_step);
            let upper = q_ref.saturating_add(region.q_step.saturating_mul(2));
            assert!(q_j > lower);
            assert!(q_j <= upper);
        }
    }

    #[test]
    fn ordinary_specialized_region_can_split_a_larger_ordinary_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = IndexDomain {
            start: ctx.a,
            end: ctx.a.saturating_add(4).min(ctx.b),
        };
        let region = ordinary_specialized_region_in_range(&ctx, ordinary);

        assert_eq!(region.specialized.end, ordinary.end);
        assert!(region.specialized.start >= ordinary.start);
    }

    #[test]
    fn ordinary_specialized_and_residual_recompose_ordinary_sum() {
        let ctx = prepare_context(20_000_000);
        assert_eq!(
            ordinary_residual_leaves(&ctx)
                + ordinary_pretransition_leaves(&ctx)
                + ordinary_transition_leaves(&ctx)
                + ordinary_specialized_leaves(&ctx),
            ordinary_leaf_sum(&ctx)
        );
    }

    #[test]
    fn ordinary_relative_quotient_region_stays_within_the_ordinary_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = ordinary_range(&ctx);
        let region = ordinary_relative_quotient_region(&ctx);

        assert_eq!(region.left_residual.start, ordinary.start);
        assert_eq!(region.left_residual.end, region.region.start);
        assert_eq!(region.region.end, region.right_residual.start);
        assert_eq!(region.right_residual.end, ordinary.end);
    }

    #[test]
    fn ordinary_relative_quotient_region_stays_within_one_local_step() {
        let ctx = prepare_context(20_000_000);
        let region = ordinary_relative_quotient_region(&ctx);

        if !region.region.is_empty() {
            let q_ref = region
                .q_ref
                .expect("non-empty ordinary relative region must expose q_ref");
            for j in region.region.start..region.region.end {
                let q_j = ctx
                    .quotient_at(j)
                    .expect("ordinary relative quotient must exist");
                assert!(q_j.abs_diff(q_ref) <= region.q_step);
            }
        }
    }

    #[test]
    fn ordinary_relative_quotient_region_can_split_a_larger_ordinary_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = IndexDomain {
            start: ctx.a,
            end: ctx.a.saturating_add(12).min(ctx.b),
        };
        let region = ordinary_relative_quotient_region_in_range(&ctx, ordinary);

        assert!(region.region.start >= ordinary.start);
        assert!(region.region.end <= ordinary.end);
    }

    #[test]
    fn ordinary_relative_quotient_region_with_params_stays_within_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = ordinary_range(&ctx);
        let region = ordinary_relative_quotient_region_with_params_in_range(&ctx, ordinary, 1, 2);

        assert!(region.region.start >= ordinary.start);
        assert!(region.region.end <= ordinary.end);
        assert_eq!(region.left_residual.end, region.region.start);
        assert_eq!(region.region.end, region.right_residual.start);
    }

    #[test]
    fn ordinary_relative_region_and_residuals_recompose_ordinary_sum() {
        let ctx = prepare_context(20_000_000);
        assert_eq!(
            ordinary_left_residual_leaves(&ctx)
                + ordinary_relative_region_leaves(&ctx)
                + ordinary_right_residual_leaves(&ctx),
            ordinary_leaf_sum(&ctx)
        );
    }

    #[test]
    fn ordinary_relative_quotient_shoulder_region_stays_within_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = ordinary_range(&ctx);
        let region = ordinary_relative_quotient_shoulder_region(&ctx);

        assert_eq!(region.left_residual.start, ordinary.start);
        assert_eq!(region.left_residual.end, region.left_shoulder.start);
        assert_eq!(region.left_shoulder.end, region.core.start);
        assert_eq!(region.core.end, region.right_shoulder.start);
        assert_eq!(region.right_shoulder.end, region.right_residual.start);
        assert_eq!(region.right_residual.end, ordinary.end);
    }

    #[test]
    fn ordinary_relative_quotient_shoulder_region_can_split_a_larger_ordinary_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = IndexDomain {
            start: ctx.a,
            end: ctx.a.saturating_add(24).min(ctx.b),
        };
        let region = ordinary_relative_quotient_shoulder_region_in_range(&ctx, ordinary);

        assert!(region.core.start >= ordinary.start);
        assert!(region.core.end <= ordinary.end);
        assert!(region.left_shoulder.start >= ordinary.start);
        assert!(region.right_shoulder.end <= ordinary.end);
    }

    #[test]
    fn ordinary_relative_quotient_shoulder_region_with_params_stays_within_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = ordinary_range(&ctx);
        let region = ordinary_relative_quotient_shoulder_region_with_params_in_range(
            &ctx, ordinary, 1, 1, 3,
        );

        assert!(region.core.start >= ordinary.start);
        assert!(region.right_residual.end <= ordinary.end);
        assert_eq!(region.left_residual.end, region.left_shoulder.start);
        assert_eq!(region.left_shoulder.end, region.core.start);
        assert_eq!(region.core.end, region.right_shoulder.start);
        assert_eq!(region.right_shoulder.end, region.right_residual.start);
    }

    #[test]
    fn ordinary_relative_quotient_shoulder_regions_recompose_ordinary_sum() {
        let ctx = prepare_context(20_000_000);
        assert_eq!(
            ordinary_left_residual_leaves(&ctx)
                + ordinary_left_shoulder_leaves(&ctx)
                + ordinary_core_region_leaves(&ctx)
                + ordinary_right_shoulder_leaves(&ctx)
                + ordinary_right_residual_leaves(&ctx),
            ordinary_leaf_sum(&ctx)
        );
    }

    #[test]
    fn ordinary_relative_quotient_envelope_region_stays_within_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = ordinary_range(&ctx);
        let region = ordinary_relative_quotient_envelope_region(&ctx);

        assert_eq!(region.left_residual.start, ordinary.start);
        assert_eq!(region.left_residual.end, region.left_envelope.start);
        assert_eq!(region.left_envelope.end, region.core.start);
        assert_eq!(region.core.end, region.right_envelope.start);
        assert_eq!(region.right_envelope.end, region.right_residual.start);
        assert_eq!(region.right_residual.end, ordinary.end);
    }

    #[test]
    fn ordinary_relative_quotient_envelope_region_can_split_a_larger_ordinary_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = IndexDomain {
            start: ctx.a,
            end: ctx.a.saturating_add(24).min(ctx.b),
        };
        let region = ordinary_relative_quotient_envelope_region_in_range(&ctx, ordinary);

        assert!(region.core.start >= ordinary.start);
        assert!(region.core.end <= ordinary.end);
        assert!(region.left_envelope.start >= ordinary.start);
        assert!(region.right_envelope.end <= ordinary.end);
    }

    #[test]
    fn ordinary_relative_quotient_envelope_region_with_params_stays_within_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = ordinary_range(&ctx);
        let region = ordinary_relative_quotient_envelope_region_with_params_in_range(
            &ctx, ordinary, 1, 1, 4,
        );

        assert_eq!(region.left_residual.end, region.left_envelope.start);
        assert_eq!(region.left_envelope.end, region.core.start);
        assert_eq!(region.core.end, region.right_envelope.start);
        assert_eq!(region.right_envelope.end, region.right_residual.start);
        assert!(region.right_residual.end <= ordinary.end);
    }

    #[test]
    fn ordinary_relative_quotient_envelope_regions_recompose_ordinary_sum() {
        let ctx = prepare_context(20_000_000);
        let region = ordinary_relative_quotient_envelope_region(&ctx);
        assert_eq!(
            (region.left_residual.start..region.left_residual.end)
                .map(|j| {
                    ctx.s2_term_at(j)
                        .expect("ordinary envelope left residual must stay within range")
                })
                .sum::<u128>()
                + ordinary_left_envelope_leaves(&ctx)
                + ordinary_envelope_core_leaves(&ctx)
                + ordinary_right_envelope_leaves(&ctx)
                + (region.right_residual.start..region.right_residual.end)
                    .map(|j| {
                        ctx.s2_term_at(j)
                            .expect("ordinary envelope right residual must stay within range")
                    })
                    .sum::<u128>(),
            ordinary_leaf_sum(&ctx)
        );
    }

    #[test]
    fn ordinary_relative_quotient_hierarchy_region_stays_within_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = ordinary_range(&ctx);
        let region = ordinary_relative_quotient_hierarchy_region(&ctx);

        assert_eq!(region.left_residual.start, ordinary.start);
        assert_eq!(region.left_residual.end, region.left_outer_band.start);
        assert_eq!(region.left_outer_band.end, region.left_near_band.start);
        assert_eq!(region.left_near_band.end, region.inner_core.start);
        assert_eq!(region.inner_core.end, region.right_near_band.start);
        assert_eq!(region.right_near_band.end, region.right_outer_band.start);
        assert_eq!(region.right_outer_band.end, region.right_residual.start);
        assert_eq!(region.right_residual.end, ordinary.end);
    }

    #[test]
    fn ordinary_relative_quotient_hierarchy_region_can_split_a_larger_ordinary_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = IndexDomain {
            start: ctx.a,
            end: ctx.a.saturating_add(24).min(ctx.b),
        };
        let region = ordinary_relative_quotient_hierarchy_region_in_range(&ctx, ordinary);

        assert!(region.inner_core.start >= ordinary.start);
        assert!(region.right_residual.end <= ordinary.end);
    }

    #[test]
    fn ordinary_relative_quotient_hierarchy_regions_recompose_ordinary_sum() {
        let ctx = prepare_context(20_000_000);
        let region = ordinary_relative_quotient_hierarchy_region(&ctx);
        assert_eq!(
            (region.left_residual.start..region.left_residual.end)
                .map(|j| ctx
                    .s2_term_at(j)
                    .expect("left residual index must be valid"))
                .sum::<u128>()
                + ordinary_left_outer_band_leaves(&ctx)
                + ordinary_left_near_band_leaves(&ctx)
                + ordinary_inner_core_leaves(&ctx)
                + ordinary_right_near_band_leaves(&ctx)
                + ordinary_right_outer_band_leaves(&ctx)
                + (region.right_residual.start..region.right_residual.end)
                    .map(|j| ctx
                        .s2_term_at(j)
                        .expect("right residual index must be valid"))
                    .sum::<u128>(),
            ordinary_leaf_sum(&ctx)
        );
    }

    #[test]
    fn ordinary_region_assembly_stays_within_range() {
        let ctx = prepare_context(20_000_000);
        let region = ordinary_region_assembly(&ctx);

        assert_eq!(
            region.left_outer_support.end,
            region.left_adjacent_support.start
        );
        assert_eq!(
            region.left_adjacent_support.end,
            region.central_assembly.start
        );
        assert_eq!(
            region.central_assembly.end,
            region.right_adjacent_support.start
        );
        assert_eq!(
            region.right_adjacent_support.end,
            region.right_outer_support.start
        );
    }

    #[test]
    fn ordinary_region_assembly_can_split_a_larger_ordinary_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = IndexDomain {
            start: ctx.a,
            end: ctx.b.saturating_sub(1),
        };
        let region = ordinary_region_assembly_in_range(&ctx, ordinary);

        assert!(region.central_assembly.start >= ordinary.start);
        assert!(region.right_outer_support.end <= ordinary.end);
    }

    #[test]
    fn ordinary_region_assembly_recomposes_ordinary_sum() {
        let ctx = prepare_context(20_000_000);
        let region = ordinary_region_assembly(&ctx);

        assert_eq!(
            region.left_outer_support.len()
                + region.left_adjacent_support.len()
                + region.central_assembly.len()
                + region.right_adjacent_support.len()
                + region.right_outer_support.len(),
            ordinary_range(&ctx).len()
        );
        assert_eq!(
            ordinary_left_outer_support_leaves(&ctx)
                + ordinary_left_adjacent_support_leaves(&ctx)
                + ordinary_central_assembly_leaves(&ctx)
                + ordinary_right_adjacent_support_leaves(&ctx)
                + ordinary_right_outer_support_leaves(&ctx),
            ordinary_leaf_sum(&ctx)
        );
    }

    #[test]
    fn ordinary_quasi_literature_region_stays_within_range() {
        let ctx = prepare_context(20_000_000);
        let region = ordinary_quasi_literature_region(&ctx);

        assert_eq!(region.left_outer_work.end, region.middle_work.start);
        assert_eq!(region.middle_work.end, region.right_outer_work.start);
    }

    #[test]
    fn ordinary_quasi_literature_region_can_split_a_larger_ordinary_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = IndexDomain {
            start: ctx.a,
            end: ctx.b.saturating_sub(1),
        };
        let region = ordinary_quasi_literature_region_in_range(&ctx, ordinary);

        assert!(region.middle_work.start >= ordinary.start);
        assert!(region.right_outer_work.end <= ordinary.end);
    }

    #[test]
    fn ordinary_quasi_literature_region_recomposes_ordinary_sum() {
        let ctx = prepare_context(20_000_000);
        let region = ordinary_quasi_literature_region(&ctx);

        assert_eq!(
            region.left_outer_work.len() + region.middle_work.len() + region.right_outer_work.len(),
            ordinary_range(&ctx).len()
        );
        assert_eq!(
            ordinary_left_outer_work_leaves(&ctx)
                + ordinary_middle_work_leaves(&ctx)
                + ordinary_right_outer_work_leaves(&ctx),
            ordinary_leaf_sum(&ctx)
        );
    }

    #[test]
    fn ordinary_dr_like_region_stays_within_range() {
        let ctx = prepare_context(20_000_000);
        let region = ordinary_dr_like_region(&ctx);

        assert_eq!(region.left_outer_work.end, region.left_transfer_work.start);
        assert_eq!(
            region.left_transfer_work.end,
            region.central_work_region.start
        );
        assert_eq!(
            region.central_work_region.end,
            region.right_transfer_work.start
        );
        assert_eq!(
            region.right_transfer_work.end,
            region.right_outer_work.start
        );
    }

    #[test]
    fn ordinary_dr_like_region_can_split_a_larger_ordinary_range() {
        let ctx = prepare_context(20_000_000);
        let ordinary = IndexDomain {
            start: ctx.a,
            end: ctx.b.saturating_sub(1),
        };
        let region = ordinary_dr_like_region_in_range(&ctx, ordinary);

        assert!(region.central_work_region.start >= ordinary.start);
        assert!(region.right_outer_work.end <= ordinary.end);
    }

    #[test]
    fn ordinary_dr_like_region_recomposes_ordinary_sum() {
        let ctx = prepare_context(20_000_000);
        let region = ordinary_dr_like_region(&ctx);

        assert_eq!(
            region.left_outer_work.len()
                + region.left_transfer_work.len()
                + region.central_work_region.len()
                + region.right_transfer_work.len()
                + region.right_outer_work.len(),
            ordinary_range(&ctx).len()
        );
        assert_eq!(
            ordinary_dr_like_left_outer_work_leaves(&ctx)
                + ordinary_dr_like_left_transfer_work_leaves(&ctx)
                + ordinary_dr_like_central_work_leaves(&ctx)
                + ordinary_dr_like_right_transfer_work_leaves(&ctx)
                + ordinary_dr_like_right_outer_work_leaves(&ctx),
            ordinary_leaf_sum(&ctx)
        );
    }
}
