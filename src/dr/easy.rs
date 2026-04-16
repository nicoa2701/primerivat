use super::types::DrContext;
use crate::parameters::IndexDomain;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EasySpecializedWindow {
    pub residual: IndexDomain,
    pub transition: IndexDomain,
    pub specialized: IndexDomain,
    pub q_ref: Option<u128>,
    pub q_step: u128,
}

/// First scaffold for easy leaves.
///
/// In the current DR skeleton, `easy` is defined as the maximal suffix of the
/// active S2 window for which every term is exactly 1:
///
///   π(x / p_j) - j = 1
///
/// These leaves are mathematically simple because each index contributes a
/// small amount, so the whole suffix can be summed directly over the already
/// available S2 terms.
pub fn easy_range(ctx: &DrContext<'_>) -> IndexDomain {
    ctx.easy_domain()
}

pub fn easy_leaves(ctx: &DrContext<'_>) -> u128 {
    let range = easy_range(ctx);
    let mut sum = 0u128;

    for j in range.start..range.end {
        sum += ctx
            .s2_term_at(j)
            .expect("easy range must stay within the prime table");
    }

    sum
}

pub fn easy_specialized_window(ctx: &DrContext<'_>) -> EasySpecializedWindow {
    easy_specialized_window_in_range(ctx, easy_range(ctx))
}

pub fn easy_specialized_window_in_range(
    ctx: &DrContext<'_>,
    easy: IndexDomain,
) -> EasySpecializedWindow {
    if easy.is_empty() {
        return EasySpecializedWindow {
            residual: easy,
            transition: easy,
            specialized: easy,
            q_ref: None,
            q_step: 1,
        };
    }

    let last = easy.end - 1;
    let q_ref = ctx.quotient_at(last);
    let q_prev = last
        .checked_sub(1)
        .filter(|index| *index >= easy.start)
        .and_then(|index| ctx.quotient_at(index))
        .or(q_ref);
    let q_step = q_prev
        .zip(q_ref)
        .map(|(prev, current)| prev.saturating_sub(current).max(1))
        .unwrap_or(1);
    let mut specialized_start = easy.end;
    while specialized_start > easy.start {
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
    while transition_start > easy.start {
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

    EasySpecializedWindow {
        residual: IndexDomain {
            start: easy.start,
            end: transition_start,
        },
        transition: IndexDomain {
            start: transition_start,
            end: specialized_start,
        },
        specialized: IndexDomain {
            start: specialized_start,
            end: easy.end,
        },
        q_ref,
        q_step,
    }
}

pub fn easy_specialized_leaves(ctx: &DrContext<'_>) -> u128 {
    let window = easy_specialized_window(ctx);
    (window.specialized.start..window.specialized.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("specialized easy domain must stay within the prime table")
        })
        .sum()
}

pub fn easy_transition_leaves(ctx: &DrContext<'_>) -> u128 {
    let window = easy_specialized_window(ctx);
    (window.transition.start..window.transition.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("transition easy domain must stay within the prime table")
        })
        .sum()
}

pub fn easy_residual_leaves(ctx: &DrContext<'_>) -> u128 {
    let window = easy_specialized_window(ctx);
    (window.residual.start..window.residual.end)
        .map(|j| {
            ctx.s2_term_at(j)
                .expect("residual easy domain must stay within the prime table")
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use crate::dr::prepare_context;
    use crate::parameters::IndexDomain;

    use super::{
        easy_leaves, easy_range, easy_residual_leaves, easy_specialized_leaves,
        easy_specialized_window, easy_specialized_window_in_range, easy_transition_leaves,
    };

    #[test]
    fn easy_range_matches_active_prime_window() {
        let ctx = prepare_context(1_000_000);
        let range = easy_range(&ctx);
        let active = ctx.active_domain();
        let rule = ctx.s2_easy_domain().rule;

        assert!(range.start >= active.start);
        assert_eq!(range.end, active.end);
        for j in range.start..range.end {
            assert!(ctx.s2_term_at(j).is_some_and(|term| rule.matches(term)));
        }
    }

    #[test]
    fn easy_leaves_match_the_sum_of_terms_in_the_easy_suffix() {
        let ctx = prepare_context(100_000);
        let range = easy_range(&ctx);
        let expected: u128 = (range.start..range.end)
            .map(|j| ctx.s2_term_at(j).expect("easy term must exist"))
            .sum();
        assert_eq!(easy_leaves(&ctx), expected);
    }

    #[test]
    fn easy_range_is_a_suffix_of_unit_s2_terms() {
        let ctx = prepare_context(1_000_000);
        let range = easy_range(&ctx);
        let rule = ctx.s2_easy_domain().rule;

        if range.start > ctx.active_domain().start {
            assert!(
                !ctx.s2_term_at(range.start - 1)
                    .is_some_and(|term| rule.matches(term))
            );
        }
    }

    #[test]
    fn easy_specialized_window_stays_within_the_easy_suffix() {
        let ctx = prepare_context(20_000_000);
        let easy = easy_range(&ctx);
        let window = easy_specialized_window(&ctx);

        assert_eq!(window.residual.start, easy.start);
        assert_eq!(window.residual.end, window.transition.start);
        assert_eq!(window.transition.end, window.specialized.start);
        assert_eq!(window.specialized.end, easy.end);
        assert!(window.specialized.start >= easy.start);
    }

    #[test]
    fn easy_specialized_window_uses_the_local_quotient_step_guard() {
        let ctx = prepare_context(20_000_000);
        let window = easy_specialized_window(&ctx);

        if !window.specialized.is_empty() {
            let q_ref = window
                .q_ref
                .expect("non-empty specialized window must expose a quotient reference");

            for j in window.specialized.start..window.specialized.end {
                let q_j = ctx.quotient_at(j).expect("specialized quotient must exist");
                assert!(q_j <= q_ref);
            }

            if window.specialized.start > window.residual.start {
                if window.transition.is_empty() {
                    let previous = ctx
                        .quotient_at(window.specialized.start - 1)
                        .expect("previous quotient must exist");
                    assert!(previous > q_ref);
                }
            }
        }
    }

    #[test]
    fn easy_transition_window_stays_between_residual_and_specialized() {
        let ctx = prepare_context(20_000_000);
        let window = easy_specialized_window(&ctx);

        for j in window.transition.start..window.transition.end {
            let q_j = ctx.quotient_at(j).expect("transition quotient must exist");
            let q_ref = window.q_ref.expect("transition window must expose q_ref");
            assert!(q_j > q_ref);
            assert!(q_j <= q_ref.saturating_add(window.q_step));
        }
    }

    #[test]
    fn easy_specialized_window_can_split_a_larger_easy_range_more_strictly() {
        let ctx = prepare_context(20_000_000);
        let easy = IndexDomain {
            start: ctx.b.saturating_sub(3),
            end: ctx.b,
        };
        let window = easy_specialized_window_in_range(&ctx, easy);

        assert_eq!(window.specialized.end, easy.end);
        assert!(window.specialized.start >= easy.start);
        assert!(window.specialized.len() <= easy.len());
    }

    #[test]
    fn easy_specialized_and_residual_leaves_recompose_easy_sum() {
        let ctx = prepare_context(20_000_000);
        assert_eq!(
            easy_specialized_leaves(&ctx)
                + easy_transition_leaves(&ctx)
                + easy_residual_leaves(&ctx),
            easy_leaves(&ctx)
        );
    }
}
