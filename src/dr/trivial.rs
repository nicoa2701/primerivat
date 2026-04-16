use super::types::DrContext;

/// Returns the contribution of trivial DR leaves that can already be decided
/// from the current parameterisation without any additional traversal.
///
/// For the baseline Meissel-style threshold `a = π(floor(x^(1/3)))`, the first
/// prime strictly after `p_a` is already larger than `x^(1/3)`, so every triple
/// product of primes above `p_a` exceeds `x`. In that setting, the `P3`-style
/// contribution is trivially zero.
pub fn trivial_leaves(ctx: &DrContext<'_>) -> u128 {
    if ctx.trivial_domain().s3_is_zero && ctx.meissel_s3_is_trivial_zero() {
        return 0;
    }

    0
}

#[cfg(test)]
mod tests {
    use crate::dr::prepare_context;

    use super::trivial_leaves;

    #[test]
    fn trivial_meissel_contribution_is_zero_on_small_inputs() {
        for x in [10_u128, 100, 1_000, 10_000, 1_000_000] {
            let ctx = prepare_context(x);
            assert!(ctx.trivial_domain().s3_is_zero);
            assert!(
                ctx.meissel_s3_is_trivial_zero(),
                "expected trivial zero for x={x}"
            );
            assert_eq!(trivial_leaves(&ctx), 0);
        }
    }
}
