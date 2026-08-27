#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct FastDivisorU32 {
    multiplier: u32,
    shift: u32,
    divisor: u32,
}

impl FastDivisorU32 {
    pub(super) fn new(divisor: u32) -> Option<Self> {
        if divisor == 0 {
            return None;
        }

        let floor_log_2 = 31 - divisor.leading_zeros();
        if divisor.is_power_of_two() {
            return Some(Self {
                multiplier: 0,
                shift: floor_log_2,
                divisor,
            });
        }

        // Generate the branch-free unsigned divider described by libdivide.
        // The CUDA side combines its final average and shift into one widened
        // `(numerator + multiply_high) >> shift` operation.
        let dividend = 1_u64 << (32 + floor_log_2);
        let proposed = dividend / u64::from(divisor);
        let remainder = (dividend - proposed * u64::from(divisor)) as u32;
        let mut proposed = proposed as u32;
        proposed = proposed.wrapping_add(proposed);
        let twice_remainder = remainder.wrapping_add(remainder);
        if twice_remainder >= divisor || twice_remainder < remainder {
            proposed = proposed.wrapping_add(1);
        }

        Some(Self {
            multiplier: proposed.wrapping_add(1),
            shift: floor_log_2 + 1,
            divisor,
        })
    }

    pub(super) fn launch_parameters(self) -> [u32; 3] {
        [self.multiplier, self.shift, self.divisor]
    }

    #[cfg(test)]
    fn divide(self, numerator: u32) -> u32 {
        let high_product = ((u64::from(numerator) * u64::from(self.multiplier)) >> 32) as u32;
        ((u64::from(numerator) + u64::from(high_product)) >> self.shift) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::FastDivisorU32;

    #[test]
    fn branch_free_parameters_match_authoritative_u32_division() {
        let mut state = 0x9e37_79b9_u32;
        for divisor in 1_u32..=4096 {
            let fast = FastDivisorU32::new(divisor).unwrap();
            let boundary_values = [
                0,
                1,
                divisor.saturating_sub(1),
                divisor,
                divisor.saturating_add(1),
                u32::MAX.saturating_sub(divisor),
                u32::MAX,
            ];
            for numerator in boundary_values {
                assert_eq!(fast.divide(numerator), numerator / divisor);
            }
            for _ in 0..128 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                assert_eq!(fast.divide(state), state / divisor);
            }
        }

        for divisor in [
            65_535,
            65_536,
            65_537,
            1_000_003,
            0x7fff_ffff,
            0x8000_0000,
            0x8000_0001,
            u32::MAX - 1,
            u32::MAX,
        ] {
            let fast = FastDivisorU32::new(divisor).unwrap();
            for numerator in [
                0,
                1,
                divisor.saturating_sub(1),
                divisor,
                divisor.saturating_add(1),
                u32::MAX - 1,
                u32::MAX,
            ] {
                assert_eq!(fast.divide(numerator), numerator / divisor);
            }
        }
    }

    #[test]
    fn zero_is_not_a_valid_divisor() {
        assert!(FastDivisorU32::new(0).is_none());
    }
}
