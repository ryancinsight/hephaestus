//! Host-side Bluestein coefficient construction.

pub(super) fn forward_radix_two(values: &mut [[f64; 2]]) {
    let n = values.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two());

    let mut reversed = 0usize;
    for index in 1..n {
        let mut bit = n >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            values.swap(index, reversed);
        }
    }

    let mut span = 2usize;
    loop {
        let angle = -core::f64::consts::TAU / span as f64;
        let step = [angle.cos(), angle.sin()];
        for start in (0..n).step_by(span) {
            let mut twiddle = [1.0, 0.0];
            for offset in 0..span / 2 {
                let even = values[start + offset];
                let odd = values[start + offset + span / 2];
                let rotated = [
                    odd[0].mul_add(twiddle[0], -(odd[1] * twiddle[1])),
                    odd[0].mul_add(twiddle[1], odd[1] * twiddle[0]),
                ];
                values[start + offset] = [even[0] + rotated[0], even[1] + rotated[1]];
                values[start + offset + span / 2] = [even[0] - rotated[0], even[1] - rotated[1]];
                twiddle = [
                    twiddle[0].mul_add(step[0], -(twiddle[1] * step[1])),
                    twiddle[0].mul_add(step[1], twiddle[1] * step[0]),
                ];
            }
        }
        if span == n {
            break;
        }
        span *= 2;
    }
}

pub(super) fn chirp_angle(index: u32, n: u32) -> f64 {
    debug_assert!(n != 0);
    let index = u64::from(index);
    let n_wide = u64::from(n);
    let phase_index = (index * index) % (2 * n_wide);
    let whole_pi = u32::try_from(phase_index / n_wide)
        .expect("invariant: a phase reduced modulo 2N contains at most one whole pi");
    let remainder = u32::try_from(phase_index % n_wide)
        .expect("invariant: the phase remainder is less than the u32 transform length");
    f64::from(whole_pi) * core::f64::consts::PI
        + core::f64::consts::PI * f64::from(remainder) / f64::from(n)
}
