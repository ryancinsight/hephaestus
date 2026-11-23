//! Scalar-typed FFT coefficients prepared once on the host.

use hephaestus_core::{ComputeDevice, HephaestusError, Result};

use super::{kernel::FUSED_MAX_LENGTH, scalar::WgpuFftScalar};
use crate::{WgpuBuffer, WgpuDevice};

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

pub(super) fn try_host_vector<T>(capacity: usize, role: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|error| HephaestusError::AllocationFailed {
            message: format!("FFT {role} host allocation for {capacity} values failed: {error}"),
        })?;
    Ok(values)
}

pub(super) fn radix_parameters(root_length: usize, fft_len: u32) -> Result<(u32, u32)> {
    let fft_len_usize =
        usize::try_from(fft_len).map_err(|_| invalid("radix axis length does not fit usize"))?;
    if root_length < fft_len_usize || !root_length.is_power_of_two() {
        return Err(invalid(format!(
            "radix root length {root_length} does not cover FFT length {fft_len}"
        )));
    }
    let root_half = u32::try_from(root_length / 2)
        .map_err(|_| invalid("radix root half length exceeds the shader u32 domain"))?;
    let scale_offset = usize::try_from(fft_len.trailing_zeros())
        .expect("invariant: a u32 trailing-zero count fits usize");
    let scale_index = u32::try_from(
        root_length
            .checked_add(scale_offset)
            .ok_or_else(|| invalid("radix scale index overflows"))?,
    )
    .map_err(|_| invalid("radix scale index exceeds the shader u32 domain"))?;
    Ok((root_half, scale_index))
}

pub(super) fn twiddle_elements(root_length: usize) -> Result<usize> {
    debug_assert!(root_length.is_power_of_two());
    // There are root_length / 2 cosine values, as many sine values, and one
    // reciprocal for each radix level including the identity scale.
    let scale_count = root_length.trailing_zeros() as usize + 1;
    root_length
        .checked_add(scale_count)
        .ok_or_else(|| invalid("FFT twiddle element count overflows"))
}

pub(super) fn build_fused_twiddle<T: WgpuFftScalar>(device: &WgpuDevice) -> Result<WgpuBuffer<T>> {
    let mut roots = try_host_vector(twiddle_elements(FUSED_MAX_LENGTH)?, "fused twiddle")?;
    for index in 0..(FUSED_MAX_LENGTH / 2) {
        let angle = -core::f64::consts::TAU * index as f64 / FUSED_MAX_LENGTH as f64;
        roots.push(T::from_fft_coefficient(angle.cos()));
    }
    for index in 0..(FUSED_MAX_LENGTH / 2) {
        let angle = -core::f64::consts::TAU * index as f64 / FUSED_MAX_LENGTH as f64;
        roots.push(T::from_fft_coefficient(angle.sin()));
    }
    for exponent in 0..=FUSED_MAX_LENGTH.ilog2() {
        roots.push(T::from_fft_coefficient(1.0 / f64::from(1_u32 << exponent)));
    }
    device.upload(&roots)
}

pub(super) fn build_radix_twiddle<T: WgpuFftScalar>(
    device: &WgpuDevice,
    root_length: usize,
) -> Result<Option<WgpuBuffer<T>>> {
    if root_length == 0 {
        return Ok(None);
    }
    let mut roots = try_host_vector(twiddle_elements(root_length)?, "radix twiddle")?;
    for index in 0..(root_length / 2) {
        let angle = -core::f64::consts::TAU * index as f64 / root_length as f64;
        roots.push(T::from_fft_coefficient(angle.cos()));
    }
    for index in 0..(root_length / 2) {
        let angle = -core::f64::consts::TAU * index as f64 / root_length as f64;
        roots.push(T::from_fft_coefficient(angle.sin()));
    }
    for exponent in 0..=root_length.trailing_zeros() {
        roots.push(T::from_fft_coefficient(1.0 / f64::from(1_u32 << exponent)));
    }
    device.upload(&roots).map(Some)
}
