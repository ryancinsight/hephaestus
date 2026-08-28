//! Provider-neutral radix-two execution stages.

use super::kernel::FftParams;

/// Value parameters for one typed radix-two execution plan.
///
/// Hephaestus owns the recurrence values, parameter transfer, and ordered
/// dispatch for every accelerator provider.
pub(crate) struct RadixStages {
    pub(crate) bit_reverse: FftParams,
    pub(crate) butterflies: Box<[FftParams]>,
    pub(crate) inverse_scale: Option<FftParams>,
    pub(crate) fft_len: u32,
    pub(crate) batch_count: u32,
    pub(crate) radix_four: bool,
}

impl RadixStages {
    fn params(
        fft_len: u32,
        stage: u32,
        inverse: u32,
        batch_count: u32,
        root_half: u32,
        scale_index: u32,
    ) -> FftParams {
        FftParams {
            n: fft_len,
            stage,
            inverse,
            batch_count,
            root_half,
            scale_index,
            padding: [0; 2],
        }
    }

    pub(crate) fn empty() -> Self {
        Self {
            bit_reverse: Self::params(0, 0, 0, 0, 0, 0),
            butterflies: Box::default(),
            inverse_scale: None,
            fft_len: 0,
            batch_count: 0,
            radix_four: false,
        }
    }

    pub(crate) fn radix_two(
        fft_len: u32,
        batch_count: u32,
        inverse: bool,
        root_half: u32,
        scale_index: u32,
    ) -> Self {
        let inverse_flag = u32::from(inverse);
        let butterflies = (0..fft_len.trailing_zeros())
            .map(|stage| {
                Self::params(
                    fft_len,
                    stage,
                    inverse_flag,
                    batch_count,
                    root_half,
                    scale_index,
                )
            })
            .collect();
        Self {
            bit_reverse: Self::params(
                fft_len,
                0,
                inverse_flag,
                batch_count,
                root_half,
                scale_index,
            ),
            butterflies,
            inverse_scale: inverse.then_some(Self::params(
                fft_len,
                0,
                1,
                batch_count,
                root_half,
                scale_index,
            )),
            fft_len,
            batch_count,
            radix_four: false,
        }
    }

    pub(crate) fn radix_four(
        fft_len: u32,
        batch_count: u32,
        inverse: bool,
        root_half: u32,
        scale_index: u32,
    ) -> Self {
        let inverse_flag = u32::from(inverse);
        let butterflies = (0..(fft_len.trailing_zeros() / 2))
            .map(|stage| {
                Self::params(
                    fft_len,
                    stage,
                    inverse_flag,
                    batch_count,
                    root_half,
                    scale_index,
                )
            })
            .collect();
        Self {
            bit_reverse: Self::params(
                fft_len,
                0,
                inverse_flag,
                batch_count,
                root_half,
                scale_index,
            ),
            butterflies,
            inverse_scale: inverse.then_some(Self::params(
                fft_len,
                0,
                1,
                batch_count,
                root_half,
                scale_index,
            )),
            fft_len,
            batch_count,
            radix_four: true,
        }
    }
}
