//! Dense-FFT axis strategies and prepared Bluestein data.

use crate::infrastructure::buffer::WgpuBuffer;

use super::{kernel::ChirpParams, stages::RadixStages};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AxisStrategy {
    FusedRadix2,
    StagedRadix2,
    ChirpZ { n: usize, m: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub(crate) const fn len(self, dimensions: [usize; 3]) -> usize {
        match self {
            Self::X => dimensions[0],
            Self::Y => dimensions[1],
            Self::Z => dimensions[2],
        }
    }

    pub(crate) fn batch_count(self, dimensions: [usize; 3]) -> Option<usize> {
        match self {
            Self::X => dimensions[1].checked_mul(dimensions[2]),
            Self::Y => dimensions[0].checked_mul(dimensions[2]),
            Self::Z => dimensions[0].checked_mul(dimensions[1]),
        }
    }

    pub(crate) const fn shader_index(self) -> u32 {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

pub(crate) struct ChirpData {
    pub(crate) real_kernel: WgpuBuffer<f32>,
    pub(crate) imaginary_kernel: WgpuBuffer<f32>,
    pub(crate) direct_real: WgpuBuffer<f32>,
    pub(crate) direct_imaginary: WgpuBuffer<f32>,
    pub(crate) params: ChirpParams,
    pub(crate) forward_radix: RadixStages,
    pub(crate) inverse_radix: RadixStages,
}
