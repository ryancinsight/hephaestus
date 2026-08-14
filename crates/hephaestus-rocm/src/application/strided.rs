//! Layout-aware operands shared by ROCm strided operator families.

use crate::RocmBuffer;

/// A typed ROCm buffer paired with a rank-specific logical layout.
///
/// This device's instantiation of the device-neutral
/// [`StridedView`](hephaestus_core::StridedView), so an operand built here
/// passes straight into the generic accelerator layer without conversion.
pub type StridedOperand<'a, T, const N: usize> = hephaestus_core::StridedView<'a, RocmBuffer<T>, N>;
