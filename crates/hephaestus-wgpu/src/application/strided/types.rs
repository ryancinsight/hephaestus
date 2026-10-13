use core::marker::PhantomData;

use hephaestus_core::StridedMeta;

use crate::infrastructure::buffer::WgpuBuffer;

/// A device buffer paired with the leto layout describing its logical view:
/// the unit every strided operand is passed as. The device-neutral
/// [`hephaestus_core::StridedOperand`] alias over a `WgpuBuffer<T>` — the same
/// two borrowed references — so a seam accepts the shared pair without a
/// per-backend conversion (the field move the alias erases).
pub type StridedOperand<'a, T, const N: usize> =
    hephaestus_core::StridedOperand<'a, WgpuBuffer<T>, N>;

/// Pipeline-cache discriminators so strided kernels never collide with the
/// contiguous kernels of the same `Op` in the `(TypeId, TypeId)` cache key.
pub(crate) struct StridedBinaryKernel<Op>(PhantomData<Op>);
pub(crate) struct StridedUnaryKernel<Op>(PhantomData<Op>);
pub(crate) struct StridedScalarKernel<Op>(PhantomData<Op>);

const _: () = assert!(core::mem::size_of::<StridedMeta>() == 144);
