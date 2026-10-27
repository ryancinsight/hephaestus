//! The Leto CPU oracle: seeded fixture operands and the host rules the
//! provider implementations are compared against.

use eunomia::FloatElement;
use leto::{ArrayView, ArrayViewMut, Layout};
use leto_ops::{
    AdaGrad as CpuAdaGrad, AdaGradParameters as CpuAdaGradParameters, Adam as CpuAdam,
    AdamParameters as CpuAdamParameters, AdamW as CpuAdamW, AdamWParameters as CpuAdamWParameters,
    RealScalar, RmsProp as CpuRmsProp, RmsPropParameters as CpuRmsPropParameters, Sgd as CpuSgd,
    SgdParameters as CpuSgdParameters, stateful_update as cpu_update,
};

/// The seeded fixture operands every rule steps over.
///
/// The values are chosen so the arithmetic stays well inside `f32`'s
/// exact-integer range while still exercising distinct per-element
/// magnitudes: the parameter entries differ by tens, the gradient
/// entries by tenths, and the state entries by hundredths.
#[derive(Clone)]
pub(super) struct HostOperands<T> {
    pub(super) parameter: [T; 6],
    pub(super) gradient: [T; 6],
    pub(super) state_zero: [T; 6],
    pub(super) state_one: [T; 6],
}

impl<T: FloatElement> HostOperands<T> {
    /// The seeded fixture, identical for every rule and every width.
    pub(super) fn seeded() -> Self {
        Self {
            parameter: [91.0, 1.0, 2.0, 92.0, 3.0, 4.0].map(T::from_f32),
            gradient: [81.0, 0.1, 0.2, 82.0, 0.3, 0.4].map(T::from_f32),
            state_zero: [71.0, 0.5, 0.6, 72.0, 0.7, 0.8].map(T::from_f32),
            state_one: [61.0, 0.25, 0.36, 62.0, 0.49, 0.64].map(T::from_f32),
        }
    }
}

/// A Leto CPU rule the provider implementations are compared against.
pub(super) trait CpuRule<T> {
    /// The rule's parameter block.
    type Parameters: Copy;
    /// How many state buffers the rule carries.
    const STATE_COUNT: usize;

    /// Apply one step of the rule to the fixture operands.
    fn step(operands: &mut HostOperands<T>, parameters: Self::Parameters);
}

impl<T: RealScalar> CpuRule<T> for CpuSgd {
    type Parameters = CpuSgdParameters<T>;
    const STATE_COUNT: usize = 1;

    fn step(operands: &mut HostOperands<T>, parameters: Self::Parameters) {
        let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
        cpu_update::<T, Self, 2>(
            ArrayViewMut::new(layout, &mut operands.parameter),
            ArrayView::new(layout, &operands.gradient),
            ArrayViewMut::new(layout, &mut operands.state_zero),
            parameters,
        )
        .expect("Leto SGD oracle");
    }
}

impl<T: RealScalar> CpuRule<T> for CpuAdam {
    type Parameters = CpuAdamParameters<T>;
    const STATE_COUNT: usize = 2;

    fn step(operands: &mut HostOperands<T>, parameters: Self::Parameters) {
        let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
        cpu_update::<T, Self, 2>(
            ArrayViewMut::new(layout, &mut operands.parameter),
            ArrayView::new(layout, &operands.gradient),
            (
                ArrayViewMut::new(layout, &mut operands.state_zero),
                ArrayViewMut::new(layout, &mut operands.state_one),
            ),
            parameters,
        )
        .expect("Leto Adam oracle");
    }
}

impl<T: RealScalar> CpuRule<T> for CpuAdamW {
    type Parameters = CpuAdamWParameters<T>;
    const STATE_COUNT: usize = 2;

    fn step(operands: &mut HostOperands<T>, parameters: Self::Parameters) {
        let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
        cpu_update::<T, Self, 2>(
            ArrayViewMut::new(layout, &mut operands.parameter),
            ArrayView::new(layout, &operands.gradient),
            (
                ArrayViewMut::new(layout, &mut operands.state_zero),
                ArrayViewMut::new(layout, &mut operands.state_one),
            ),
            parameters,
        )
        .expect("Leto AdamW oracle");
    }
}

impl<T: RealScalar> CpuRule<T> for CpuRmsProp {
    type Parameters = CpuRmsPropParameters<T>;
    const STATE_COUNT: usize = 1;

    fn step(operands: &mut HostOperands<T>, parameters: Self::Parameters) {
        let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
        cpu_update::<T, Self, 2>(
            ArrayViewMut::new(layout, &mut operands.parameter),
            ArrayView::new(layout, &operands.gradient),
            ArrayViewMut::new(layout, &mut operands.state_zero),
            parameters,
        )
        .expect("Leto RMSProp oracle");
    }
}

impl<T: RealScalar> CpuRule<T> for CpuAdaGrad {
    type Parameters = CpuAdaGradParameters<T>;
    const STATE_COUNT: usize = 1;

    fn step(operands: &mut HostOperands<T>, parameters: Self::Parameters) {
        let layout = Layout::try_new([2, 2], [3, 1], 1).expect("valid conformance fixture layout");
        cpu_update::<T, Self, 2>(
            ArrayViewMut::new(layout, &mut operands.parameter),
            ArrayView::new(layout, &operands.gradient),
            ArrayViewMut::new(layout, &mut operands.state_zero),
            parameters,
        )
        .expect("Leto AdaGrad oracle");
    }
}
