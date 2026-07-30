mod backward;
mod common;
mod forward;

pub(super) use backward::{GradientTarget, backward_shader};
pub(super) use forward::forward_shader;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ConvolutionDirection {
    Regular,
    Transposed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BiasMode {
    Absent,
    Present,
}
