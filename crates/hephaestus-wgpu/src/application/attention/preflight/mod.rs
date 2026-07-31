//! Read-only semantic preflight preparation for WGPU attention dispatch.

mod backward;
mod forward;
mod kernel;

pub(super) use backward::prepare_backward;
pub(super) use forward::prepare_forward;
