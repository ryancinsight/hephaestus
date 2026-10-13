//! Expression and identity tests for the operation markers.

use super::*;
use crate::domain::dialect::{CudaC, HipC, Host, Wgsl};
use eunomia::RealField;

mod double_double;

use double_double::DoubleDouble;

#[path = "tests/binary_dispatch/mod.rs"]
mod binary_dispatch;
#[path = "tests/fixtures.rs"]
mod fixtures;
#[path = "tests/gelu.rs"]
mod gelu;
#[path = "tests/hardswish.rs"]
mod hardswish;
#[path = "tests/mish.rs"]
mod mish;
#[path = "tests/silu.rs"]
mod silu;
#[path = "tests/special_values.rs"]
mod special_values;
#[path = "tests/tanh.rs"]
mod tanh;
pub(crate) use fixtures::*;
