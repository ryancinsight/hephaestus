//! ROCm providers for generic pooling and sliding-window operations.

use hephaestus_core::{CFamilyPoolingOps, CFamilySlidingWindowOps};

use crate::RocmDevice;

/// ROCm's generic pooling provider.
pub type RocmPoolingOps = CFamilyPoolingOps<RocmDevice>;

/// ROCm's generic sliding-window provider.
pub type RocmSlidingWindowOps = CFamilySlidingWindowOps<RocmDevice>;
