//! Rank-2 prefix/suffix scans, written once over the [`DeviceApi`](crate::DeviceApi) seam.
//!
//! Kernel source is selected by *dialect* rather than by device, because two
//! backends compiling the same language want the same source: CUDA C++ and
//! HIP C++ share [`c_family_axis_scan_source`](crate::c_family_axis_scan_source) verbatim. Host orchestration â€”
//! validation, cache keying, shared-memory sizing, launch, and the allocating
//! entry points â€” is selected by nothing at all; it is one implementation
//! monomorphized per backend.

mod dialect;
mod entry;
mod launch;
mod ops;

#[cfg(test)]
mod tests;

pub use dialect::{AXIS_SCAN_ENTRY, AxisScanDialect, c_family_axis_scan_source};
pub use entry::{
    cumprod, cumprod_into, cumsum, cumsum_into, scan_axis, scan_axis_into, suffix_prod,
    suffix_prod_into, suffix_sum, suffix_sum_into,
};
pub use launch::{AxisScanKey, AxisScanLaunch, launch_planned_axis_scan, plan_axis_scan_launch};
pub use ops::{AxisScanOps, PreparedAxisScan};
