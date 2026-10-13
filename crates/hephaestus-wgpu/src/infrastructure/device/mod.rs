//! Canonical WGPU device acquisition, ownership, transfers, and bounded waits.

mod accessors;
mod acquisition;
mod allocation;
mod cache;
mod construction;
mod constructors;
mod features;
mod mapping;
mod pools;
mod provider;
mod retention;
mod state;
mod transfer;
mod wait;

#[cfg(test)]
mod pool_tests;
#[cfg(test)]
mod readback_tests;
#[cfg(test)]
mod tests;

pub(crate) use cache::{FusionPipelineKey, PipelineCache, PipelineKey};
pub(super) use retention::StagingPoolAccounting;
pub use state::WgpuDevice;

use hephaestus_core::{HephaestusError, Result};
use mapping::{MappingLifecycle, ReadbackRegion};
use retention::{
    STAGING_POOL_MAX_BUFFERS, STAGING_POOL_MAX_BYTES, UNIFORM_POOL_MAX_BUFFERS,
    UNIFORM_POOL_MAX_BYTES,
};
pub(crate) use wait::{device_wait_deadline, poll_failure};

#[cfg(test)]
use hephaestus_core::{ComputeDevice, DeviceFeature, DevicePreference};
#[cfg(test)]
use std::sync::{Arc, atomic::Ordering};
#[cfg(test)]
use std::time::Duration;
#[cfg(test)]
use wait::{DEFAULT_DEVICE_WAIT, TEST_WAIT_DEADLINE_NS};
