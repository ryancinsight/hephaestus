//! Host reference implementor of [`hephaestus_core::CrossProductOps`].
//!
//! A direct per-triple loop: the reference substrate favors an obviously
//! correct implementation over a fast one (crate docs), and the standard
//! `(x, y, z)` cross-product formula has no numerically interesting
//! structure to delegate to leto for.

use eunomia::Pod;
use hephaestus_core::{CrossProductOps, DeviceBuffer, Result, validate_cross_product_lengths};

use crate::operands::require_disjoint_output;
use crate::{HostBuffer, HostDevice};

/// Host-backed batched cross product for the reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostCrossProductOps;

impl<T> CrossProductOps<HostDevice, T> for HostCrossProductOps
where
    T: Pod + Copy + core::ops::Mul<Output = T> + core::ops::Sub<Output = T>,
{
    fn cross_into(
        &self,
        _device: &HostDevice,
        a: &HostBuffer<T>,
        b: &HostBuffer<T>,
        out: &HostBuffer<T>,
    ) -> Result<()> {
        require_disjoint_output(a, b, out)?;
        let a_cells = a.read();
        let b_cells = b.read();
        let triples = validate_cross_product_lengths(a_cells.len(), b_cells.len(), out.len())?;

        let mut out_cells = out.write();
        for i in 0..triples {
            let base = i * 3;
            let (ax, ay, az) = (a_cells[base], a_cells[base + 1], a_cells[base + 2]);
            let (bx, by, bz) = (b_cells[base], b_cells[base + 1], b_cells[base + 2]);
            out_cells[base] = ay * bz - az * by;
            out_cells[base + 1] = az * bx - ax * bz;
            out_cells[base + 2] = ax * by - ay * bx;
        }
        Ok(())
    }
}
