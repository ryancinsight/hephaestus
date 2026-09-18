//! Leto as a dense-vector-seam implementor (ADR 0046).
//!
//! [`HostDenseVectorOps`] adapts leto-ops' slice kernels (`Scalar` elementwise
//! and dot hooks, `RealScalar` norms) onto
//! [`DenseVectorOps<HostDevice, T>`](hephaestus_core::DenseVectorOps), so the
//! conformance suite's dense-vector clause runs on the host pair.
//!
//! Lengths are validated before any element is written. In-place updates
//! (`axpy`, `xpay`) accept `source` naming the same allocation as `target` and
//! compute the aliased recurrence under one write guard; the `*_into` methods
//! reject an output aliasing an input, as the seam contract requires distinct
//! storage.

use eunomia::Pod;
use hephaestus_core::{DenseVectorOps, HephaestusError, Result};
use leto::{ArrayView, Layout};
use leto_ops::RealScalar;

use crate::operands::{require_disjoint_output, with_operands};
use crate::{HostBuffer, HostDevice, map_leto_error};

/// Dense rank-one vector operations for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostDenseVectorOps;

/// A dot product bound to its operand allocations.
///
/// The host dispatches no kernel, so preparation binds nothing but operand
/// identity: the borrows are what [`DenseVectorOps::dot_prepared`] checks a
/// call's operands against.
pub struct HostPreparedDot<'a, T> {
    left: &'a HostBuffer<T>,
    right: &'a HostBuffer<T>,
}

/// A Euclidean norm bound to its operand allocation.
pub struct HostPreparedNorm<'a, T> {
    vector: &'a HostBuffer<T>,
}

/// Reject `operand` unless its length equals `reference`, the operand whose
/// length the operation is defined over (the target, or the left input).
fn require_same_len<T>(reference: &[T], operand: &[T]) -> Result<()> {
    HostDevice::require_matching_len(reference.len(), operand.len())
}

fn require_same_allocation<T>(
    role: &str,
    expected: &HostBuffer<T>,
    actual: &HostBuffer<T>,
) -> Result<()> {
    if expected.aliases(actual) {
        Ok(())
    } else {
        Err(HephaestusError::DispatchFailed {
            message: format!("prepared {role} received a different host allocation"),
        })
    }
}

/// Validate equal lengths, then write `output = kernel(left, right)`.
fn binary_into<T: Pod>(
    left: &HostBuffer<T>,
    right: &HostBuffer<T>,
    output: &HostBuffer<T>,
    kernel: fn(&[T], &[T], &mut [T]),
) -> Result<()> {
    require_disjoint_output(left, right, output)?;
    let mut out_cells = output.write();
    with_operands(left, right, |left_cells, right_cells| {
        require_same_len(left_cells, right_cells)?;
        require_same_len(left_cells, &out_cells)?;
        kernel(left_cells, right_cells, &mut out_cells);
        Ok(())
    })
}

/// Validate equal lengths, then apply `update(target_cell, source_cell)` in
/// place; a `source` naming `target` reads each cell before it is written.
fn update_in_place<T: Pod>(
    target: &HostBuffer<T>,
    source: &HostBuffer<T>,
    update: impl Fn(T, T) -> T,
) -> Result<()> {
    if source.aliases(target) {
        for cell in target.write().iter_mut() {
            *cell = update(*cell, *cell);
        }
        return Ok(());
    }
    let source_cells = source.read();
    let mut target_cells = target.write();
    require_same_len(&target_cells, &source_cells)?;
    for (cell, &value) in target_cells.iter_mut().zip(source_cells.iter()) {
        *cell = update(*cell, value);
    }
    Ok(())
}

/// Reduce one vector through a leto norm over a rank-one view of its cells.
fn norm_of<T: Pod + RealScalar, E: core::fmt::Display>(
    vector: &HostBuffer<T>,
    norm: fn(&ArrayView<'_, T, 1>) -> core::result::Result<T, E>,
) -> Result<T> {
    let cells = vector.read();
    let layout = Layout::c_contiguous([cells.len()]).map_err(map_leto_error)?;
    norm(&ArrayView::try_new(layout, &cells).map_err(map_leto_error)?).map_err(map_leto_error)
}

impl<T> DenseVectorOps<HostDevice, T> for HostDenseVectorOps
where
    T: Pod + RealScalar,
{
    type PreparedDot<'a>
        = HostPreparedDot<'a, T>
    where
        Self: 'a;
    type PreparedNorm<'a>
        = HostPreparedNorm<'a, T>
    where
        Self: 'a;

    fn copy_vector(
        &self,
        _device: &HostDevice,
        source: &HostBuffer<T>,
        target: &HostBuffer<T>,
    ) -> Result<()> {
        if source.aliases(target) {
            return Ok(());
        }
        let source_cells = source.read();
        let mut target_cells = target.write();
        require_same_len(&target_cells, &source_cells)?;
        target_cells.copy_from_slice(&source_cells);
        Ok(())
    }

    fn scale_vector(&self, _device: &HostDevice, target: &HostBuffer<T>, factor: T) -> Result<()> {
        for cell in target.write().iter_mut() {
            *cell *= factor;
        }
        Ok(())
    }

    fn axpy(
        &self,
        _device: &HostDevice,
        target: &HostBuffer<T>,
        source: &HostBuffer<T>,
        factor: T,
    ) -> Result<()> {
        update_in_place(target, source, |accumulator, increment| {
            accumulator + factor * increment
        })
    }

    fn xpay(
        &self,
        _device: &HostDevice,
        target: &HostBuffer<T>,
        source: &HostBuffer<T>,
        factor: T,
    ) -> Result<()> {
        update_in_place(target, source, |accumulator, increment| {
            increment + factor * accumulator
        })
    }

    fn subtract_into(
        &self,
        _device: &HostDevice,
        left: &HostBuffer<T>,
        right: &HostBuffer<T>,
        output: &HostBuffer<T>,
    ) -> Result<()> {
        binary_into(left, right, output, T::sub_slice)
    }

    fn add_into(
        &self,
        _device: &HostDevice,
        left: &HostBuffer<T>,
        right: &HostBuffer<T>,
        output: &HostBuffer<T>,
    ) -> Result<()> {
        binary_into(left, right, output, T::add_slice)
    }

    fn multiply_into(
        &self,
        _device: &HostDevice,
        left: &HostBuffer<T>,
        right: &HostBuffer<T>,
        output: &HostBuffer<T>,
    ) -> Result<()> {
        binary_into(left, right, output, T::mul_slice)
    }

    fn divide_into(
        &self,
        _device: &HostDevice,
        left: &HostBuffer<T>,
        right: &HostBuffer<T>,
        output: &HostBuffer<T>,
    ) -> Result<()> {
        binary_into(left, right, output, T::div_slice)
    }

    fn prepare_dot<'a>(
        &self,
        _device: &HostDevice,
        left: &'a HostBuffer<T>,
        right: &'a HostBuffer<T>,
    ) -> Result<Self::PreparedDot<'a>> {
        with_operands(left, right, require_same_len)?;
        Ok(HostPreparedDot { left, right })
    }

    fn dot_prepared<'a>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedDot<'a>,
        left: &HostBuffer<T>,
        right: &HostBuffer<T>,
    ) -> Result<T> {
        require_same_allocation("dot left operand", prepared.left, left)?;
        require_same_allocation("dot right operand", prepared.right, right)?;
        with_operands(left, right, |left_cells, right_cells| {
            require_same_len(left_cells, right_cells)?;
            Ok(T::dot_slice(left_cells, right_cells))
        })
    }

    fn prepare_norm_l2<'a>(
        &self,
        _device: &HostDevice,
        vector: &'a HostBuffer<T>,
    ) -> Result<Self::PreparedNorm<'a>> {
        Ok(HostPreparedNorm { vector })
    }

    fn norm_l2_prepared<'a>(
        &self,
        _device: &HostDevice,
        prepared: &Self::PreparedNorm<'a>,
        vector: &HostBuffer<T>,
    ) -> Result<T> {
        require_same_allocation("norm operand", prepared.vector, vector)?;
        norm_of(vector, leto_ops::norm_l2)
    }

    fn norm_l1(&self, _device: &HostDevice, vector: &HostBuffer<T>) -> Result<T> {
        norm_of(vector, leto_ops::norm_l1)
    }

    fn norm_max(&self, _device: &HostDevice, vector: &HostBuffer<T>) -> Result<T> {
        norm_of(vector, leto_ops::norm_max)
    }
}
