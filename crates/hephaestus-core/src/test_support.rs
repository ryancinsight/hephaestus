//! Assertions that name the constraint a call violated, and a device that
//! violates the contract on purpose.
//!
//! An assertion that only checks a result is an error passes whenever the call
//! fails, including for a reason unrelated to the one the test exists to
//! catch. That is sharper here than in most crates: a conformance clause runs
//! against every backend, so an error-only assertion also hides the case where
//! two backends reject the same input with different diagnostics.

use core::fmt::Display;
use std::sync::RwLock;

/// Assert that `result` is an error whose rendered form contains `fragment`.
///
/// Generic over the error type, and taking the result by reference so a caller
/// that inspects the value afterwards keeps its binding.
///
/// # Panics
///
/// Panics when `result` is `Ok`, or when its error does not contain
/// `fragment`.
#[track_caller]
pub fn assert_rejects<T, E: Display>(result: &Result<T, E>, fragment: &str) {
    match result {
        Err(error) => {
            let rendered = format!("{error}");
            assert!(
                rendered.contains(fragment),
                "rejection must name the violated constraint: expected a \
                 message containing {fragment:?}, got {rendered:?}"
            );
        }
        Ok(_) => panic!("expected a rejection containing {fragment:?}, got Ok"),
    }
}

/// Which contract a [`FaultyDevice`] breaks.
///
/// The conformance suite's clauses are only meaningful if they can fail. A
/// clause that passes against every device, including one that violates it, is
/// not testing the contract — it is testing that the call returned. Each
/// variant below is a failure mode some clause exists to catch, attached to the
/// smallest observable symptom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// Transfers are not faithful: the downloaded slice comes back reversed.
    ///
    /// Reversal rather than a bit flip because `Pod` exposes no byte view and
    /// this module adds no dependency to get one. It is a stronger fault
    /// anyway: every element differs unless the slice is a palindrome, so a
    /// round-trip clause that survived it would have to be comparing nothing.
    ///
    /// The clause's own operand is not a palindrome, which is what makes the
    /// fault observable — a reversed palindrome round-trips correctly and would
    /// let a corrupting device pass.
    TransferReverses,
    /// `download` accepts an `out` slice whose length differs from the
    /// buffer's, writing what fits and leaving the rest untouched.
    LengthMismatchAccepted,
    /// `copy_buffer` accepts operands of different element counts.
    CopyLengthMismatchAccepted,
}

// A fault that makes zeroed storage non-zero is deliberately absent. `Pod`
// offers `zeroed()` and no other constructor, so a generic non-zero `T` cannot
// be built here without adding a dependency or a numeric bound this module has
// no business imposing. A `Fault` variant whose implementation silently did
// nothing would be worse than no variant at all: its test would pass against a
// correct device and prove nothing, which is the exact failure this module
// exists to catch.

/// A [`ComputeDevice`](crate::ComputeDevice) that satisfies the trait and violates the contract.
///
/// Exists so the conformance clauses have a negative control: run a clause
/// against this device and it must panic, naming its own clause. Without that,
/// a clause that silently stopped asserting — an inverted comparison, an
/// `assert!` whose condition became constant — would keep passing in every
/// backend's test run and no one would see it.
///
/// The storage is ordinary host memory. The point is not to simulate a GPU; it
/// is to be a device whose *observable behaviour* is wrong in exactly one way,
/// so the clause under test is the thing that fails and nothing else.
#[derive(Clone, Copy, Debug)]
pub struct FaultyDevice {
    fault: Fault,
}

impl FaultyDevice {
    /// A device that breaks `fault` and behaves correctly otherwise.
    #[must_use]
    pub const fn new(fault: Fault) -> Self {
        Self { fault }
    }

    /// Which contract this device breaks.
    #[must_use]
    pub const fn fault(&self) -> Fault {
        self.fault
    }

    /// Build a length-mismatch rejection.
    fn length_mismatch(host_len: usize, device_len: usize) -> crate::HephaestusError {
        crate::HephaestusError::LengthMismatch {
            host_len,
            device_len,
        }
    }
}

impl crate::domain::device::ComputeDevice for FaultyDevice {
    type Buffer<T: eunomia::Pod> = FaultyBuffer<T>;

    fn backend_name(&self) -> &'static str {
        "faulty"
    }

    fn alloc_zeroed_with_hint<T: eunomia::Pod>(
        &self,
        len: usize,
        _hint: themis::PlacementHint,
    ) -> crate::Result<Self::Buffer<T>> {
        let mut data = Vec::new();
        data.try_reserve_exact(len)
            .map_err(|error| crate::HephaestusError::AllocationFailed {
                message: format!("faulty device allocation for {len} elements failed: {error}"),
            })?;
        data.resize(len, T::zeroed());
        Ok(FaultyBuffer::new(data))
    }

    fn alloc_uninitialized_with_hint<T: eunomia::Pod>(
        &self,
        len: usize,
        hint: themis::PlacementHint,
    ) -> crate::Result<Self::Buffer<T>> {
        self.alloc_zeroed_with_hint(len, hint)
    }

    fn upload_with_hint<T: eunomia::Pod>(
        &self,
        host: &[T],
        hint: themis::PlacementHint,
    ) -> crate::Result<Self::Buffer<T>> {
        let buffer = self.alloc_zeroed_with_hint(host.len(), hint)?;
        self.write_buffer(&buffer, host)?;
        Ok(buffer)
    }

    fn download<T: eunomia::Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> crate::Result<()> {
        let source = buffer.read();
        if out.len() != source.len() && self.fault != Fault::LengthMismatchAccepted {
            return Err(Self::length_mismatch(out.len(), source.len()));
        }
        let count = out.len().min(source.len());
        out[..count].copy_from_slice(&source[..count]);
        if self.fault == Fault::TransferReverses {
            out[..count].reverse();
        }
        Ok(())
    }

    fn write_buffer<T: eunomia::Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        host: &[T],
    ) -> crate::Result<()> {
        let mut target = buffer.write();
        if host.len() != target.len() {
            return Err(Self::length_mismatch(host.len(), target.len()));
        }
        target.copy_from_slice(host);
        Ok(())
    }

    fn write_sub_buffer<T: eunomia::Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        offset: usize,
        host: &[T],
    ) -> crate::Result<()> {
        let mut target = buffer.write();
        let end = offset.checked_add(host.len()).ok_or_else(|| {
            Self::length_mismatch(offset.saturating_add(host.len()), target.len())
        })?;
        if end > target.len() {
            return Err(Self::length_mismatch(end, target.len()));
        }
        target[offset..end].copy_from_slice(host);
        Ok(())
    }

    fn copy_buffer<T: eunomia::Pod>(
        &self,
        src: &Self::Buffer<T>,
        dst: &Self::Buffer<T>,
    ) -> crate::Result<()> {
        let source = src.read();
        let mut target = dst.write();
        if source.len() != target.len() && self.fault != Fault::CopyLengthMismatchAccepted {
            return Err(Self::length_mismatch(source.len(), target.len()));
        }
        let count = source.len().min(target.len());
        target[..count].copy_from_slice(&source[..count]);
        Ok(())
    }

    fn topology(&self) -> Option<&themis::GpuTopology> {
        None
    }

    fn synchronize(&self) -> crate::Result<()> {
        // Storage is host memory and every operation is complete on return, so
        // a correct device and this one are indistinguishable here. That is the
        // honest result rather than a gap: there is no pending work to lie
        // about, and a fault that invented one would test the fault, not the
        // clause.
        Ok(())
    }
}

/// Storage for [`FaultyDevice`]. Host memory, deliberately not a GPU handle.
///
/// Interior mutability is required by the seams' shared-reference mutation
/// contract (`write_buffer(&self, &Buffer, ..)`), the same reason
/// `hephaestus-host`'s buffer holds a lock. Poisoning is treated as
/// unrecoverable: this device exists inside tests, and a panicking test that
/// left the lock held has already failed.
#[derive(Debug)]
pub struct FaultyBuffer<T: eunomia::Pod> {
    data: RwLock<Vec<T>>,
}

impl<T: eunomia::Pod> FaultyBuffer<T> {
    fn new(data: Vec<T>) -> Self {
        Self {
            data: RwLock::new(data),
        }
    }

    /// Element count, without holding the lock.
    fn count(&self) -> usize {
        self.read().len()
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Vec<T>> {
        self.data
            .read()
            .expect("invariant: faulty buffer lock is never poisoned")
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, Vec<T>> {
        self.data
            .write()
            .expect("invariant: faulty buffer lock is never poisoned")
    }
}

impl<T: eunomia::Pod> crate::domain::buffer::DeviceBuffer<T> for FaultyBuffer<T> {
    fn len(&self) -> usize {
        self.count()
    }

    fn tier(&self) -> themis::MemoryTier {
        themis::MemoryTier::Dram
    }
}
