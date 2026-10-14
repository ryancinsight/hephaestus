//! Direct-loop n-D nonzero-coordinate extraction for the host reference device.
//!
//! There is no `leto::nonzero` to delegate to, so this implementation
//! scans the input's logical indices in row-major order through
//! [`leto::Layout::offset_of`] directly, writing one `u32` coordinate
//! row per nonzero cell plus the nonzero count. The shared conformance
//! clauses pin these value semantics with hand-computed fixtures every
//! accelerator's differential test must match.

use eunomia::{NumericElement, Pod};
use hephaestus_core::{NonzeroOps, Result, StridedView};

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Direct-loop nonzero-coordinate extraction for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostNonzeroOps;

impl<T> NonzeroOps<HostDevice, T> for HostNonzeroOps
where
    T: Pod + NumericElement,
{
    type Dialect = hephaestus_core::Host;

    fn nonzero_into<const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, N>,
        coords: StridedView<'_, HostBuffer<u32>, 2>,
        count: StridedView<'_, HostBuffer<u32>, 1>,
    ) -> Result<()> {
        hephaestus_core::validate_nonzero_shape(
            input.layout.shape(),
            coords.layout.shape(),
            count.layout.shape(),
        )?;
        if coords.buffer.aliases(input.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "nonzero coords buffer must not alias the input buffer".to_string(),
            });
        }
        if count.buffer.aliases(input.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "nonzero count buffer must not alias the input buffer".to_string(),
            });
        }
        if coords.buffer.aliases(count.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "nonzero coords buffer must not alias the count buffer".to_string(),
            });
        }

        let input_cells = input.buffer.read();
        let mut coords_cells = coords.buffer.write();
        let mut count_cells = count.buffer.write();

        // Row-major logical scan: flat 0..numel decodes to coordinates
        // with the last axis fastest, so rows emerge in the pinned
        // order. A zero extent makes numel zero and the loop vacuous —
        // no modulo by zero.
        let shape = input.layout.shape();
        let numel: usize = shape.iter().product();
        let zero = T::ZERO;
        let mut n = 0usize;
        for flat in 0..numel {
            let mut index = [0usize; N];
            let mut rest = flat;
            for d in (0..N).rev() {
                index[d] = rest % shape[d];
                rest /= shape[d];
            }
            let offset = input.layout.offset_of(index).map_err(map_leto_error)?;
            if input_cells[offset] != zero {
                for (j, &c) in index.iter().enumerate() {
                    let coord_offset = coords.layout.offset_of([n, j]).map_err(map_leto_error)?;
                    coords_cells[coord_offset] =
                        u32::try_from(c).expect("coordinate fits u32 within a validated shape");
                }
                n += 1;
            }
        }
        let count_offset = count.layout.offset_of([0]).map_err(map_leto_error)?;
        count_cells[count_offset] =
            u32::try_from(n).expect("nonzero count fits u32 within a validated shape");
        Ok(())
    }
}
