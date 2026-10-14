//! Direct-loop n-D masked fill for the host reference device.
//!
//! There is no `leto::masked_fill` to delegate to, so this implementation
//! walks the strided views through [`leto::ArrayView`] directly: for every
//! logical index it tests the mask cell against zero and writes the fill
//! value or copies the input cell. The shared conformance clauses pin
//! these value semantics with hand-computed fixtures every accelerator's
//! differential test must match.

use eunomia::{NumericElement, Pod};
use hephaestus_core::{MaskedFillOps, Result, StridedView};
use leto::ArrayView;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Direct-loop masked fill for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostMaskedFillOps;

impl<T> MaskedFillOps<HostDevice, T> for HostMaskedFillOps
where
    T: Pod + NumericElement,
{
    type Dialect = hephaestus_core::Host;

    fn masked_fill_into<const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, N>,
        mask: StridedView<'_, HostBuffer<T>, N>,
        value: T,
        output: StridedView<'_, HostBuffer<T>, N>,
    ) -> Result<()> {
        hephaestus_core::validate_masked_fill_shape(
            input.layout.shape(),
            mask.layout.shape(),
            output.layout.shape(),
        )?;
        if output.buffer.aliases(input.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "masked_fill output buffer must not alias the input buffer".to_string(),
            });
        }
        if output.buffer.aliases(mask.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "masked_fill output buffer must not alias the mask buffer".to_string(),
            });
        }

        let in_cells = input.buffer.read();
        let mask_cells = mask.buffer.read();
        let in_view =
            ArrayView::<T, N>::try_new(*input.layout, &in_cells).map_err(map_leto_error)?;
        let mask_view =
            ArrayView::<T, N>::try_new(*mask.layout, &mask_cells).map_err(map_leto_error)?;

        // The mask test is scalar equality against zero, never a bitwise
        // test: `-0.0 == 0.0` keeps the input while `NaN != 0.0` fills.
        let zero = T::ZERO;
        let mut out_cells = output.buffer.write();
        for (index, mask_cell) in mask_view.indexed_iter() {
            let offset = output.layout.offset_of(index).map_err(map_leto_error)?;
            out_cells[offset] = if *mask_cell == zero {
                in_view[index]
            } else {
                value
            };
        }
        Ok(())
    }
}
