//! Direct-loop inner product for the host reference device.
//!
//! There is no strided `leto::dot` to delegate to, so this
//! implementation folds the elementwise products in ascending flat
//! order through [`leto::Layout::offset_of`] directly, starting from
//! zero in native `T` precision. The shared conformance clauses pin
//! these value semantics with hand-computed fixtures every
//! accelerator's differential test must match.

use eunomia::{NumericElement, Pod};
use hephaestus_core::{DotOps, Result, StridedView};

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Direct-loop inner product for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostDotOps;

impl<T> DotOps<HostDevice, T> for HostDotOps
where
    T: Pod + NumericElement,
{
    fn dot_into<const N: usize, const M: usize>(
        &self,
        _device: &HostDevice,
        a: StridedView<'_, HostBuffer<T>, N>,
        b: StridedView<'_, HostBuffer<T>, M>,
        out: StridedView<'_, HostBuffer<T>, 1>,
    ) -> Result<()> {
        hephaestus_core::validate_dot_shape(
            a.layout.shape(),
            b.layout.shape(),
            out.layout.shape(),
        )?;
        if out.buffer.aliases(a.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "dot output buffer must not alias the first input buffer".to_string(),
            });
        }
        if out.buffer.aliases(b.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "dot output buffer must not alias the second input buffer".to_string(),
            });
        }

        let a_cells = a.buffer.read();
        let b_cells = b.buffer.read();
        let mut out_cells = out.buffer.write();

        // Single pass in ascending flat order: decode each side's flat
        // index to logical coordinates with the last axis fastest, form
        // the product, and accumulate into zero. A zero extent makes
        // numel zero and the loop vacuous — no modulo by zero.
        let a_shape = a.layout.shape();
        let b_shape = b.layout.shape();
        let numel: usize = a_shape.iter().product();
        let mut acc = T::ZERO;
        for flat in 0..numel {
            let mut a_index = [0usize; N];
            let mut rest = flat;
            for d in (0..N).rev() {
                a_index[d] = rest % a_shape[d];
                rest /= a_shape[d];
            }
            let mut b_index = [0usize; M];
            rest = flat;
            for d in (0..M).rev() {
                b_index[d] = rest % b_shape[d];
                rest /= b_shape[d];
            }
            let a_offset = a.layout.offset_of(a_index).map_err(map_leto_error)?;
            let b_offset = b.layout.offset_of(b_index).map_err(map_leto_error)?;
            acc += a_cells[a_offset] * b_cells[b_offset];
        }
        let out_offset = out.layout.offset_of([0]).map_err(map_leto_error)?;
        out_cells[out_offset] = acc;
        Ok(())
    }
}
