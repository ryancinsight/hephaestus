//! Direct-loop n-D stable sort for the host reference device.
//!
//! There is no `leto::sort` to delegate to, so this implementation
//! gathers each axis lane through [`leto::ArrayView`] directly, sorts
//! the `(position, value)` pairs by the value with ties broken toward
//! the lower original position, and scatters the sorted values plus the
//! `u32` argsort positions. The shared conformance clauses pin these
//! value semantics with hand-computed fixtures every accelerator's
//! differential test must match.

use std::cmp::Ordering;

use eunomia::Pod;
use hephaestus_core::{Result, SortOps, StridedView};
use leto::ArrayView;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Direct-loop stable sort for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostSortOps;

/// Compare two lane values with `NaN` greatest in the ascending
/// direction, reversed wholesale for descending.
///
/// Rust's float `PartialOrd` returns `None` for any comparison touching
/// `NaN`, so the unordered case re-tests each side against itself: a
/// side that is unordered with itself is `NaN` and sorts after every
/// ordered value. The `(false, false)` arm is unreachable for scalars —
/// integers never compare unordered — and keeps the comparator total
/// for exotic `PartialOrd` implementors.
fn compare_values<T: PartialOrd>(a: &T, b: &T, descending: bool) -> Ordering {
    let ascending = match a.partial_cmp(b) {
        Some(order) => order,
        None => match (a.partial_cmp(a).is_none(), b.partial_cmp(b).is_none()) {
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            // Both NaN, or the unreachable no-NaN case for exotic
            // `PartialOrd` implementors (integers never compare
            // unordered): either way the values tie.
            (true, true) | (false, false) => Ordering::Equal,
        },
    };
    if descending {
        ascending.reverse()
    } else {
        ascending
    }
}

impl<T> SortOps<HostDevice, T> for HostSortOps
where
    T: Pod + PartialOrd + Copy,
{
    type Dialect = hephaestus_core::Host;

    fn sort_into<const N: usize>(
        &self,
        _device: &HostDevice,
        input: StridedView<'_, HostBuffer<T>, N>,
        dim: usize,
        descending: bool,
        values: StridedView<'_, HostBuffer<T>, N>,
        indices: StridedView<'_, HostBuffer<u32>, N>,
    ) -> Result<()> {
        hephaestus_core::validate_sort_shape(
            input.layout.shape(),
            dim,
            values.layout.shape(),
            indices.layout.shape(),
        )?;
        if values.buffer.aliases(input.buffer) {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: "sort values buffer must not alias the input buffer".to_string(),
            });
        }

        let input_cells = input.buffer.read();
        let input_view =
            ArrayView::<T, N>::try_new(*input.layout, &input_cells).map_err(map_leto_error)?;
        let mut values_cells = values.buffer.write();
        let mut indices_cells = indices.buffer.write();

        let shape = input.layout.shape();
        let extent = shape[dim];
        let outer_axes: Vec<usize> = (0..N).filter(|&d| d != dim).collect();
        let outer_count: usize = outer_axes.iter().map(|&d| shape[d]).product();

        // Each lane sorts independently: gather (position, value) pairs,
        // order by value with ties toward the lower original position —
        // the (value, position) total order, ascending ties in both
        // directions — then scatter values and u32 positions.
        for outer_linear in 0..outer_count {
            let mut index = [0usize; N];
            let mut rest = outer_linear;
            for &d in outer_axes.iter().rev() {
                index[d] = rest % shape[d];
                rest /= shape[d];
            }
            let mut lane: Vec<(usize, T)> = (0..extent)
                .map(|pos| {
                    index[dim] = pos;
                    (pos, input_view[index])
                })
                .collect();
            lane.sort_by(|(a_pos, a), (b_pos, b)| {
                compare_values(a, b, descending).then_with(|| a_pos.cmp(b_pos))
            });
            for (rank, (pos, value)) in lane.iter().enumerate() {
                index[dim] = rank;
                let values_offset = values.layout.offset_of(index).map_err(map_leto_error)?;
                values_cells[values_offset] = *value;
                let indices_offset = indices.layout.offset_of(index).map_err(map_leto_error)?;
                indices_cells[indices_offset] =
                    u32::try_from(*pos).expect("axis position fits u32 within a rank-N shape");
            }
        }
        Ok(())
    }
}
