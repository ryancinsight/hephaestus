//! Byte-exact prefix copies over WGPU's aligned transfer commands.

use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{ComputeDevice, HephaestusError, Result};

use super::WgpuCommandStream;
use crate::application::pipeline::{encode_compute_pass, try_cached_pipeline};
use crate::application::prepared::checked_bind_group;
use crate::infrastructure::buffer::WgpuBuffer;

// The tail kernel merges a WGSL word, whose representation is four bytes.
const _: () = assert!(wgpu::COPY_BUFFER_ALIGNMENT == 4);

struct PrefixTail;

impl WgpuCommandStream<'_> {
    pub(super) fn copy_prefix_bytes<T: Pod>(
        &mut self,
        source: &WgpuBuffer<T>,
        destination: &WgpuBuffer<T>,
        bytes: u64,
    ) -> Result<()> {
        if bytes == 0 || source.aliases(destination) {
            return Ok(());
        }
        let tail = bytes % wgpu::COPY_BUFFER_ALIGNMENT;
        let aligned = bytes - tail;
        if tail == 0 {
            self.encoder
                .copy_buffer_to_buffer(source.raw(), 0, destination.raw(), 0, aligned);
            return Ok(());
        }

        // A prefix may end inside a word containing later destination values.
        // Stage only that word on-device, merge its low bytes, then restore it.
        // Both allocations include the final aligned word. Four-byte scratch
        // bindings also avoid imposing storage-binding limits on large copies.
        let source_word = self.device.alloc_uninitialized::<u32>(1)?;
        let destination_word = self.device.alloc_uninitialized::<u32>(1)?;
        let tail = u32::try_from(tail).map_err(|_| HephaestusError::TransferFailed {
            message: "copy prefix remainder exceeds u32".to_owned(),
        })?;
        // COPY_BUFFER_ALIGNMENT is four; 1 <= tail <= 3 gives 8..=24 bits.
        // WGSL storage words use little-endian byte order, so a low-bit mask
        // selects exactly the prefix bytes, independent of the scalar type.
        let mask = (1_u32 << (tail * 8)) - 1;
        let label = "hephaestus-prefix-tail";
        let pipeline = try_cached_pipeline(
            self.device,
            (TypeId::of::<PrefixTail>(), TypeId::of::<u32>(), tail),
            label,
            || {
                format!(
                    "@group(0) @binding(0) var<storage, read> source: array<u32>;\n\
                 @group(0) @binding(1) var<storage, read_write> destination: array<u32>;\n\
                 @compute @workgroup_size(1) fn main() {{\n\
                 let mask = {mask}u;\n\
                 destination[0] = (source[0] & mask) | (destination[0] & ~mask);\n}}"
                )
            },
        )?;
        let bindings = checked_bind_group(
            self.device,
            &pipeline,
            label,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source_word.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: destination_word.raw().as_entire_binding(),
                },
            ],
        )?;
        // Preparation can fail without leaving a partial prefix in this
        // stream. Once encoding begins, all fallible preparation is complete.
        if aligned != 0 {
            self.encoder
                .copy_buffer_to_buffer(source.raw(), 0, destination.raw(), 0, aligned);
        }
        self.encoder.copy_buffer_to_buffer(
            source.raw(),
            aligned,
            source_word.raw(),
            0,
            wgpu::COPY_BUFFER_ALIGNMENT,
        );
        self.encoder.copy_buffer_to_buffer(
            destination.raw(),
            aligned,
            destination_word.raw(),
            0,
            wgpu::COPY_BUFFER_ALIGNMENT,
        );
        encode_compute_pass(&mut self.encoder, &pipeline, &bindings, 1, label);
        self.encoder.copy_buffer_to_buffer(
            destination_word.raw(),
            0,
            destination.raw(),
            aligned,
            wgpu::COPY_BUFFER_ALIGNMENT,
        );
        Ok(())
    }
}
