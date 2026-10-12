//! CUDA implementation of [`hephaestus_core::TriangularOps`].
//!
//! One thread per `(row, col)` pair (flat-dispatched over `rows * cols`)
//! evaluates the same diagonal-side test the WGSL seam uses, and copies or
//! zeroes directly — no shared state between threads. `mode` bakes into the
//! generated source at pipeline-cache-key granularity (there is no `axis` to
//! bake in — this seam has none); `diagonal` is an ordinary runtime kernel
//! argument, clamped host-side to `i32` exactly as the WGSL seam clamps it
//! before uniform upload.

use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, CudaC, DeviceBuffer, DialectScalar, HephaestusError, Result, StridedView,
    TriangularMode, TriangularOps, map_layout_err, validate_triangular_shape,
};

use crate::application::linalg::{to_i32, to_u32};
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

fn triangular_source<T: DialectScalar<CudaC>>(mode: TriangularMode) -> String {
    let keep = match mode {
        // Differences are evaluated only after ordering makes them
        // non-negative, so signed coordinate arithmetic cannot overflow.
        // Mirrors the WGSL seam's predicates exactly.
        TriangularMode::Lower => {
            "(diagonal >= 0 && (col <= row || col - row <= diagonal)) || (diagonal < 0 && diagonal != (-2147483647 - 1) && col < row && row - col >= -diagonal)"
        }
        TriangularMode::Upper => {
            "(diagonal >= 0 && col >= row && col - row >= diagonal) || (diagonal < 0 && (diagonal == (-2147483647 - 1) || col >= row || row - col <= -diagonal))"
        }
    };
    format!(
        r#"{prelude}extern "C" __global__ void triangular_kernel(
    const {ty}* input,
    {ty}* out,
    int in_stride_row,
    int in_stride_col,
    int out_stride_row,
    int out_stride_col,
    unsigned int in_offset,
    unsigned int out_offset,
    unsigned int rows,
    unsigned int cols,
    int diagonal
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = rows * cols;
    if (tid >= total) {{
        return;
    }}
    int row = (int)(tid / cols);
    int col = (int)(tid % cols);
    int out_off = (int)out_offset + row * out_stride_row + col * out_stride_col;
    if ({keep}) {{
        int in_off = (int)in_offset + row * in_stride_row + col * in_stride_col;
        out[out_off] = input[in_off];
    }} else {{
        out[out_off] = ({ty})0;
    }}
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
        keep = keep,
    )
}

/// CUDA device marker implementing [`TriangularOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaTriangularOps;

impl<T> TriangularOps<CudaDevice, T> for CudaTriangularOps
where
    T: DialectScalar<CudaC> + Pod,
{
    fn triangular_into(
        &self,
        device: &CudaDevice,
        input: StridedView<'_, CudaBuffer<T>, 2>,
        mode: TriangularMode,
        diagonal: i64,
        output: StridedView<'_, CudaBuffer<T>, 2>,
    ) -> Result<()> {
        let in_layout = input.layout;
        let out_layout = output.layout;
        validate_triangular_shape(in_layout.shape(), out_layout.shape())?;
        if output.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "output buffer must not alias input buffer".to_string(),
            });
        }

        in_layout
            .validate_storage_len(input.buffer.len())
            .map_err(map_layout_err)?;
        out_layout
            .validate_storage_len(output.buffer.len())
            .map_err(map_layout_err)?;
        if !out_layout.is_injective().map_err(map_layout_err)? {
            return Err(HephaestusError::DispatchFailed {
                message: "triangular output layout must be non-overlapping".to_string(),
            });
        }
        let [rows, cols] = in_layout.shape();
        if rows > i32::MAX as usize || cols > i32::MAX as usize {
            return Err(HephaestusError::InvalidConfiguration {
                message: "triangular shape exceeds signed shader coordinate range".to_string(),
            });
        }
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        // Coordinates are in [0, i32::MAX - 1], so every `col - row`
        // comparison lies in the i32 range. Clamping the threshold preserves
        // both comparisons for values beyond that interval.
        let diagonal_i32 = i32::try_from(diagonal.clamp(i64::from(i32::MIN), i64::from(i32::MAX)))
            .expect("invariant: clamped triangular diagonal fits i32");

        let in_strides = in_layout.strides();
        let out_strides = out_layout.strides();
        let total =
            rows.checked_mul(cols)
                .ok_or_else(|| HephaestusError::InvalidConfiguration {
                    message: "triangular dispatch size overflows".to_string(),
                })?;

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::Triangular {
            scalar: TypeId::of::<T>(),
            width: width.get(),
            upper: matches!(mode, TriangularMode::Upper),
        };
        let kernel = cached_kernel(device, key, "triangular_kernel", || {
            triangular_source::<T>(mode)
        })?;

        let mut input_ptr = input.buffer.raw();
        let mut out_ptr = output.buffer.raw();
        let mut in_stride_row = to_i32(in_strides[0], "input stride")?;
        let mut in_stride_col = to_i32(in_strides[1], "input stride")?;
        let mut out_stride_row = to_i32(out_strides[0], "output stride")?;
        let mut out_stride_col = to_i32(out_strides[1], "output stride")?;
        let mut in_offset = to_u32(in_layout.offset(), "input offset")?;
        let mut out_offset = to_u32(out_layout.offset(), "output offset")?;
        let mut rows_val = to_u32(rows, "row count")?;
        let mut cols_val = to_u32(cols, "column count")?;
        let mut diagonal_val = diagonal_i32;

        // Argument list mirrors `triangular_kernel(const T*, T*, int x4,
        // unsigned int x4, int)`.
        let mut args: [*mut core::ffi::c_void; 11] = [
            (&mut input_ptr as *mut u64).cast(),
            (&mut out_ptr as *mut u64).cast(),
            (&mut in_stride_row as *mut i32).cast(),
            (&mut in_stride_col as *mut i32).cast(),
            (&mut out_stride_row as *mut i32).cast(),
            (&mut out_stride_col as *mut i32).cast(),
            (&mut in_offset as *mut u32).cast(),
            (&mut out_offset as *mut u32).cast(),
            (&mut rows_val as *mut u32).cast(),
            (&mut cols_val as *mut u32).cast(),
            (&mut diagonal_val as *mut i32).cast(),
        ];
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::linear(grid_size(total, width)?, width),
            &mut args,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_predicates_avoid_signed_addition_overflow() {
        let lower = triangular_source::<f32>(TriangularMode::Lower);
        let upper = triangular_source::<f32>(TriangularMode::Upper);
        assert!(!lower.contains("row + diagonal"));
        assert!(!upper.contains("row + diagonal"));
        assert!(lower.contains("col - row <= diagonal"));
        assert!(upper.contains("row - col <= -diagonal"));
    }
}
