//! [`BlockedDecompositionBackend`] implementation for the wgpu backend.
//!
//! Each method is a direct wrapper of an existing backend routine — the
//! region gather/scatter in [`super::region`], the trailing GEMM kernel in
//! [`super::lu`], and the allocation/copy primitives of the
//! [`ComputeDevice`] trait. The region transfers stage through the compact
//! device scratch buffer the core loop allocates once per call.

use hephaestus_core::{
    BlockedDecompositionBackend, ComputeDevice, PanelRegion, Result, TrailingGemm,
};

use super::lu::{GemmTrailingUpdate, gemm_trailing_update};
use super::region::{
    MatrixRegion, download_matrix_region_compact_into, write_matrix_region_compact_reusable,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

fn matrix_region(region: PanelRegion) -> MatrixRegion {
    MatrixRegion {
        stride: region.stride,
        row_start: region.row0,
        col_start: region.col0,
        rows: region.rows,
        cols: region.cols,
    }
}

impl BlockedDecompositionBackend for WgpuDevice {
    type Buffer = WgpuBuffer<f32>;

    fn alloc(&self, len: usize) -> Result<Self::Buffer> {
        self.alloc_zeroed::<f32>(len)
    }

    fn clone_device(&self, src: &Self::Buffer, len: usize) -> Result<Self::Buffer> {
        let dst = self.alloc_uninitialized::<f32>(len)?;
        self.copy_buffer(src, &dst)?;
        Ok(dst)
    }

    fn download_region(
        &self,
        buf: &Self::Buffer,
        region: PanelRegion,
        scratch: &Self::Buffer,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        download_matrix_region_compact_into(self, buf, scratch, matrix_region(region), out)
    }

    fn write_region(
        &self,
        buf: &Self::Buffer,
        region: PanelRegion,
        scratch: &Self::Buffer,
        data: &[f32],
    ) -> Result<()> {
        write_matrix_region_compact_reusable(self, buf, scratch, data, matrix_region(region))
    }

    fn gemm_trailing(&self, buf: &Self::Buffer, spec: TrailingGemm) -> Result<()> {
        gemm_trailing_update(
            self,
            GemmTrailingUpdate {
                a_buf: buf,
                a_offset: spec.a_offset,
                a_stride: spec.a_stride,
                a_rows: spec.a_rows,
                a_cols: spec.a_cols,
                b_buf: buf,
                b_offset: spec.b_offset,
                b_stride: spec.b_stride,
                b_cols: spec.b_cols,
                c_buf: buf,
                c_offset: spec.c_offset,
                c_stride: spec.c_stride,
            },
        )
    }
}
