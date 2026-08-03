#![cfg(feature = "cuda")]

//! [`BlockedDecompositionBackend`] implementation for the CUDA backend.
//!
//! Each method is a direct wrapper of an existing backend routine — the
//! region gather/scatter in [`super::region`], the trailing GEMM kernel in
//! [`super::lu`]'s `gemm_impl`, and the allocation/copy primitives of the
//! [`ComputeDevice`] trait. Downloads keep their pinned-host staging
//! (CU-P6/CU-M3 row-wise async copies) and then refill the caller's reusable
//! host buffer; the compact device scratch the core loop allocates is unused
//! here (CUDA transfers stage through pinned host memory, not a device
//! staging buffer).

use hephaestus_core::{
    BlockedDecompositionBackend, ComputeDevice, PanelRegion, Result, TrailingGemm,
};

use super::lu::gemm_impl::gemm_trailing_update;
use super::region::{MatrixRegion, download_matrix_region_compact, write_matrix_region_compact};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

fn matrix_region(region: PanelRegion) -> MatrixRegion {
    MatrixRegion {
        stride: region.stride,
        row_start: region.row0,
        col_start: region.col0,
        rows: region.rows,
        cols: region.cols,
    }
}

impl BlockedDecompositionBackend for CudaDevice {
    type Buffer = CudaBuffer<f32>;

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
        _scratch: &Self::Buffer,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        let region = matrix_region(region);
        let staging = download_matrix_region_compact(self, buf, region)?;
        out.clear();
        out.extend_from_slice(&staging);
        Ok(())
    }

    fn write_region(
        &self,
        buf: &Self::Buffer,
        region: PanelRegion,
        _scratch: &Self::Buffer,
        data: &[f32],
    ) -> Result<()> {
        write_matrix_region_compact(self, buf, data, matrix_region(region))
    }

    fn gemm_trailing(&self, buf: &Self::Buffer, spec: TrailingGemm) -> Result<()> {
        gemm_trailing_update(
            self,
            buf,
            spec.a_offset,
            spec.a_stride,
            spec.a_rows,
            spec.a_cols,
            buf,
            spec.b_offset,
            spec.b_stride,
            spec.b_cols,
            buf,
            spec.c_offset,
            spec.c_stride,
        )
    }
}
