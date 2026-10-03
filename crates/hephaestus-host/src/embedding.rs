//! Host reference implementor of [`hephaestus_core::EmbeddingOps`].
//!
//! A direct per-row copy loop with an immediate typed rejection on the first
//! out-of-range index — the host has direct memory access, so there is no
//! analog of the device backend's atomic-flag-then-check step.

use eunomia::Pod;
use hephaestus_core::{EmbeddingOps, HephaestusError, Result, validate_embedding_gather_shape};

use crate::operands::require_disjoint_output;
use crate::{HostBuffer, HostDevice};

/// Host-backed embedding-table gather for the reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostEmbeddingOps;

impl<T: Pod + Copy> EmbeddingOps<HostDevice, T> for HostEmbeddingOps {
    fn gather_into(
        &self,
        _device: &HostDevice,
        table: &HostBuffer<T>,
        num_embeddings: usize,
        embedding_dim: usize,
        indices: &HostBuffer<u32>,
        output: &HostBuffer<T>,
    ) -> Result<()> {
        require_disjoint_output(table, table, output)?;

        let table_cells = table.read();
        let index_cells = indices.read();
        let rows = validate_embedding_gather_shape(
            table_cells.len(),
            num_embeddings,
            embedding_dim,
            index_cells.len(),
            output.read().len(),
        )?;

        let mut out_cells = output.write();
        for row in 0..rows {
            let index = index_cells[row] as usize;
            if index >= num_embeddings {
                return Err(HephaestusError::InvalidConfiguration {
                    message: format!(
                        "embedding index {index} at position {row} is out of \
                         range for {num_embeddings} embeddings"
                    ),
                });
            }
            let table_base = index * embedding_dim;
            let out_base = row * embedding_dim;
            out_cells[out_base..out_base + embedding_dim]
                .copy_from_slice(&table_cells[table_base..table_base + embedding_dim]);
        }
        Ok(())
    }
}
