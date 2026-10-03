//! Contract clauses for the device-neutral
//! [`hephaestus_core::EmbeddingOps`] seam.

use hephaestus_core::{ComputeDevice, EmbeddingOps, HephaestusError};

fn table() -> Vec<f32> {
    vec![
        0.0, 1.0, 2.0, // row 0
        10.0, 11.0, 12.0, // row 1
        20.0, 21.0, 22.0, // row 2
        30.0, 31.0, 32.0, // row 3
    ]
}

/// Gathering rows `[2, 0, 3]` from a 4x3 table must reproduce those rows
/// contiguously, in order.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees.
pub fn assert_embedding_gather_contract<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: EmbeddingOps<D, f32>,
{
    let name = device.backend_name();
    let table_buf = device.upload(&table()).expect("table upload");
    let indices = device.upload(&[2u32, 0, 3]).expect("indices upload");
    let output = device.alloc_zeroed::<f32>(9).expect("output alloc");

    ops.gather_into(device, &table_buf, 4, 3, &indices, &output)
        .expect("gather dispatch");

    let mut got = vec![0.0f32; 9];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got,
        vec![20.0, 21.0, 22.0, 0.0, 1.0, 2.0, 30.0, 31.0, 32.0],
        "{name}: gathered rows must match the selected table rows in order"
    );
}

/// An index at or beyond `num_embeddings` must be a typed rejection, never a
/// clamped or wrapped read.
///
/// # Panics
///
/// Panics when an out-of-range index is not rejected.
pub fn assert_embedding_gather_rejects_out_of_range_index<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: EmbeddingOps<D, f32>,
{
    let name = device.backend_name();
    let table_buf = device.upload(&table()).expect("table upload");
    let indices = device.upload(&[0u32, 4]).expect("indices upload");
    let output = device.alloc_zeroed::<f32>(6).expect("output alloc");

    let err = ops
        .gather_into(device, &table_buf, 4, 3, &indices, &output)
        .expect_err("index 4 is out of range for 4 embeddings");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: an out-of-range embedding index must be a typed InvalidConfiguration \
         rejection, not silently gathered; got {err:?}"
    );
}
