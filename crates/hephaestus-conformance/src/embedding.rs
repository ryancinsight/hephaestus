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

/// Scattering upstream rows `[1,2,3,4]` through indices `[2,0,2,1]` must
/// accumulate each row into its addressed table row, adding onto the entry
/// values (row 1 is pre-seeded) and leaving untouched rows alone.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees.
pub fn assert_embedding_scatter_add_contract<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: EmbeddingOps<D, f32>,
{
    let name = device.backend_name();
    let grad_table = device
        .upload(&[
            0.0f32, 0.0, 0.0, // row 0
            100.0, 200.0, 300.0, // row 1 (pre-seeded: accumulation adds on)
            0.0, 0.0, 0.0, // row 2
            0.0, 0.0, 0.0, // row 3
        ])
        .expect("grad-table upload");
    let indices = device.upload(&[2u32, 0, 2, 1]).expect("indices upload");
    let grad_output = device
        .upload(&[
            1.0f32, 2.0, 3.0, // position 0 -> row 2
            4.0, 5.0, 6.0, // position 1 -> row 0
            7.0, 8.0, 9.0, // position 2 -> row 2
            10.0, 11.0, 12.0, // position 3 -> row 1
        ])
        .expect("grad-output upload");

    ops.scatter_add_into(device, &grad_table, 4, 3, &indices, &grad_output)
        .expect("scatter-add dispatch");

    let mut got = vec![0.0f32; 12];
    device.download(&grad_table, &mut got).expect("download");
    assert_eq!(
        got,
        vec![
            4.0, 5.0, 6.0, // row 0
            110.0, 211.0, 312.0, // row 1 (seed + row)
            8.0, 10.0, 12.0, // row 2 (positions 0 + 2)
            0.0, 0.0, 0.0, // row 3 untouched
        ],
        "{name}: scatter-add must accumulate each upstream row into its addressed table row"
    );
}

/// Repeated indices must accumulate in ascending index-position order, pinned
/// by an order-sensitive fixture: `1e20 + 1 - 1e20 + 1` in f32 is exactly
/// `1.0` ascending (`1.0` vanishes below `ulp(1e20)` twice), while a tree or
/// reverse reduction yields `0.0`.
///
/// # Panics
///
/// Panics when the backend sums in any other order.
pub fn assert_embedding_scatter_add_accumulates_in_position_order<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: EmbeddingOps<D, f32>,
{
    let name = device.backend_name();
    let grad_table = device.alloc_zeroed::<f32>(1).expect("grad-table alloc");
    let indices = device.upload(&[0u32, 0, 0, 0]).expect("indices upload");
    let grad_output = device
        .upload(&[1e20f32, 1.0, -1e20, 1.0])
        .expect("grad-output upload");

    ops.scatter_add_into(device, &grad_table, 1, 1, &indices, &grad_output)
        .expect("scatter-add dispatch");

    let mut got = vec![0.0f32; 1];
    device.download(&grad_table, &mut got).expect("download");
    assert_eq!(
        got,
        vec![1.0],
        "{name}: repeated indices must accumulate in ascending index-position order"
    );
}

/// An index at or beyond `num_embeddings` must be a typed rejection, never a
/// clamped or wrapped write — including the empty-table edge, where no
/// accumulate thread runs and only a dedicated range check can observe the
/// fault.
///
/// # Panics
///
/// Panics when an out-of-range index is not rejected.
pub fn assert_embedding_scatter_add_rejects_out_of_range_index<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: EmbeddingOps<D, f32>,
{
    let name = device.backend_name();
    let grad_table = device.alloc_zeroed::<f32>(12).expect("grad-table alloc");
    let indices = device.upload(&[0u32, 4]).expect("indices upload");
    let grad_output = device.alloc_zeroed::<f32>(6).expect("grad-output alloc");

    let err = ops
        .scatter_add_into(device, &grad_table, 4, 3, &indices, &grad_output)
        .expect_err("index 4 is out of range for 4 embeddings");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: an out-of-range embedding index must be a typed InvalidConfiguration \
         rejection, not a silent scatter; got {err:?}"
    );

    let empty_table = device
        .alloc_zeroed::<f32>(0)
        .expect("empty grad-table alloc");
    let one_index = device.upload(&[0u32]).expect("single index upload");
    let one_grad = device
        .upload(&[1.0f32, 2.0, 3.0])
        .expect("single grad upload");

    let err = ops
        .scatter_add_into(device, &empty_table, 0, 3, &one_index, &one_grad)
        .expect_err("index 0 is out of range for 0 embeddings");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: an index into an empty table must be rejected even though no accumulate thread runs; got {err:?}"
    );
}
