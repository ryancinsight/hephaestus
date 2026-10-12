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

/// Scattering gradient rows into a zeroed table must accumulate each row at
/// its index: repeated indices sum in ascending input order, unreferenced
/// rows stay zero, and a second scatter adds onto the existing contents
/// rather than overwriting them.
///
/// All values are small integers, so every sum is exact in `f32`.
pub fn assert_embedding_scatter_add_contract<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: EmbeddingOps<D, f32>,
{
    let name = device.backend_name();
    // grad rows r0=[1,2], r1=[4,8], r2=[16,32]; indices [2,0,2] into a
    // 4x2 table. Row 2 sees r0 then r2: [1+16, 2+32] = [17,34].
    let grad_out = device
        .upload(&[1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0])
        .expect("grad upload");
    let indices = device.upload(&[2u32, 0, 2]).expect("indices upload");
    let grad_table = device.alloc_zeroed::<f32>(8).expect("table alloc");

    ops.scatter_add_into(device, &grad_table, 4, 2, &indices, &grad_out)
        .expect("scatter dispatch");

    let mut got = vec![0.0f32; 8];
    device.download(&grad_table, &mut got).expect("download");
    assert_eq!(
        got,
        vec![4.0, 8.0, 0.0, 0.0, 17.0, 34.0, 0.0, 0.0],
        "{name}: scattered rows must accumulate at their indices in ascending input order"
    );

    ops.scatter_add_into(device, &grad_table, 4, 2, &indices, &grad_out)
        .expect("second scatter dispatch");
    device.download(&grad_table, &mut got).expect("download");
    assert_eq!(
        got,
        vec![8.0, 16.0, 0.0, 0.0, 34.0, 68.0, 0.0, 0.0],
        "{name}: scatter-add must accumulate onto the existing table contents"
    );
}

/// An index at or beyond `num_embeddings` must be a typed rejection, never
/// a clamped or wrapped write — including against an empty table, where no
/// output row exists to detect it on-device.
pub fn assert_embedding_scatter_add_rejects_out_of_range_index<D, E>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: EmbeddingOps<D, f32>,
{
    let name = device.backend_name();
    let grad_out = device
        .upload(&[1.0f32, 2.0, 4.0, 8.0])
        .expect("grad upload");
    let indices = device.upload(&[0u32, 4]).expect("indices upload");
    let grad_table = device.alloc_zeroed::<f32>(8).expect("table alloc");

    let err = ops
        .scatter_add_into(device, &grad_table, 4, 2, &indices, &grad_out)
        .expect_err("index 4 is out of range for 4 embeddings");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: an out-of-range embedding index must be a typed InvalidConfiguration \
         rejection, not silently scattered; got {err:?}"
    );

    let empty_table = device.alloc_zeroed::<f32>(0).expect("empty table alloc");
    let one_row = device.upload(&[1.0f32, 2.0]).expect("single grad row");
    let one_index = device.upload(&[0u32]).expect("single index");
    let err = ops
        .scatter_add_into(device, &empty_table, 0, 2, &one_index, &one_row)
        .expect_err("every index is out of range for an empty table");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: an index against an empty table must be a typed InvalidConfiguration \
         rejection; got {err:?}"
    );
}
