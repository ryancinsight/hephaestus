//! Contract clauses for the device-neutral [`hephaestus_core::TileOps`] seam.
//!
//! There is no `leto::tile` to delegate to, so the host reference carries
//! the direct implementation and these clauses pin its value semantics
//! with hand-computed fixtures: `output[i] = input[i with each lane
//! taken modulo the input extent]` — the `torch.Tensor.repeat` contract
//! `coeus_ops::tile` documents, restricted to the fixed-rank case.
//! Fixtures use small signed integers so every copy is an exact equality,
//! never a tolerance-bounded comparison.

use hephaestus_core::{ComputeDevice, HephaestusError, TileOps};
use leto::Layout;

/// Tile against hand-computed fixtures: a 1-D input repeated twice, and a
/// 2x2 input repeated `[2, 3]` along its axes.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees with the
/// fixture.
pub fn assert_tile_value_contract<D, L, T>(device: &D, ops: &L)
where
    D: ComputeDevice,
    L: TileOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();

    // 1-D: [1, 2, 3] repeated twice.
    let host: Vec<T> = [1i8, 2, 3].map(T::from).to_vec();
    let in_layout = Layout::c_contiguous([3]).expect("input layout");
    let out_layout = Layout::c_contiguous([6]).expect("output layout");
    let input = device.upload(&host).expect("input upload");
    let output = device.alloc_zeroed::<T>(6).expect("output alloc");
    ops.tile_into::<1>(
        device,
        hephaestus_core::StridedView::new(&input, &in_layout),
        [2],
        hephaestus_core::StridedView::new(&output, &out_layout),
    )
    .expect("tile dispatch");

    let expected: Vec<T> = [1i8, 2, 3, 1, 2, 3].map(T::from).to_vec();
    let mut got = vec![T::from(0i8); 6];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, expected,
        "{name}: 1-D tile must match the hand-computed fixture exactly"
    );

    // 2-D: [[1, 2], [3, 4]] repeated [2, 3] into a 4x6 output.
    let square: Vec<T> = [1i8, 2, 3, 4].map(T::from).to_vec();
    let square_in = Layout::c_contiguous([2, 2]).expect("square input layout");
    let square_out = Layout::c_contiguous([4, 6]).expect("square output layout");
    let square_input = device.upload(&square).expect("square upload");
    let square_output = device.alloc_zeroed::<T>(24).expect("square output alloc");
    ops.tile_into::<2>(
        device,
        hephaestus_core::StridedView::new(&square_input, &square_in),
        [2, 3],
        hephaestus_core::StridedView::new(&square_output, &square_out),
    )
    .expect("tile dispatch");

    // Rows 0 and 2 read [1, 2, 1, 2, 1, 2]; rows 1 and 3 read
    // [3, 4, 3, 4, 3, 4].
    let square_expected: Vec<T> = [
        1i8, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4,
    ]
    .map(T::from)
    .to_vec();
    let mut square_got = vec![T::from(0i8); 24];
    device
        .download(&square_output, &mut square_got)
        .expect("download");
    assert_eq!(
        square_got, square_expected,
        "{name}: 2-D tile must match the hand-computed fixture exactly"
    );
}

/// All-ones reps reproduce the input exactly: tiling once along every
/// axis is the identity.
///
/// # Panics
///
/// Panics when an identity tile changes any element.
pub fn assert_tile_identity_reps<D, L, T>(device: &D, ops: &L)
where
    D: ComputeDevice,
    L: TileOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..6i8).map(T::from).collect();
    let layout = Layout::c_contiguous([2, 3]).expect("layout");
    let input = device.upload(&host).expect("input upload");
    let output = device.alloc_zeroed::<T>(6).expect("output alloc");

    ops.tile_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &layout),
        [1, 1],
        hephaestus_core::StridedView::new(&output, &layout),
    )
    .expect("identity tile must dispatch");

    let mut got = vec![T::from(0i8); 6];
    device.download(&output, &mut got).expect("download");
    assert_eq!(
        got, host,
        "{name}: all-ones reps must reproduce the input exactly"
    );
}

/// Empty tilings dispatch vacuously: a zero rep count empties its axis,
/// and an empty input stays empty under any reps. Either way zero cells
/// copy and no modulo by zero executes.
///
/// # Panics
///
/// Panics when a vacuous tile fails to dispatch.
pub fn assert_tile_of_empty_is_vacuous<D, L, T>(device: &D, ops: &L)
where
    D: ComputeDevice,
    L: TileOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    // A zero rep empties its axis: [2, 3] tiled [2, 0] yields [4, 0].
    let host: Vec<T> = (0..6i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([2, 3]).expect("input layout");
    let empty_layout = Layout::c_contiguous([4, 0]).expect("empty layout");
    let input = device.upload(&host).expect("input upload");
    let output = device.alloc_zeroed::<T>(0).expect("output alloc");
    ops.tile_into::<2>(
        device,
        hephaestus_core::StridedView::new(&input, &in_layout),
        [2, 0],
        hephaestus_core::StridedView::new(&output, &empty_layout),
    )
    .expect("zero-rep tile must dispatch");

    // An empty input stays empty: [2, 0] tiled [3, 5] yields [6, 0].
    let empty_in = Layout::c_contiguous([2, 0]).expect("empty input layout");
    let empty_out = Layout::c_contiguous([6, 0]).expect("empty output layout");
    let empty_input = device.alloc_zeroed::<T>(0).expect("empty input alloc");
    let empty_output = device.alloc_zeroed::<T>(0).expect("empty output alloc");
    ops.tile_into::<2>(
        device,
        hephaestus_core::StridedView::new(&empty_input, &empty_in),
        [3, 5],
        hephaestus_core::StridedView::new(&empty_output, &empty_out),
    )
    .expect("empty-input tile must dispatch");
}

/// Invalid geometry must fail typed: an output extent that is not its
/// input extent times its rep count, and an overflowing rep count, each
/// return [`HephaestusError::InvalidConfiguration`] rather than
/// dispatching. Shape validation runs before any dispatch, so the
/// overflow case needs no large buffers.
///
/// # Panics
///
/// Panics when any invalid geometry dispatches or fails with another error.
pub fn assert_tile_rejects_invalid_geometry<D, L, T>(device: &D, ops: &L)
where
    D: ComputeDevice,
    L: TileOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let in_layout = Layout::c_contiguous([2, 2]).expect("input layout");
    let input = device.alloc_zeroed::<T>(4).expect("input alloc");

    let short_layout = Layout::c_contiguous([4, 5]).expect("short layout");
    let short = device.alloc_zeroed::<T>(20).expect("short alloc");
    let err = ops
        .tile_into::<2>(
            device,
            hephaestus_core::StridedView::new(&input, &in_layout),
            [2, 3],
            hephaestus_core::StridedView::new(&short, &short_layout),
        )
        .expect_err("extent mismatch must fail");
    assert!(
        matches!(err, HephaestusError::InvalidConfiguration { .. }),
        "{name}: extent mismatch must fail typed, got {err:?}"
    );

    let tiny = device.alloc_zeroed::<T>(0).expect("tiny alloc");
    let tiny_layout = Layout::c_contiguous([0]).expect("tiny layout");
    let flat_in = Layout::c_contiguous([2]).expect("flat input layout");
    let flat_input = device.alloc_zeroed::<T>(2).expect("flat input alloc");
    let err = ops
        .tile_into::<1>(
            device,
            hephaestus_core::StridedView::new(&flat_input, &flat_in),
            [usize::MAX],
            hephaestus_core::StridedView::new(&tiny, &tiny_layout),
        )
        .expect_err("overflowing rep must fail");
    assert!(
        matches!(&err, HephaestusError::InvalidConfiguration { message } if message.contains("overflows")),
        "{name}: overflowing rep must fail typed, got {err:?}"
    );
}

/// An output aliasing the input must fail before mutating anything: reads
/// and writes race across threads, so sharing storage is a dispatch error
/// on every backend, not just the device ones.
///
/// # Panics
///
/// Panics when aliased dispatch succeeds or mutates the shared buffer.
pub fn assert_tile_rejects_aliased_output<D, L, T>(device: &D, ops: &L)
where
    D: ComputeDevice,
    L: TileOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    let name = device.backend_name();
    let host: Vec<T> = (0..24i8).map(T::from).collect();
    let in_layout = Layout::c_contiguous([2, 2]).expect("input layout");
    let out_layout = Layout::c_contiguous([4, 6]).expect("output layout");

    // The shared buffer holds all 24 output cells, so both the input view
    // and the tiled output view fit its storage: the geometry is fully
    // valid and only the aliasing fails.
    let shared = device.upload(&host).expect("shared upload");

    let before = device
        .download_owned(&shared)
        .expect("alias fixture download");
    let error = ops
        .tile_into::<2>(
            device,
            hephaestus_core::StridedView::new(&shared, &in_layout),
            [2, 3],
            hephaestus_core::StridedView::new(&shared, &out_layout),
        )
        .expect_err("tile dispatch must reject aliased input/output");
    assert!(
        error.to_string().contains("must not alias"),
        "{name}: unexpected alias error: {error}"
    );
    let after = device
        .download_owned(&shared)
        .expect("alias fixture download");
    assert_eq!(after, before, "{name}: alias rejection mutated input");
}

/// Run every clause in this module against one backend.
///
/// # Panics
///
/// Panics with the first violated clause.
pub fn assert_tile_contract<D, L, T>(device: &D, ops: &L)
where
    D: ComputeDevice,
    L: TileOps<D, T>,
    T: leto_ops::Scalar + From<i8> + eunomia::Pod,
{
    assert_tile_value_contract(device, ops);
    assert_tile_identity_reps(device, ops);
    assert_tile_of_empty_is_vacuous(device, ops);
    assert_tile_rejects_invalid_geometry(device, ops);
    assert_tile_rejects_aliased_output(device, ops);
}
