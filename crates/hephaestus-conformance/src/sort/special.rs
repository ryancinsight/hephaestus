//! IEEE float special-values (`-0.0`, `-inf`, `NaN`) clauses for [`SortOps`].

use hephaestus_core::{ComputeDevice, SortOps};
use leto::Layout;

/// The float order edges are pinned by IEEE comparison in two layers.
/// Every backend, on every dialect, must sort `-0.0` as equal to `0.0`
/// (stable by input order) and `-inf` below everything finite:
/// `[-0.0, 1.0, 0.0, -inf]` sorts to `[-inf, -0.0, 0.0, 1.0]` with
/// indices `[3, 0, 2, 1]` ascending, and to `[1.0, -0.0, 0.0, -inf]`
/// with indices `[1, 0, 2, 3]` descending — a naive bitwise comparator
/// fails the signed-zero cells.
///
/// The `NaN` fixtures assert only where the dialect advertises IEEE
/// special values (`KernelDialect::IEEE_SPECIAL_VALUES`, ADR 0043), and
/// they skip wholesale rather than per cell: one unordered `NaN`
/// comparison can corrupt a whole lane's order on a non-advertising
/// dialect, so no assertion about a `NaN`-mixed lane can hold there.
/// Where they run, `NaN` sorts as greater than everything: ascending,
/// `[-0.0, NaN, 1.0, -inf, 0.0]` sorts to
/// `[-inf, -0.0, 0.0, 1.0, NaN]` with indices `[3, 0, 4, 2, 1]` and
/// `[NaN, 1.0, NaN]` to `[1.0, NaN, NaN]` with indices `[1, 0, 2]`
/// (`NaN`s stable among themselves); descending, the first lane sorts to
/// `[NaN, 1.0, -0.0, 0.0, -inf]` with indices `[1, 2, 0, 4, 3]`.
///
/// # Panics
///
/// Panics when the backend misorders a special-values fixture.
pub fn assert_sort_special_float_values<D, S>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: SortOps<D, f32>,
{
    let name = device.backend_name();

    // No NaN: every backend asserts.
    let clean_host: Vec<f32> = vec![-0.0, 1.0, 0.0, f32::NEG_INFINITY];
    let clean_layout = Layout::c_contiguous([4]).expect("clean layout");
    let clean_input = device.upload(&clean_host).expect("clean upload");
    let clean = || hephaestus_core::StridedView::new(&clean_input, &clean_layout);
    let clean_values = device.alloc_zeroed::<f32>(4).expect("clean values alloc");
    let clean_indices = device.alloc_zeroed::<u32>(4).expect("clean indices alloc");
    ops.sort_into::<1>(
        device,
        clean(),
        0,
        false,
        hephaestus_core::StridedView::new(&clean_values, &clean_layout),
        hephaestus_core::StridedView::new(&clean_indices, &clean_layout),
    )
    .expect("clean ascending dispatch");
    let mut clean_got = vec![0.0f32; 4];
    let mut clean_idx = vec![0u32; 4];
    device
        .download(&clean_values, &mut clean_got)
        .expect("download");
    device
        .download(&clean_indices, &mut clean_idx)
        .expect("download");
    assert_eq!(
        clean_got,
        vec![f32::NEG_INFINITY, -0.0, 0.0, 1.0],
        "{name}: ascending -0.0/-inf values must match exactly"
    );
    assert_eq!(
        clean_idx,
        vec![3u32, 0, 2, 1],
        "{name}: ascending -0.0 must stay ahead of +0.0 by input order"
    );

    let clean_down_values = device.alloc_zeroed::<f32>(4).expect("clean values alloc");
    let clean_down_indices = device.alloc_zeroed::<u32>(4).expect("clean indices alloc");
    ops.sort_into::<1>(
        device,
        clean(),
        0,
        true,
        hephaestus_core::StridedView::new(&clean_down_values, &clean_layout),
        hephaestus_core::StridedView::new(&clean_down_indices, &clean_layout),
    )
    .expect("clean descending dispatch");
    let mut clean_down_got = vec![0.0f32; 4];
    let mut clean_down_idx = vec![0u32; 4];
    device
        .download(&clean_down_values, &mut clean_down_got)
        .expect("download");
    device
        .download(&clean_down_indices, &mut clean_down_idx)
        .expect("download");
    assert_eq!(
        clean_down_got,
        vec![1.0, -0.0, 0.0, f32::NEG_INFINITY],
        "{name}: descending -0.0/-inf values must match exactly"
    );
    assert_eq!(
        clean_down_idx,
        vec![1u32, 0, 2, 3],
        "{name}: descending signed zeros must keep input order"
    );

    // NaN-mixed: only IEEE-advertising dialects assert, wholesale.
    if <S::Dialect as hephaestus_core::KernelDialect>::IEEE_SPECIAL_VALUES {
        let nan_host: Vec<f32> = vec![-0.0, f32::NAN, 1.0, f32::NEG_INFINITY, 0.0];
        let nan_layout = Layout::c_contiguous([5]).expect("nan layout");
        let nan_input = device.upload(&nan_host).expect("nan upload");
        let nan = || hephaestus_core::StridedView::new(&nan_input, &nan_layout);
        let nan_values = device.alloc_zeroed::<f32>(5).expect("nan values alloc");
        let nan_indices = device.alloc_zeroed::<u32>(5).expect("nan indices alloc");
        ops.sort_into::<1>(
            device,
            nan(),
            0,
            false,
            hephaestus_core::StridedView::new(&nan_values, &nan_layout),
            hephaestus_core::StridedView::new(&nan_indices, &nan_layout),
        )
        .expect("nan ascending dispatch");
        let mut nan_got = vec![0.0f32; 5];
        let mut nan_idx = vec![0u32; 5];
        device
            .download(&nan_values, &mut nan_got)
            .expect("download");
        device
            .download(&nan_indices, &mut nan_idx)
            .expect("download");
        assert_eq!(
            nan_idx,
            vec![3u32, 0, 4, 2, 1],
            "{name}: ascending NaN lane indices must match exactly"
        );
        assert!(
            nan_got[4].is_nan(),
            "{name}: ascending NaN must sort last, got {nan_got:?}"
        );
        assert_eq!(
            &nan_got[..4],
            &[f32::NEG_INFINITY, -0.0, 0.0, 1.0],
            "{name}: ascending NaN lane finite values must match exactly"
        );

        let nan_down_values = device.alloc_zeroed::<f32>(5).expect("nan values alloc");
        let nan_down_indices = device.alloc_zeroed::<u32>(5).expect("nan indices alloc");
        ops.sort_into::<1>(
            device,
            nan(),
            0,
            true,
            hephaestus_core::StridedView::new(&nan_down_values, &nan_layout),
            hephaestus_core::StridedView::new(&nan_down_indices, &nan_layout),
        )
        .expect("nan descending dispatch");
        let mut nan_down_got = vec![0.0f32; 5];
        let mut nan_down_idx = vec![0u32; 5];
        device
            .download(&nan_down_values, &mut nan_down_got)
            .expect("download");
        device
            .download(&nan_down_indices, &mut nan_down_idx)
            .expect("download");
        assert_eq!(
            nan_down_idx,
            vec![1u32, 2, 0, 4, 3],
            "{name}: descending NaN lane indices must match exactly"
        );
        assert!(
            nan_down_got[0].is_nan(),
            "{name}: descending NaN must sort first, got {nan_down_got:?}"
        );
        assert_eq!(
            &nan_down_got[1..],
            &[1.0, -0.0, 0.0, f32::NEG_INFINITY],
            "{name}: descending NaN lane finite values must match exactly"
        );

        // NaNs stable among themselves: [NaN, 1.0, NaN] -> [1.0, NaN, NaN].
        let twins_host: Vec<f32> = vec![f32::NAN, 1.0, f32::NAN];
        let twins_layout = Layout::c_contiguous([3]).expect("twins layout");
        let twins_input = device.upload(&twins_host).expect("twins upload");
        let twins_values = device.alloc_zeroed::<f32>(3).expect("twins values alloc");
        let twins_indices = device.alloc_zeroed::<u32>(3).expect("twins indices alloc");
        ops.sort_into::<1>(
            device,
            hephaestus_core::StridedView::new(&twins_input, &twins_layout),
            0,
            false,
            hephaestus_core::StridedView::new(&twins_values, &twins_layout),
            hephaestus_core::StridedView::new(&twins_indices, &twins_layout),
        )
        .expect("nan twins dispatch");
        let mut twins_got = vec![0.0f32; 3];
        let mut twins_idx = vec![0u32; 3];
        device
            .download(&twins_values, &mut twins_got)
            .expect("download");
        device
            .download(&twins_indices, &mut twins_idx)
            .expect("download");
        assert_eq!(
            twins_idx,
            vec![1u32, 0, 2],
            "{name}: NaNs must stay stable among themselves"
        );
        assert_eq!(twins_got[0], 1.0, "{name}: NaN twins finite cell mismatch");
        assert!(
            twins_got[1].is_nan() && twins_got[2].is_nan(),
            "{name}: NaN twins must both be NaN, got {twins_got:?}"
        );
    }
}
