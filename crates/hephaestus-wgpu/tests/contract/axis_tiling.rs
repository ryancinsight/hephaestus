use super::*;

/// Pins the WG-P5 grid-stride fix at the scale it targets: an axis longer
/// than `BlockWidth::DEFAULT` (256), which previously fell to a genuinely
/// serial one-thread-per-row kernel and now runs through the grid-strided
/// workgroup-parallel kernel for every lane. The per-lane stride
/// accumulation reassociates the float sum relative to Leto's sequential
/// CPU reference (elements interleave across lanes before the tree
/// reduction), so this asserts a derived epsilon bound rather than exact
/// equality — the naive-summation error model is O(n*eps*sum(|x_i|)); the
/// bound below is that model with headroom for the extra tree-reduction
/// level, not a widened-to-pass tolerance.
pub(crate) fn axis_reduction_grid_stride_matches_leto_reference_beyond_block_width() {
    use hephaestus_wgpu::StridedOperand;
    use leto::Layout;

    let Some(device) = device_or_skip() else {
        return;
    };

    let axis_len = 500usize; // > BlockWidth::DEFAULT (256): forces the
    // previously-serial path onto the grid-strided kernel.
    let rows = 3usize;
    let host: Vec<f32> = (0..rows * axis_len)
        .map(|i| ((i % 97) as f32) * 0.1 + 0.01)
        .collect();
    let input = device.upload(&host).unwrap();
    let input_layout = Layout::c_contiguous([rows, axis_len]).unwrap();
    let input_operand = StridedOperand {
        buffer: &input,
        layout: &input_layout,
    };
    let leto_input = leto::Array::from_shape_vec([rows, axis_len], host).unwrap();

    let got = sum_axis(&device, input_operand, 1, BlockWidth::DEFAULT).unwrap();
    let mut got_host = vec![0.0f32; rows];
    device.download(&got, &mut got_host).unwrap();

    let expected = leto_ops::sum_axis(&leto_input.view(), 1)
        .unwrap()
        .into_vec();

    for row in 0..rows {
        let view = leto_input.view();
        let abs_sum: f32 = (0..axis_len).map(|col| view[[row, col]].abs()).sum();
        let tol = abs_sum * axis_len as f32 * f32::EPSILON * 4.0;
        assert!(
            (got_host[row] - expected[row]).abs() <= tol,
            "row {row}: grid-strided sum {} must match Leto's sequential sum {} within {tol} \
             (derived from O(n*eps*sum|x|) with tree-reduction headroom)",
            got_host[row],
            expected[row]
        );
    }
}

pub(crate) fn axis_zero_tiling_matches_exact_multi_tile_oracles() {
    use hephaestus_wgpu::StridedOperand;
    use leto::Layout;

    let Some(device) = device_or_skip() else {
        return;
    };

    const EXTENT: usize = 256;
    let host: Vec<f32> = (0..EXTENT)
        .flat_map(|row| (0..EXTENT).map(move |col| (row + col) as f32))
        .collect();
    let input = device.upload(&host).unwrap();
    let input_layout = Layout::c_contiguous([EXTENT, EXTENT]).unwrap();
    let operand = StridedOperand {
        buffer: &input,
        layout: &input_layout,
    };

    let expected_sum: Vec<f32> = (0..EXTENT)
        .map(|col| 32_640.0 + 256.0 * col as f32)
        .collect();
    let expected_min: Vec<f32> = (0..EXTENT).map(|col| col as f32).collect();
    let expected_max: Vec<f32> = (0..EXTENT).map(|col| 255.0 + col as f32).collect();
    let expected_mean: Vec<f32> = (0..EXTENT).map(|col| 127.5 + col as f32).collect();

    for (result, expected) in [
        (
            sum_axis(&device, operand, 0, BlockWidth::DEFAULT),
            &expected_sum,
        ),
        (
            min_axis(&device, operand, 0, BlockWidth::DEFAULT),
            &expected_min,
        ),
        (
            max_axis(&device, operand, 0, BlockWidth::DEFAULT),
            &expected_max,
        ),
        (
            mean_axis(&device, operand, 0, BlockWidth::DEFAULT),
            &expected_mean,
        ),
    ] {
        let output = result.unwrap();
        let mut actual = vec![0.0; EXTENT];
        device.download(&output, &mut actual).unwrap();
        assert_eq!(&actual, expected);
    }

    let narrow = sum_axis(
        &device,
        operand,
        0,
        BlockWidth::new(32).expect("non-zero width"),
    )
    .unwrap();
    let mut narrow_actual = vec![0.0; EXTENT];
    device.download(&narrow, &mut narrow_actual).unwrap();
    assert_eq!(narrow_actual, expected_sum);
}
