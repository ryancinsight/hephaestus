use super::*;

pub(crate) fn reduction_sum_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let test_sizes = [0, 1, 255, 256, 257, 1027];

    for &size in &test_sizes {
        // f32
        let host_f32: Vec<f32> = (0..size).map(|i| i as f32 * 0.5).collect();
        let expected_f32: f32 = host_f32.iter().sum();
        let buf_f32 = device.upload(&host_f32).unwrap();
        let out_f32 = reduction::<SumOp, f32>(&device, &buf_f32).unwrap();
        let mut got_f32 = vec![0.0f32; 1];
        device.download(&out_f32, &mut got_f32).unwrap();
        assert_eq!(
            got_f32[0], expected_f32,
            "f32 sum mismatch at size {}",
            size
        );
        let prepared_f32 = prepare_reduction::<SumOp, f32>(&device, &buf_f32).unwrap();
        prepared_f32.dispatch(&device).unwrap();
        let mut got_prepared_f32 = vec![0.0f32; 1];
        device
            .download(prepared_f32.output(), &mut got_prepared_f32)
            .unwrap();
        assert_eq!(
            got_prepared_f32[0], expected_f32,
            "prepared f32 sum mismatch at size {}",
            size
        );

        // u32
        let host_u32: Vec<u32> = (0..size).map(|i| i as u32).collect();
        let expected_u32: u32 = host_u32.iter().sum();
        let buf_u32 = device.upload(&host_u32).unwrap();
        let out_u32 = reduction::<SumOp, u32>(&device, &buf_u32).unwrap();
        let mut got_u32 = vec![0u32; 1];
        device.download(&out_u32, &mut got_u32).unwrap();
        assert_eq!(
            got_u32[0], expected_u32,
            "u32 sum mismatch at size {}",
            size
        );
        let prepared_u32 = prepare_reduction::<SumOp, u32>(&device, &buf_u32).unwrap();
        prepared_u32.dispatch(&device).unwrap();
        let mut got_prepared_u32 = vec![0u32; 1];
        device
            .download(prepared_u32.output(), &mut got_prepared_u32)
            .unwrap();
        assert_eq!(
            got_prepared_u32[0], expected_u32,
            "prepared u32 sum mismatch at size {}",
            size
        );
        if size == 0 {
            submit_prepared_reduction_batch::<u32>(&device, &[]).unwrap();
            submit_prepared_reduction_batch(&device, &[&prepared_u32]).unwrap();
            device
                .download(prepared_u32.output(), &mut got_prepared_u32)
                .unwrap();
            assert_eq!(got_prepared_u32, [0]);
        }
        if size == 1027 {
            let batch_a = prepare_reduction::<SumOp, u32>(&device, &buf_u32).unwrap();
            let singleton_input = device.upload(&[99u32]).unwrap();
            let singleton = prepare_reduction::<SumOp, u32>(&device, &singleton_input).unwrap();
            let batch_b = prepare_reduction::<SumOp, u32>(&device, &buf_u32).unwrap();
            submit_prepared_reduction_batch(&device, &[&batch_a, &singleton, &batch_b]).unwrap();
            let mut got_batch_a = vec![0u32; 1];
            let mut got_singleton = vec![0u32; 1];
            let mut got_batch_b = vec![0u32; 1];
            device.download(batch_a.output(), &mut got_batch_a).unwrap();
            device
                .download(singleton.output(), &mut got_singleton)
                .unwrap();
            device.download(batch_b.output(), &mut got_batch_b).unwrap();
            assert_eq!(got_batch_a[0], expected_u32);
            assert_eq!(got_singleton[0], 99);
            assert_eq!(got_batch_b[0], expected_u32);
        }

        // i32
        let host_i32: Vec<i32> = (0..size).map(|i| if i % 2 == 0 { i } else { -i }).collect();
        let expected_i32: i32 = host_i32.iter().sum();
        let buf_i32 = device.upload(&host_i32).unwrap();
        let out_i32 = reduction::<SumOp, i32>(&device, &buf_i32).unwrap();
        let mut got_i32 = vec![0i32; 1];
        device.download(&out_i32, &mut got_i32).unwrap();
        assert_eq!(
            got_i32[0], expected_i32,
            "i32 sum mismatch at size {}",
            size
        );
    }
}

pub(crate) fn reduction_min_max_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let test_sizes = [0, 1, 255, 256, 257, 1027];

    for &size in &test_sizes {
        // f32 Min/Max
        let host_f32: Vec<f32> = (0..size)
            .map(|i| (i as f32 * 12.34 - 100.0).sin())
            .collect();
        let expected_min_f32 = if size == 0 {
            f32::MAX
        } else {
            host_f32.iter().copied().fold(f32::NAN, f32::min)
        };
        let expected_max_f32 = if size == 0 {
            f32::MIN
        } else {
            host_f32.iter().copied().fold(f32::NAN, f32::max)
        };

        let buf_f32 = device.upload(&host_f32).unwrap();

        let out_min_f32 = reduction::<MinOp, f32>(&device, &buf_f32).unwrap();
        let mut got_min_f32 = vec![0.0f32; 1];
        device.download(&out_min_f32, &mut got_min_f32).unwrap();
        assert_eq!(
            got_min_f32[0], expected_min_f32,
            "f32 min mismatch at size {}",
            size
        );

        let out_max_f32 = reduction::<MaxOp, f32>(&device, &buf_f32).unwrap();
        let mut got_max_f32 = vec![0.0f32; 1];
        device.download(&out_max_f32, &mut got_max_f32).unwrap();
        assert_eq!(
            got_max_f32[0], expected_max_f32,
            "f32 max mismatch at size {}",
            size
        );

        // i32 Min/Max
        let host_i32: Vec<i32> = (0..size)
            .map(|i| if i % 3 == 0 { i * 7 } else { -(i * 5) })
            .collect();
        let expected_min_i32 = if size == 0 {
            i32::MAX
        } else {
            *host_i32.iter().min().unwrap()
        };
        let expected_max_i32 = if size == 0 {
            i32::MIN
        } else {
            *host_i32.iter().max().unwrap()
        };

        let buf_i32 = device.upload(&host_i32).unwrap();

        let out_min_i32 = reduction::<MinOp, i32>(&device, &buf_i32).unwrap();
        let mut got_min_i32 = vec![0i32; 1];
        device.download(&out_min_i32, &mut got_min_i32).unwrap();
        assert_eq!(
            got_min_i32[0], expected_min_i32,
            "i32 min mismatch at size {}",
            size
        );

        let out_max_i32 = reduction::<MaxOp, i32>(&device, &buf_i32).unwrap();
        let mut got_max_i32 = vec![0i32; 1];
        device.download(&out_max_i32, &mut got_max_i32).unwrap();
        assert_eq!(
            got_max_i32[0], expected_max_i32,
            "i32 max mismatch at size {}",
            size
        );
    }
}

pub(crate) fn reduction_width_is_part_of_dispatch_contract() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let host: Vec<u32> = (0..1027).collect();
    let expected: u32 = host.iter().sum();
    let input = device.upload(&host).unwrap();

    let narrow = BlockWidth::new(128).unwrap();
    let out_narrow = reduction_with_width::<SumOp, u32>(&device, &input, narrow).unwrap();
    let mut got_narrow = vec![0u32; 1];
    device.download(&out_narrow, &mut got_narrow).unwrap();
    assert_eq!(got_narrow[0], expected);

    let non_power = BlockWidth::new(192).unwrap();
    assert_dispatch_message(
        reduction_with_width::<SumOp, u32>(&device, &input, non_power),
        "reduction block width 192 must be a power of two",
    );

    let empty = device.upload::<u32>(&[]).unwrap();
    assert_dispatch_message(
        reduction_with_width::<SumOp, u32>(&device, &empty, non_power),
        "reduction block width 192 must be a power of two",
    );

    let singleton = device.upload(&[13u32]).unwrap();
    assert_dispatch_message(
        reduction_with_width::<SumOp, u32>(&device, &singleton, non_power),
        "reduction block width 192 must be a power of two",
    );
}

pub(crate) fn linalg_reductions_accept_strided_views() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{StridedOperand, dot, norm_l1, norm_l2, norm_max};
    use leto::Layout;

    let a_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_host = vec![10.0f32, 20.0, 30.0, 40.0];
    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let reversed = Layout::try_new([4], [-1], 3).expect("valid test layout");
    let contiguous = Layout::c_contiguous([4]).unwrap();
    let reversed_a = StridedOperand {
        buffer: &a,
        layout: &reversed,
    };
    let contiguous_b = StridedOperand {
        buffer: &b,
        layout: &contiguous,
    };

    let dot_buf = dot(&device, reversed_a, contiguous_b).unwrap();
    let mut got_dot = [0.0f32; 1];
    device.download(&dot_buf, &mut got_dot).unwrap();
    assert_eq!(
        got_dot[0],
        4.0 * 10.0 + 3.0 * 20.0 + 2.0 * 30.0 + 1.0 * 40.0
    );

    let l1_buf = norm_l1(&device, reversed_a).unwrap();
    let mut got_l1 = [0.0f32; 1];
    device.download(&l1_buf, &mut got_l1).unwrap();
    assert_eq!(got_l1[0], 10.0);

    let l2_buf = norm_l2(&device, reversed_a).unwrap();
    let mut got_l2 = [0.0f32; 1];
    device.download(&l2_buf, &mut got_l2).unwrap();
    let expected_l2 = 30.0f32.sqrt();
    let l2_tolerance = 2.0 * f32::EPSILON * expected_l2.max(1.0);
    assert!(
        (got_l2[0] - expected_l2).abs() <= l2_tolerance,
        "l2 norm mismatch: got {}, expected {}, tolerance {}",
        got_l2[0],
        expected_l2,
        l2_tolerance
    );

    let max_buf = norm_max(&device, reversed_a).unwrap();
    let mut got_max = [0.0f32; 1];
    device.download(&max_buf, &mut got_max).unwrap();
    assert_eq!(got_max[0], 4.0);
}
