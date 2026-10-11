use super::*;

pub(crate) fn elementwise_add_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    // 1027 elements: exercises a partial trailing workgroup (1027 = 4*256 + 3).
    let a_host: Vec<f32> = (0..1027).map(|i| i as f32 * 1.25).collect();
    let b_host: Vec<f32> = (0..1027).map(|i| 1000.0 - i as f32).collect();
    let expected: Vec<f32> = a_host.iter().zip(&b_host).map(|(x, y)| x + y).collect();

    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let out = binary_elementwise::<AddOp, f32>(&device, &a, &b).unwrap();

    let mut got = vec![0.0f32; a_host.len()];
    device.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

pub(crate) fn elementwise_mul_matches_cpu_reference_integral() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let a_host: Vec<u32> = (0..513).collect();
    let b_host: Vec<u32> = (0..513).map(|i| i + 7).collect();
    let expected: Vec<u32> = a_host.iter().zip(&b_host).map(|(x, y)| x * y).collect();

    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let out = binary_elementwise::<MulOp, u32>(&device, &a, &b).unwrap();

    let mut got = vec![0u32; a_host.len()];
    device.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

pub(crate) fn elementwise_rejects_input_length_mismatch() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let a = device.upload(&[1.0f32, 2.0]).unwrap();
    let b = device.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    assert_length_mismatch(binary_elementwise::<AddOp, f32>(&device, &a, &b), 2, 3);
}

pub(crate) fn elementwise_into_reuses_caller_output_buffers() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let width = BlockWidth::new(128).unwrap();
    let a_host: Vec<f32> = (0..513).map(|i| i as f32 * 0.25).collect();
    let b_host: Vec<f32> = (0..513).map(|i| 50.0 - i as f32).collect();
    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let out = device.alloc_zeroed::<f32>(a_host.len()).unwrap();

    binary_elementwise_into::<SubOp, f32>(&device, &a, &b, &out, width).unwrap();
    let mut got = vec![0.0f32; a_host.len()];
    device.download(&out, &mut got).unwrap();
    let expected: Vec<f32> = a_host.iter().zip(&b_host).map(|(x, y)| x - y).collect();
    assert_eq!(got, expected);

    unary_elementwise_into::<NegOp, f32>(&device, &a, &out, width).unwrap();
    device.download(&out, &mut got).unwrap();
    let expected: Vec<f32> = a_host.iter().map(|x| -x).collect();
    assert_eq!(got, expected);

    scalar_elementwise_into::<AddOp, f32>(&device, &a, 7.5, &out, width).unwrap();
    device.download(&out, &mut got).unwrap();
    let expected: Vec<f32> = a_host.iter().map(|x| x + 7.5).collect();
    assert_eq!(got, expected);

    let short = device.alloc_zeroed::<f32>(a_host.len() - 1).unwrap();
    assert_length_mismatch(
        binary_elementwise_into::<AddOp, f32>(&device, &a, &b, &short, width),
        short.len(),
        a.len(),
    );
    assert_length_mismatch(
        unary_elementwise_into::<NegOp, f32>(&device, &a, &short, width),
        short.len(),
        a.len(),
    );
    assert_length_mismatch(
        scalar_elementwise_into::<AddOp, f32>(&device, &a, 1.0, &short, width),
        short.len(),
        a.len(),
    );
}

pub(crate) fn elementwise_into_rejects_output_input_aliasing() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let width = BlockWidth::new(128).unwrap();
    let a = device.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let b = device.upload(&[4.0f32, 5.0, 6.0]).unwrap();

    assert_elementwise_alias_rejected(binary_elementwise_into::<AddOp, f32>(
        &device, &a, &b, &a, width,
    ));
    assert_elementwise_alias_rejected(binary_elementwise_into::<AddOp, f32>(
        &device, &a, &b, &b, width,
    ));
    assert_elementwise_alias_rejected(unary_elementwise_into::<NegOp, f32>(&device, &a, &a, width));
    assert_elementwise_alias_rejected(scalar_elementwise_into::<AddOp, f32>(
        &device, &a, 1.0, &a, width,
    ));
}

pub(crate) fn elementwise_unary_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let host = vec![-4.0f32, -1.0, 0.0, 2.0, 16.0];
    let a = device.upload(&host).unwrap();

    // SqrtOp (note: sqrt(-4.0) and sqrt(-1.0) on f32 produce NaN, we compare matching values manually)
    let out_sqrt = unary_elementwise::<SqrtOp, f32>(&device, &a).unwrap();
    let mut got_sqrt = vec![0.0f32; host.len()];
    device.download(&out_sqrt, &mut got_sqrt).unwrap();
    assert!(got_sqrt[0].is_nan());
    assert!(got_sqrt[1].is_nan());
    assert_eq!(got_sqrt[2], 0.0f32);
    assert_eq!(got_sqrt[3], std::f32::consts::SQRT_2);
    assert_eq!(got_sqrt[4], 4.0f32);

    // AbsOp
    let out_abs = unary_elementwise::<AbsOp, f32>(&device, &a).unwrap();
    let mut got_abs = vec![0.0f32; host.len()];
    device.download(&out_abs, &mut got_abs).unwrap();
    assert_eq!(got_abs, vec![4.0f32, 1.0, 0.0, 2.0, 16.0]);

    // NegOp
    let out_neg = unary_elementwise::<NegOp, f32>(&device, &a).unwrap();
    let mut got_neg = vec![0.0f32; host.len()];
    device.download(&out_neg, &mut got_neg).unwrap();
    assert_eq!(got_neg, vec![4.0f32, 1.0, 0.0, -2.0, -16.0]);

    // ExpOp
    let out_exp = unary_elementwise::<ExpOp, f32>(&device, &a).unwrap();
    let mut got_exp = vec![0.0f32; host.len()];
    device.download(&out_exp, &mut got_exp).unwrap();
    for (i, &x) in host.iter().enumerate() {
        let expected = x.exp();
        let diff = (got_exp[i] - expected).abs();
        let tolerance = 1e-5 * expected.abs().max(1.0);
        assert!(
            diff < tolerance,
            "Exp mismatch at index {}: got {}, expected {}, diff {}, tol {}",
            i,
            got_exp[i],
            expected,
            diff,
            tolerance
        );
    }

    // ExpNegOp — fused exp(−x); must equal the NegOp → ExpOp chain (differential
    // oracle) and the CPU (-x).exp() reference.
    let out_expneg = unary_elementwise::<ExpNegOp, f32>(&device, &a).unwrap();
    let mut got_expneg = vec![0.0f32; host.len()];
    device.download(&out_expneg, &mut got_expneg).unwrap();
    for (i, &x) in host.iter().enumerate() {
        let expected = (-x).exp();
        let diff = (got_expneg[i] - expected).abs();
        let tolerance = 1e-5 * expected.abs().max(1.0);
        assert!(
            diff < tolerance,
            "ExpNeg mismatch at index {}: got {}, expected {}, diff {}, tol {}",
            i,
            got_expneg[i],
            expected,
            diff,
            tolerance
        );
    }

    // RecipOp
    let host_recip = vec![1.0f32, 2.0, 4.0, 8.0];
    let b = device.upload(&host_recip).unwrap();
    let out_recip = unary_elementwise::<RecipOp, f32>(&device, &b).unwrap();
    let mut got_recip = vec![0.0f32; host_recip.len()];
    device.download(&out_recip, &mut got_recip).unwrap();
    assert_eq!(got_recip, vec![1.0f32, 0.5, 0.25, 0.125]);
}

pub(crate) fn elementwise_lgamma_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let host = [0.5_f32, 1.0, 4.0, 0.0, -1.0];
    let input = device.upload(&host).unwrap();
    let output = unary_elementwise::<LgammaOp, f32>(&device, &input).unwrap();
    let mut got = [0.0_f32; 5];
    device.download(&output, &mut got).unwrap();

    let expected = [
        0.572_364_9_f32,
        0.0,
        6.0_f32.ln(),
        f32::INFINITY,
        f32::INFINITY,
    ];
    for (index, (&actual, &reference)) in got.iter().zip(expected.iter()).enumerate() {
        if reference.is_infinite() {
            assert!(
                actual.is_infinite() && actual.is_sign_positive(),
                "lgamma[{index}] expected positive infinity, got {actual}"
            );
            continue;
        }
        let tolerance = f32::EPSILON * 2048.0 * reference.abs().max(1.0);
        assert!(
            (actual - reference).abs() <= tolerance,
            "lgamma[{index}] expected {reference}, got {actual}, tolerance {tolerance}"
        );
    }
}

pub(crate) fn elementwise_activation_markers_match_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let host = [-2.0_f32, -0.5, 0.0, 0.5, 2.0];
    let input = device.upload(&host).unwrap();
    let gelu_scale = 0.797_884_6_f32;
    let gelu_cubic = 0.044715_f32;

    macro_rules! check {
        ($label:literal, $operation:ty, $expected:expr) => {{
            let expected = $expected;
            let output = unary_elementwise::<$operation, f32>(&device, &input).unwrap();
            let mut actual = [0.0_f32; 5];
            device.download(&output, &mut actual).unwrap();
            for (index, (&got, &reference)) in actual.iter().zip(expected.iter()).enumerate() {
                let tolerance = f32::EPSILON * 512.0 * reference.abs().max(1.0);
                assert!(
                    (got - reference).abs() <= tolerance,
                    "{}[{index}] got {got}, expected {reference}, tolerance {tolerance}",
                    $label
                );
            }
        }};
    }

    check!(
        "gelu_tanh",
        GeluTanhOp,
        host.iter()
            .map(|&x| 0.5 * x * (1.0 + (gelu_scale * (x + gelu_cubic * x * x * x)).tanh()))
            .collect::<Vec<_>>()
    );
    check!(
        "gelu_tanh_grad",
        GeluTanhGradOp,
        host.iter()
            .map(|&x| {
                let tanh = (gelu_scale * (x + gelu_cubic * x * x * x)).tanh();
                0.5 * (1.0 + tanh)
                    + 0.5 * x * (1.0 - tanh * tanh) * gelu_scale * (1.0 + 3.0 * gelu_cubic * x * x)
            })
            .collect::<Vec<_>>()
    );
    check!(
        "silu",
        SiluOp,
        host.iter()
            .map(|&x| x / (1.0 + (-x).exp()))
            .collect::<Vec<_>>()
    );
    check!(
        "silu_grad",
        SiluGradOp,
        host.iter()
            .map(|&x| {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid * (1.0 + x * (1.0 - sigmoid))
            })
            .collect::<Vec<_>>()
    );
    check!(
        "softplus",
        SoftplusOp,
        host.iter()
            .map(|&x| (1.0 + x.exp()).ln())
            .collect::<Vec<_>>()
    );
    check!(
        "softplus_grad",
        SoftplusGradOp,
        host.iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect::<Vec<_>>()
    );
    check!(
        "mish",
        MishOp,
        host.iter()
            .map(|&x| x * (1.0 + x.exp()).ln().tanh())
            .collect::<Vec<_>>()
    );
    check!(
        "mish_grad",
        MishGradOp,
        host.iter()
            .map(|&x| {
                let softplus = (1.0 + x.exp()).ln();
                softplus.tanh() + x * (1.0 - softplus.tanh().powi(2)) / (1.0 + (-x).exp())
            })
            .collect::<Vec<_>>()
    );
    check!(
        "elu",
        EluOp,
        host.iter()
            .map(|&x| if x >= 0.0 { x } else { x.exp() - 1.0 })
            .collect::<Vec<_>>()
    );
    check!(
        "elu_grad",
        EluGradOp,
        host.iter()
            .map(|&x| if x >= 0.0 { 1.0 } else { x.exp() })
            .collect::<Vec<_>>()
    );
}

pub(crate) fn elementwise_scalar_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let a = device.upload(&host).unwrap();

    // scalar add
    let out_add = scalar_elementwise::<AddOp, f32>(&device, &a, 10.0).unwrap();
    let mut got_add = vec![0.0f32; host.len()];
    device.download(&out_add, &mut got_add).unwrap();
    assert_eq!(got_add, vec![11.0f32, 12.0, 13.0, 14.0, 15.0]);

    // scalar mul
    let out_mul = scalar_elementwise::<MulOp, f32>(&device, &a, 3.0).unwrap();
    let mut got_mul = vec![0.0f32; host.len()];
    device.download(&out_mul, &mut got_mul).unwrap();
    assert_eq!(got_mul, vec![3.0f32, 6.0, 9.0, 12.0, 15.0]);
}
