use super::*;

#[test]
fn elementwise_add_matches_cpu_reference() {
    let Some(dev) = device("elementwise_add_matches_cpu_reference") else {
        return;
    };
    let a_host: Vec<f32> = (0..1027).map(|i| i as f32 * 1.25).collect();
    let b_host: Vec<f32> = (0..1027).map(|i| 1000.0 - i as f32).collect();
    let expected: Vec<f32> = a_host.iter().zip(&b_host).map(|(x, y)| x + y).collect();

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = binary_elementwise::<AddOp, f32>(&dev, &a, &b).unwrap();

    let mut got = vec![0.0f32; a_host.len()];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn elementwise_mul_matches_cpu_reference_integral() {
    let Some(dev) = device("elementwise_mul_matches_cpu_reference_integral") else {
        return;
    };
    let a_host: Vec<u32> = (0..513).collect();
    let b_host: Vec<u32> = (0..513).map(|i| i + 7).collect();
    let expected: Vec<u32> = a_host.iter().zip(&b_host).map(|(x, y)| x * y).collect();

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = binary_elementwise::<MulOp, u32>(&dev, &a, &b).unwrap();

    let mut got = vec![0u32; a_host.len()];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn elementwise_rejects_input_length_mismatch() {
    let Some(dev) = device("elementwise_rejects_input_length_mismatch") else {
        return;
    };
    let a = dev.upload(&[1.0f32, 2.0]).unwrap();
    let b = dev.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    assert_length_mismatch(binary_elementwise::<AddOp, f32>(&dev, &a, &b), 2, 3);
}

#[test]
fn elementwise_into_reuses_caller_output_buffers() {
    let Some(dev) = device("elementwise_into_reuses_caller_output_buffers") else {
        return;
    };
    let width = BlockWidth::new(128).unwrap();
    let a_host: Vec<f32> = (0..513).map(|i| i as f32 * 0.25).collect();
    let b_host: Vec<f32> = (0..513).map(|i| 50.0 - i as f32).collect();
    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(a_host.len()).unwrap();

    binary_elementwise_into::<SubOp, f32>(&dev, &a, &b, &out, width).unwrap();
    let mut got = vec![0.0f32; a_host.len()];
    dev.download(&out, &mut got).unwrap();
    let expected: Vec<f32> = a_host.iter().zip(&b_host).map(|(x, y)| x - y).collect();
    assert_eq!(got, expected);

    unary_elementwise_into::<NegOp, f32>(&dev, &a, &out, width).unwrap();
    dev.download(&out, &mut got).unwrap();
    let expected: Vec<f32> = a_host.iter().map(|x| -x).collect();
    assert_eq!(got, expected);

    scalar_elementwise_into::<AddOp, f32>(&dev, &a, 7.5, &out, width).unwrap();
    dev.download(&out, &mut got).unwrap();
    let expected: Vec<f32> = a_host.iter().map(|x| x + 7.5).collect();
    assert_eq!(got, expected);

    let short = dev.alloc_zeroed::<f32>(a_host.len() - 1).unwrap();
    assert_length_mismatch(
        binary_elementwise_into::<AddOp, f32>(&dev, &a, &b, &short, width),
        short.len(),
        a.len(),
    );
    assert_length_mismatch(
        unary_elementwise_into::<NegOp, f32>(&dev, &a, &short, width),
        short.len(),
        a.len(),
    );
    assert_length_mismatch(
        scalar_elementwise_into::<AddOp, f32>(&dev, &a, 1.0, &short, width),
        short.len(),
        a.len(),
    );
}

#[test]
fn elementwise_into_rejects_output_input_aliasing() {
    let Some(dev) = device("elementwise_into_rejects_output_input_aliasing") else {
        return;
    };
    let width = BlockWidth::new(128).unwrap();
    let a = dev.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let b = dev.upload(&[4.0f32, 5.0, 6.0]).unwrap();

    assert_elementwise_alias_rejected(binary_elementwise_into::<AddOp, f32>(
        &dev, &a, &b, &a, width,
    ));
    assert_elementwise_alias_rejected(binary_elementwise_into::<AddOp, f32>(
        &dev, &a, &b, &b, width,
    ));
    assert_elementwise_alias_rejected(unary_elementwise_into::<NegOp, f32>(&dev, &a, &a, width));
    assert_elementwise_alias_rejected(scalar_elementwise_into::<AddOp, f32>(
        &dev, &a, 1.0, &a, width,
    ));
}

#[test]
fn elementwise_unary_matches_cpu_reference() {
    let Some(dev) = device("elementwise_unary_matches_cpu_reference") else {
        return;
    };
    let host = vec![-4.0f32, -1.0, 0.0, 2.0, 16.0];
    let a = dev.upload(&host).unwrap();

    // SqrtOp
    let out_sqrt = unary_elementwise::<SqrtOp, f32>(&dev, &a).unwrap();
    let mut got_sqrt = vec![0.0f32; host.len()];
    dev.download(&out_sqrt, &mut got_sqrt).unwrap();
    assert!(got_sqrt[0].is_nan());
    assert!(got_sqrt[1].is_nan());
    assert_eq!(got_sqrt[2], 0.0f32);
    assert_eq!(got_sqrt[3], std::f32::consts::SQRT_2);
    assert_eq!(got_sqrt[4], 4.0f32);

    // AbsOp
    let out_abs = unary_elementwise::<AbsOp, f32>(&dev, &a).unwrap();
    let mut got_abs = vec![0.0f32; host.len()];
    dev.download(&out_abs, &mut got_abs).unwrap();
    assert_eq!(got_abs, vec![4.0f32, 1.0, 0.0, 2.0, 16.0]);

    // NegOp
    let out_neg = unary_elementwise::<NegOp, f32>(&dev, &a).unwrap();
    let mut got_neg = vec![0.0f32; host.len()];
    dev.download(&out_neg, &mut got_neg).unwrap();
    assert_eq!(got_neg, vec![4.0f32, 1.0, 0.0, -2.0, -16.0]);

    // ExpOp
    let out_exp = unary_elementwise::<ExpOp, f32>(&dev, &a).unwrap();
    let mut got_exp = vec![0.0f32; host.len()];
    dev.download(&out_exp, &mut got_exp).unwrap();
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

    // RecipOp
    let host_recip = vec![1.0f32, 2.0, 4.0, 8.0];
    let b = dev.upload(&host_recip).unwrap();
    let out_recip = unary_elementwise::<RecipOp, f32>(&dev, &b).unwrap();
    let mut got_recip = vec![0.0f32; host_recip.len()];
    dev.download(&out_recip, &mut got_recip).unwrap();
    assert_eq!(got_recip, vec![1.0f32, 0.5, 0.25, 0.125]);
}

#[test]
fn elementwise_activation_markers_match_cpu_reference() {
    let Some(device) = device("elementwise_activation_markers_match_cpu_reference") else {
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

#[test]
fn elementwise_scalar_matches_cpu_reference() {
    let Some(dev) = device("elementwise_scalar_matches_cpu_reference") else {
        return;
    };
    let host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let a = dev.upload(&host).unwrap();

    let out_add = scalar_elementwise::<AddOp, f32>(&dev, &a, 10.0).unwrap();
    let mut got_add = vec![0.0f32; host.len()];
    dev.download(&out_add, &mut got_add).unwrap();
    assert_eq!(got_add, vec![11.0f32, 12.0, 13.0, 14.0, 15.0]);

    let out_mul = scalar_elementwise::<MulOp, f32>(&dev, &a, 3.0).unwrap();
    let mut got_mul = vec![0.0f32; host.len()];
    dev.download(&out_mul, &mut got_mul).unwrap();
    assert_eq!(got_mul, vec![3.0f32, 6.0, 9.0, 12.0, 15.0]);
}

#[test]
fn elementwise_header_backed_scalars_match_cpu_reference() {
    let Some(dev) = device("elementwise_header_backed_scalars_match_cpu_reference") else {
        return;
    };

    let host_f16 = [
        eunomia::F16::from_f32(1.0),
        eunomia::F16::from_f32(2.0),
        eunomia::F16::from_f32(4.0),
    ];
    let input_f16 = dev.upload(&host_f16).unwrap();
    let output_f16 =
        scalar_elementwise::<AddOp, eunomia::F16>(&dev, &input_f16, eunomia::F16::from_f32(3.0))
            .unwrap();
    let mut actual_f16 = [eunomia::F16::ZERO; 3];
    dev.download(&output_f16, &mut actual_f16).unwrap();
    assert_eq!(actual_f16.map(eunomia::F16::to_f32), [4.0, 5.0, 7.0]);

    let host_bf16 = [
        eunomia::Bf16::from_f32(1.0),
        eunomia::Bf16::from_f32(2.0),
        eunomia::Bf16::from_f32(4.0),
    ];
    let input_bf16 = dev.upload(&host_bf16).unwrap();
    let output_bf16 =
        scalar_elementwise::<AddOp, eunomia::Bf16>(&dev, &input_bf16, eunomia::Bf16::from_f32(3.0))
            .unwrap();
    let mut actual_bf16 = [eunomia::Bf16::ZERO; 3];
    dev.download(&output_bf16, &mut actual_bf16).unwrap();
    assert_eq!(actual_bf16.map(eunomia::Bf16::to_f32), [4.0, 5.0, 7.0]);
}
