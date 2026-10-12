use super::*;

#[test]
fn elementwise_add_matches_cpu_reference() {
    let Some(d) = device("elementwise_add_matches_cpu_reference") else {
        return;
    };
    let a = d.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let b = d.upload(&[4.0f32, 5.0, 6.0]).unwrap();
    let out = binary_elementwise::<AddOp, f32>(&d, &a, &b).unwrap();
    let mut host_out = [0.0f32; 3];
    d.download(&out, &mut host_out).unwrap();
    assert_eq!(host_out, [5.0, 7.0, 9.0]);
}

#[test]
fn elementwise_unary_matches_cpu_reference() {
    let Some(d) = device("elementwise_unary_matches_cpu_reference") else {
        return;
    };
    let a = d.upload(&[4.0f32, 9.0, 16.0]).unwrap();
    let out = unary_elementwise::<SqrtOp, f32>(&d, &a).unwrap();
    let mut host_out = [0.0f32; 3];
    d.download(&out, &mut host_out).unwrap();
    assert_eq!(host_out, [2.0, 3.0, 4.0]);
}

#[test]
fn elementwise_exp_neg_matches_cpu_and_composed_references() {
    let Some(d) = device("elementwise_exp_neg_matches_cpu_and_composed_references") else {
        return;
    };
    let host = [-4.0f32, -1.0, 0.0, 2.0, 16.0];
    let input = d.upload(&host).unwrap();
    let fused = unary_elementwise::<ExpNegOp, f32>(&d, &input).unwrap();
    let negated = unary_elementwise::<NegOp, f32>(&d, &input).unwrap();
    let composed = unary_elementwise::<ExpOp, f32>(&d, &negated).unwrap();
    let mut fused_host = [0.0f32; 5];
    let mut composed_host = [0.0f32; 5];
    d.download(&fused, &mut fused_host).unwrap();
    d.download(&composed, &mut composed_host).unwrap();

    for (index, &x) in host.iter().enumerate() {
        let expected = (-x).exp();
        let tolerance = 1e-5 * expected.abs().max(1.0);
        assert!(
            (fused_host[index] - expected).abs() < tolerance,
            "fused ExpNeg mismatch at index {index}: got {}, expected {expected}, tolerance {tolerance}",
            fused_host[index]
        );
        assert!(
            (fused_host[index] - composed_host[index]).abs() < tolerance,
            "fused/composed ExpNeg mismatch at index {index}: fused {}, composed {}, tolerance {tolerance}",
            fused_host[index],
            composed_host[index]
        );
    }
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
    let Some(d) = device("elementwise_scalar_matches_cpu_reference") else {
        return;
    };
    let a = d.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let out = scalar_elementwise::<MulOp, f32>(&d, &a, 3.0).unwrap();
    let mut host_out = [0.0f32; 3];
    d.download(&out, &mut host_out).unwrap();
    assert_eq!(host_out, [3.0, 6.0, 9.0]);
}

#[test]
fn elementwise_into_rejects_output_input_aliasing() {
    let Some(d) = device("elementwise_into_rejects_output_input_aliasing") else {
        return;
    };
    let a = d.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    assert_elementwise_alias_rejected(unary_elementwise_into::<NegOp, f32>(
        &d,
        &a,
        &a,
        BlockWidth::DEFAULT,
    ));
}
