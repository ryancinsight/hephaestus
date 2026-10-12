use super::*;

#[test]
fn seeded_random_initializers_match_determinism_and_distribution_contracts() {
    let Some(device) =
        device("seeded_random_initializers_match_determinism_and_distribution_contracts")
    else {
        return;
    };

    let shape = [1000];
    let low = -2.0_f32;
    let high = 5.0_f32;
    let uniform = uniform_with_seed(&device, shape, low, high, 42).unwrap();
    let mut got_uniform = vec![0.0_f32; 1000];
    device.download(&uniform, &mut got_uniform).unwrap();

    let uniform_again = uniform_with_seed(&device, shape, low, high, 42).unwrap();
    let mut got_uniform_again = vec![0.0_f32; 1000];
    device
        .download(&uniform_again, &mut got_uniform_again)
        .unwrap();
    assert_eq!(got_uniform, got_uniform_again);
    assert!(
        got_uniform
            .iter()
            .all(|&value| value >= low && value < high),
        "uniform samples must stay in the half-open interval"
    );

    let normal = normal_with_seed(&device, shape, 0.0_f32, 1.0_f32, 42).unwrap();
    let mut got_normal = vec![0.0_f32; 1000];
    device.download(&normal, &mut got_normal).unwrap();
    assert!(
        got_normal.iter().any(|&value| value != 0.0),
        "normal samples must contain a nonzero value"
    );
}

#[test]
fn linalg_matmul_matches_cpu_reference() {
    let Some(d) = device("linalg_matmul_matches_cpu_reference") else {
        return;
    };
    let a = d.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let b = d.upload(&[5.0f32, 6.0, 7.0, 8.0]).unwrap();
    let out = matmul(
        &d,
        StridedOperand {
            buffer: &a,
            layout: &Layout::c_contiguous([2, 2]).unwrap(),
        },
        StridedOperand {
            buffer: &b,
            layout: &Layout::c_contiguous([2, 2]).unwrap(),
        },
    )
    .unwrap();
    let mut host_out = [0.0f32; 4];
    d.download(&out, &mut host_out).unwrap();
    assert_eq!(host_out, [19.0, 22.0, 43.0, 50.0,]);
}
