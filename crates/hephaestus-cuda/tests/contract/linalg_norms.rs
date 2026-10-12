use super::*;

#[test]
fn linalg_norms_match_cpu_reference() {
    let Some(dev) = device("linalg_norms_match_cpu_reference") else {
        return;
    };

    let a_host = vec![-1.0f32, 2.0, -3.0, 4.0];
    let a = dev.upload(&a_host).unwrap();
    let a_layout = Layout::c_contiguous([4]).unwrap();
    let operand = StridedOperand {
        buffer: &a,
        layout: &a_layout,
    };

    // L1
    let l1_buf = norm_l1(&dev, operand).unwrap();
    let mut got_l1 = [0.0f32; 1];
    dev.download(&l1_buf, &mut got_l1).unwrap();
    assert_eq!(got_l1[0], 10.0);

    // L2
    let l2_buf = norm_l2(&dev, operand).unwrap();
    let mut got_l2 = [0.0f32; 1];
    dev.download(&l2_buf, &mut got_l2).unwrap();
    let expected_l2 = 30.0f32.sqrt();
    assert!((got_l2[0] - expected_l2).abs() <= 1e-5);

    // Max
    let max_buf = norm_max(&dev, operand).unwrap();
    let mut got_max = [0.0f32; 1];
    dev.download(&max_buf, &mut got_max).unwrap();
    assert_eq!(got_max[0], 4.0);
}

#[test]
fn linalg_rank8_norms_match_cpu_reference() {
    let Some(dev) = device("linalg_rank8_norms_match_cpu_reference") else {
        return;
    };

    let shape = [2usize; 8];
    let a_host: Vec<f32> = (1..=256).map(|value| value as f32).collect();
    let a_layout = Layout::try_new(shape, [1isize, 2, 4, 8, 16, 32, 64, 128], 0)
        .expect("valid rank-eight layout");
    let a = dev.upload(&a_host).unwrap();
    let operand = StridedOperand {
        buffer: &a,
        layout: &a_layout,
    };

    let l1 = norm_l1(&dev, operand).unwrap();
    let max = norm_max(&dev, operand).unwrap();
    let mut got_l1 = [0.0f32; 1];
    let mut got_max = [0.0f32; 1];
    dev.download(&l1, &mut got_l1).unwrap();
    dev.download(&max, &mut got_max).unwrap();

    let expected_l1: f32 = a_host.iter().copied().sum();
    assert_eq!(got_l1, [expected_l1]);
    assert_eq!(got_max, [256.0]);
}

#[test]
fn test_cuda_uniform_and_normal_with_seed() {
    let Some(dev) = device("test_cuda_uniform_and_normal_with_seed") else {
        return;
    };
    use hephaestus_cuda::{normal_with_seed, uniform_with_seed};

    let shape = [1000];
    let low = -2.0f32;
    let high = 5.0f32;
    let u_buf = uniform_with_seed(&dev, shape, low, high, 42).unwrap();
    let mut got_u = vec![0.0f32; 1000];
    dev.download(&u_buf, &mut got_u).unwrap();

    // Verify determinism & range
    let u_buf_2 = uniform_with_seed(&dev, shape, low, high, 42).unwrap();
    let mut got_u_2 = vec![0.0f32; 1000];
    dev.download(&u_buf_2, &mut got_u_2).unwrap();
    assert_eq!(got_u, got_u_2);

    for &val in &got_u {
        assert!(val >= low && val < high, "value out of bounds: {val}");
    }

    let n_buf = normal_with_seed(&dev, shape, 0.0f32, 1.0f32, 42).unwrap();
    let mut got_n = vec![0.0f32; 1000];
    dev.download(&n_buf, &mut got_n).unwrap();
    assert!(got_n.iter().any(|&val| val != 0.0));
}
