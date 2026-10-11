use super::*;

#[test]
fn linalg_fused_map_reductions_accept_reversed_views() {
    let Some(dev) = device("linalg_fused_map_reductions_accept_reversed_views") else {
        return;
    };

    let values = dev.upload(&[-1.0_f32, 2.0, -3.0, 4.0]).unwrap();
    let weights = dev.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let reversed = Layout::try_new([4], [-1], 3).expect("valid test layout");
    let contiguous = Layout::c_contiguous([4]).unwrap();

    let reversed_operand = StridedOperand {
        buffer: &values,
        layout: &reversed,
    };
    let weights_operand = StridedOperand {
        buffer: &weights,
        layout: &contiguous,
    };

    let dot_buf = dot(&dev, reversed_operand, weights_operand).unwrap();
    let l1_buf = norm_l1(&dev, reversed_operand).unwrap();
    let l2_buf = norm_l2(&dev, reversed_operand).unwrap();
    let max_buf = norm_max(&dev, reversed_operand).unwrap();

    let mut got = [0.0_f32; 1];
    dev.download(&dot_buf, &mut got).unwrap();
    assert_eq!(got, [0.0]);
    dev.download(&l1_buf, &mut got).unwrap();
    assert_eq!(got, [10.0]);
    dev.download(&l2_buf, &mut got).unwrap();
    let expected_l2 = 30.0_f32.sqrt();
    // The four-term tree is exact for this integer square sum; the tolerance
    // allows two f32 ulps for the final square-root rounding.
    assert!((got[0] - expected_l2).abs() <= 2.0 * f32::EPSILON * expected_l2);
    dev.download(&max_buf, &mut got).unwrap();
    assert_eq!(got, [4.0]);
}
