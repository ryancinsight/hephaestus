use super::*;

#[test]
fn map_reductions_match_cpu_values_for_strided_views_and_reject_invalid_shapes() {
    let Some(device) =
        device("map_reductions_match_cpu_values_for_strided_views_and_reject_invalid_shapes")
    else {
        return;
    };

    let matrix_values = [-3_i32, 4, -5, 6, -7, 8];
    let matrix_buffer = device
        .upload(&matrix_values)
        .expect("HIP norm input upload");
    let matrix_layout = Layout::c_contiguous([2, 3]).expect("norm matrix layout");
    let matrix = StridedOperand {
        buffer: &matrix_buffer,
        layout: &matrix_layout,
    };

    let l1 = norm_l1(&device, matrix).expect("HIP L1 norm");
    let mut l1_value = [0_i32];
    device
        .download(&l1, &mut l1_value)
        .expect("HIP L1 download");
    assert_eq!(l1_value, [33]);

    let max = norm_max(&device, matrix).expect("HIP max norm");
    let mut max_value = [0_i32];
    device
        .download(&max, &mut max_value)
        .expect("HIP max norm download");
    assert_eq!(max_value, [8]);

    let dot_left_values = [99_i32, 1, 2, 3, 4, 5];
    let dot_right_values = [6_i32, 7, 8, 9, 10];
    let dot_left_buffer = device
        .upload(&dot_left_values)
        .expect("HIP strided dot lhs upload");
    let dot_right_buffer = device
        .upload(&dot_right_values)
        .expect("HIP strided dot rhs upload");
    let dot_left_layout = Layout::try_new([5], [1], 1).expect("valid test layout");
    let dot_right_layout = Layout::c_contiguous([5]).expect("dot rhs layout");
    let dot_value_buffer = dot(
        &device,
        StridedOperand {
            buffer: &dot_left_buffer,
            layout: &dot_left_layout,
        },
        StridedOperand {
            buffer: &dot_right_buffer,
            layout: &dot_right_layout,
        },
    )
    .expect("HIP strided dot");
    let mut dot_value = [0_i32];
    device
        .download(&dot_value_buffer, &mut dot_value)
        .expect("HIP dot download");
    assert_eq!(dot_value, [130]);

    let trace_values = [1_i32, 2, 3, 4, 5, 6, 7, 8, 9];
    let trace_buffer = device.upload(&trace_values).expect("HIP trace upload");
    let trace_layout = Layout::c_contiguous([3, 3]).expect("trace layout");
    let trace_value_buffer = trace(
        &device,
        StridedOperand {
            buffer: &trace_buffer,
            layout: &trace_layout,
        },
    )
    .expect("HIP trace");
    let mut trace_value = [0_i32];
    device
        .download(&trace_value_buffer, &mut trace_value)
        .expect("HIP trace download");
    assert_eq!(trace_value, [15]);

    let l2_values = [3.0_f32, 4.0];
    let l2_buffer = device.upload(&l2_values).expect("HIP L2 input upload");
    let l2_layout = Layout::c_contiguous([2]).expect("L2 layout");
    let l2 = norm_l2(
        &device,
        StridedOperand {
            buffer: &l2_buffer,
            layout: &l2_layout,
        },
    )
    .expect("HIP L2 norm");
    let mut l2_value = [0.0_f32];
    device
        .download(&l2, &mut l2_value)
        .expect("HIP L2 download");
    assert_eq!(l2_value, [5.0]);

    let wrong_dot_layout = Layout::c_contiguous([4]).expect("wrong dot layout");
    assert!(matches!(
        dot(
            &device,
            StridedOperand {
                buffer: &dot_left_buffer,
                layout: &dot_left_layout,
            },
            StridedOperand {
                buffer: &dot_right_buffer,
                layout: &wrong_dot_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("dot product shape mismatch")
    ));

    let rectangular_layout = Layout::c_contiguous([2, 3]).expect("rectangular trace layout");
    assert!(matches!(
        trace(
            &device,
            StridedOperand {
                buffer: &matrix_buffer,
                layout: &rectangular_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("trace requires a square matrix")
    ));
}
