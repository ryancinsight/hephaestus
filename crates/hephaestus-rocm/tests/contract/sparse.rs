use super::*;

#[test]
fn sparse_csr_products_match_cpu_values_and_reject_wrong_shapes() {
    let Some(device) = device("sparse_csr_products_match_cpu_values_and_reject_wrong_shapes")
    else {
        return;
    };

    let cpu_csr = leto_ops::CsrMatrix::from_parts(
        vec![2.0_f32, -1.0, 3.0, 4.0],
        vec![0, 2, 1, 2],
        vec![0, 2, 3, 4],
        3,
        3,
    )
    .expect("valid CSR contract fixture");
    let gpu_csr = GpuCsrMatrix::from_cpu(&device, &cpu_csr).expect("HIP CSR upload");
    assert_eq!(gpu_csr.shape(), (3, 3));
    assert_eq!(gpu_csr.nnz(), 4);
    assert_eq!(gpu_csr.to_cpu(&device).expect("HIP CSR download"), cpu_csr);

    let zero_nnz_csr =
        leto_ops::CsrMatrix::from_parts(Vec::<f32>::new(), Vec::new(), vec![0, 0, 0], 2, 3)
            .expect("valid zero-nnz CSR fixture");
    let zero_nnz_gpu =
        GpuCsrMatrix::from_cpu(&device, &zero_nnz_csr).expect("zero-nnz HIP CSR upload");
    assert_eq!(
        zero_nnz_gpu
            .to_cpu(&device)
            .expect("zero-nnz HIP CSR download"),
        zero_nnz_csr
    );

    let x = device
        .upload(&[1.0_f32, 2.0, 3.0])
        .expect("SpMV input upload");
    let y = spmv(&device, &gpu_csr, &x).expect("HIP SpMV");
    let mut y_values = [0.0_f32; 3];
    device.download(&y, &mut y_values).expect("SpMV download");
    assert_eq!(y_values, [-1.0, 6.0, 12.0]);

    let b = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("SpMM input upload");
    let b_layout = Layout::c_contiguous([3, 2]).expect("SpMM input layout");
    let b_operand = StridedOperand {
        buffer: &b,
        layout: &b_layout,
    };
    let c = spmm(&device, &gpu_csr, b_operand).expect("HIP SpMM");
    let mut c_values = [0.0_f32; 6];
    device.download(&c, &mut c_values).expect("SpMM download");
    assert_eq!(c_values, [-3.0, -2.0, 9.0, 12.0, 20.0, 24.0]);

    let many = spmv_many(&device, &gpu_csr, b_operand).expect("HIP batched SpMV");
    let mut many_values = [0.0_f32; 6];
    device
        .download(&many, &mut many_values)
        .expect("batched SpMV download");
    assert_eq!(many_values, c_values);

    let mut c_reused = device.upload(&[99.0_f32; 6]).expect("SpMM output upload");
    spmm_into(&device, &gpu_csr, b_operand, &mut c_reused).expect("HIP SpMM into");
    let mut reused_values = [0.0_f32; 6];
    device
        .download(&c_reused, &mut reused_values)
        .expect("reused SpMM download");
    assert_eq!(reused_values, c_values);

    let mut many_reused = device
        .upload(&[88.0_f32; 6])
        .expect("batched output upload");
    spmv_many_into(&device, &gpu_csr, b_operand, &mut many_reused).expect("HIP batched SpMV into");
    let mut many_reused_values = [0.0_f32; 6];
    device
        .download(&many_reused, &mut many_reused_values)
        .expect("reused batched SpMV download");
    assert_eq!(many_reused_values, c_values);

    let empty_row_csr = leto_ops::CsrMatrix::from_parts(
        vec![2.0_f32, -1.0, 4.0],
        vec![0, 2, 2],
        vec![0, 2, 2, 3],
        3,
        3,
    )
    .expect("valid empty-row CSR fixture");
    let empty_row_gpu =
        GpuCsrMatrix::from_cpu(&device, &empty_row_csr).expect("empty-row HIP CSR upload");
    let empty_row_y = spmv(&device, &empty_row_gpu, &x).expect("empty-row HIP SpMV");
    let mut empty_row_y_values = [0.0_f32; 3];
    device
        .download(&empty_row_y, &mut empty_row_y_values)
        .expect("empty-row SpMV download");
    assert_eq!(empty_row_y_values, [-1.0, 0.0, 12.0]);

    let empty_row_c = spmm(&device, &empty_row_gpu, b_operand).expect("empty-row HIP SpMM");
    let mut empty_row_c_values = [0.0_f32; 6];
    device
        .download(&empty_row_c, &mut empty_row_c_values)
        .expect("empty-row SpMM download");
    assert_eq!(empty_row_c_values, [-3.0, -2.0, 0.0, 0.0, 20.0, 24.0]);

    let wrong_x = device.upload(&[1.0_f32, 2.0]).expect("wrong SpMV upload");
    assert_length_mismatch(spmv(&device, &gpu_csr, &wrong_x), 3, 2);
}

#[test]
fn prepared_sparse_dispatch_matches_reference() {
    let Some(device) = device("prepared_sparse_dispatch_matches_reference") else {
        return;
    };
    use hephaestus_rocm::{
        PreparedSparseDispatch, prepare_spmm, prepare_spmv, prepare_spmv_many,
        submit_prepared_sparse_batch,
    };

    let cpu_csr = leto_ops::CsrMatrix::from_parts(
        vec![2.0_f32, -1.0, 3.0, 4.0],
        vec![0, 2, 1, 2],
        vec![0, 2, 3, 4],
        3,
        3,
    )
    .expect("valid CSR contract fixture");
    let gpu_csr = GpuCsrMatrix::from_cpu(&device, &cpu_csr).expect("HIP CSR upload");
    let x = device
        .upload(&[1.0_f32, 2.0, 3.0])
        .expect("SpMV input upload");
    let y = device.upload(&[77.0_f32; 3]).expect("SpMV output upload");
    let prepared_spmv = prepare_spmv(&device, &gpu_csr, &x, &y).expect("prepare HIP SpMV");
    prepared_spmv.dispatch().expect("dispatch HIP SpMV");
    prepared_spmv.dispatch().expect("repeat HIP SpMV");

    let b = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("SpMM input upload");
    let b_layout = Layout::try_new([3, 2], [1, 3], 0).expect("valid test layout");
    let b_operand = StridedOperand {
        buffer: &b,
        layout: &b_layout,
    };
    let mut c = device.upload(&[88.0_f32; 6]).expect("SpMM output upload");
    let prepared_spmm =
        prepare_spmm(&device, &gpu_csr, &b_operand, &mut c).expect("prepare HIP SpMM");
    let mut many = device
        .upload(&[99.0_f32; 6])
        .expect("batched output upload");
    let prepared_many = prepare_spmv_many(&device, &gpu_csr, &b_operand, &mut many)
        .expect("prepare HIP batched SpMV");
    submit_prepared_sparse_batch(&[
        PreparedSparseDispatch::Spmv(&prepared_spmv),
        PreparedSparseDispatch::Spmm(&prepared_spmm),
        PreparedSparseDispatch::Spmm(&prepared_many),
    ])
    .expect("submit HIP sparse batch");

    let mut got_y = [0.0_f32; 3];
    device.download(&y, &mut got_y).expect("SpMV download");
    assert_eq!(got_y, [-1.0, 6.0, 12.0]);
    let mut got_c = [0.0_f32; 6];
    device.download(&c, &mut got_c).expect("SpMM download");
    assert_eq!(got_c, [-1.0, 2.0, 6.0, 15.0, 12.0, 24.0]);
    let mut got_many = [0.0_f32; 6];
    device
        .download(&many, &mut got_many)
        .expect("batched SpMV download");
    assert_eq!(got_many, got_c);

    let wrong_x = device.upload(&[1.0_f32, 2.0]).expect("wrong SpMV upload");
    let wrong_output = device
        .upload(&[0.0_f32; 3])
        .expect("wrong SpMV output upload");
    match prepare_spmv(&device, &gpu_csr, &wrong_x, &wrong_output) {
        Err(HephaestusError::LengthMismatch {
            host_len: 3,
            device_len: 2,
        }) => {}
        Err(error) => panic!("expected SpMV length rejection, got {error:?}"),
        Ok(_) => panic!("expected SpMV length rejection, got success"),
    }
    let bad_layout = Layout::try_new([3, 2], [2, 1], 5).expect("valid test layout");
    let bad_operand = StridedOperand {
        buffer: &b,
        layout: &bad_layout,
    };
    let mut bad_output = device
        .upload(&[0.0_f32; 6])
        .expect("bad SpMM output upload");
    match prepare_spmm(&device, &gpu_csr, &bad_operand, &mut bad_output) {
        Err(HephaestusError::DispatchFailed { message }) => {
            assert!(
                message.contains("layout rejected"),
                "unexpected error: {message}"
            );
        }
        Err(error) => panic!("expected invalid layout rejection, got {error:?}"),
        Ok(_) => panic!("expected invalid layout rejection, got success"),
    }
}
