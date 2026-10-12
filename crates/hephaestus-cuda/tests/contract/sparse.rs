use super::*;

#[test]
fn test_cuda_sparse_matrix_spmv_spmm() {
    let Some(dev) = device("test_cuda_sparse_matrix_spmv_spmm") else {
        return;
    };
    use hephaestus_cuda::{GpuCsrMatrix, StridedOperand, spmm, spmv};
    use leto::Layout;

    // Create a 3x3 diagonal-ish matrix:
    // [ 2.0  0.0 -1.0 ]
    // [ 0.0  3.0  0.0 ]
    // [ 0.0  0.0  4.0 ]
    let dense_host = vec![2.0f32, 0.0, -1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
    let layout = Layout::c_contiguous([3, 3]).unwrap();
    let cpu_csr = leto_ops::CsrMatrix::from_dense(&leto::ArrayView2::new(layout, &dense_host));

    let gpu_csr = GpuCsrMatrix::from_cpu(&dev, &cpu_csr).unwrap();
    assert_eq!(gpu_csr.shape(), (3, 3));
    assert_eq!(gpu_csr.nnz(), 4);

    // Round-trip back to CPU
    let cpu_csr_2 = gpu_csr.to_cpu(&dev).unwrap();
    assert_eq!(cpu_csr, cpu_csr_2);

    // SpMV: y = A * x, x = [1.0, 2.0, 3.0]
    // Expected y = [ 2*1 - 3, 3*2, 4*3 ] = [ -1.0, 6.0, 12.0 ]
    let x_host = vec![1.0f32, 2.0, 3.0];
    let x_buf = dev.upload(&x_host).unwrap();
    let y_buf = spmv(&dev, &gpu_csr, &x_buf).unwrap();
    let mut got_y = vec![0.0f32; 3];
    dev.download(&y_buf, &mut got_y).unwrap();
    assert_close_slice(&got_y, &[-1.0, 6.0, 12.0], 1.0e-4, 0.0);

    // SpMM: C = A * B, B = [ 1.0  2.0 ]
    //                      [ 3.0  4.0 ]
    //                      [ 5.0  6.0 ]
    // Expected C = [ 2*1 - 5, 2*2 - 6 ] = [ -3.0, -2.0 ]
    //              [ 3*3,     3*4     ]   [  9.0, 12.0 ]
    //              [ 4*5,     4*6     ]   [ 20.0, 24.0 ]
    let b_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_buf = dev.upload(&b_host).unwrap();
    let b_layout = Layout::c_contiguous([3, 2]).unwrap();
    let b_op = StridedOperand {
        buffer: &b_buf,
        layout: &b_layout,
    };
    let c_buf = spmm(&dev, &gpu_csr, &b_op).unwrap();
    let mut got_c = vec![0.0f32; 6];
    dev.download(&c_buf, &mut got_c).unwrap();
    assert_close_slice(&got_c, &[-3.0, -2.0, 9.0, 12.0, 20.0, 24.0], 1.0e-4, 0.0);

    let empty_row_csr = leto_ops::CsrMatrix::from_parts(
        vec![2.0_f32, -1.0, 4.0],
        vec![0, 2, 2],
        vec![0, 2, 2, 3],
        3,
        3,
    )
    .unwrap();
    let empty_row_gpu = GpuCsrMatrix::from_cpu(&dev, &empty_row_csr).unwrap();
    let empty_row_y = spmv(&dev, &empty_row_gpu, &x_buf).unwrap();
    let mut got_empty_row_y = [0.0_f32; 3];
    dev.download(&empty_row_y, &mut got_empty_row_y).unwrap();
    assert_eq!(got_empty_row_y, [-1.0, 0.0, 12.0]);

    let empty_row_c = spmm(&dev, &empty_row_gpu, &b_op).unwrap();
    let mut got_empty_row_c = [0.0_f32; 6];
    dev.download(&empty_row_c, &mut got_empty_row_c).unwrap();
    assert_eq!(got_empty_row_c, [-3.0, -2.0, 0.0, 0.0, 20.0, 24.0]);
}

#[test]
fn test_cuda_prepared_sparse_dispatch_matches_reference() {
    let Some(dev) = device("test_cuda_prepared_sparse_dispatch_matches_reference") else {
        return;
    };
    use hephaestus_cuda::{
        GpuCsrMatrix, PreparedSparseDispatch, StridedOperand, prepare_spmm, prepare_spmv,
        prepare_spmv_many, submit_prepared_sparse_batch,
    };

    let cpu_csr = leto_ops::CsrMatrix::from_parts(
        vec![2.0_f32, -1.0, 3.0, 4.0],
        vec![0, 2, 1, 2],
        vec![0, 2, 3, 4],
        3,
        3,
    )
    .unwrap();
    let gpu_csr = GpuCsrMatrix::from_cpu(&dev, &cpu_csr).unwrap();
    let x = dev.upload(&[1.0_f32, 2.0, 3.0]).unwrap();
    let y = dev.upload(&[77.0_f32; 3]).unwrap();
    let prepared_spmv = prepare_spmv(&dev, &gpu_csr, &x, &y).unwrap();
    prepared_spmv.dispatch().unwrap();
    prepared_spmv.dispatch().unwrap();

    let b = dev.upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b_layout = Layout::try_new([3, 2], [1, 3], 0).expect("valid test layout");
    let b_operand = StridedOperand {
        buffer: &b,
        layout: &b_layout,
    };
    let mut c = dev.upload(&[88.0_f32; 6]).unwrap();
    let prepared_spmm = prepare_spmm(&dev, &gpu_csr, &b_operand, &mut c).unwrap();
    let mut many = dev.upload(&[99.0_f32; 6]).unwrap();
    let prepared_many = prepare_spmv_many(&dev, &gpu_csr, &b_operand, &mut many).unwrap();
    submit_prepared_sparse_batch(&[
        PreparedSparseDispatch::Spmv(&prepared_spmv),
        PreparedSparseDispatch::Spmm(&prepared_spmm),
        PreparedSparseDispatch::Spmm(&prepared_many),
    ])
    .unwrap();

    let mut got_y = [0.0_f32; 3];
    dev.download(&y, &mut got_y).unwrap();
    assert_close_slice(&got_y, &[-1.0, 6.0, 12.0], 1.0e-4, 0.0);
    let mut got_c = [0.0_f32; 6];
    dev.download(&c, &mut got_c).unwrap();
    assert_close_slice(&got_c, &[-1.0, 2.0, 6.0, 15.0, 12.0, 24.0], 1.0e-4, 0.0);
    let mut got_many = [0.0_f32; 6];
    dev.download(&many, &mut got_many).unwrap();
    assert_eq!(got_many, got_c);

    let wrong_x = dev.upload(&[1.0_f32, 2.0]).unwrap();
    let wrong_output = dev.upload(&[0.0_f32; 3]).unwrap();
    assert_length_mismatch(prepare_spmv(&dev, &gpu_csr, &wrong_x, &wrong_output), 3, 2);
    let bad_layout = Layout::try_new([3, 2], [2, 1], 5).expect("valid test layout");
    let bad_operand = StridedOperand {
        buffer: &b,
        layout: &bad_layout,
    };
    let mut bad_output = dev.upload(&[0.0_f32; 6]).unwrap();
    match prepare_spmm(&dev, &gpu_csr, &bad_operand, &mut bad_output) {
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
