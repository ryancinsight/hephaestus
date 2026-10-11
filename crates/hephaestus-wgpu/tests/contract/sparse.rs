use super::*;

#[cfg(feature = "sparse")]
pub(crate) fn test_wgpu_sparse_matrix_spmv_spmm() {
    let Some(device) = device_or_skip() else {
        return;
    };
    use hephaestus_wgpu::{
        GpuCsrMatrix, PreparedSparseDispatch, StridedOperand, prepare_spmm, prepare_spmv,
        prepare_spmv_many, spmm, spmm_into, spmv, spmv_into, spmv_many, spmv_many_into,
        submit_prepared_sparse_batch,
    };
    use leto::Layout;

    // Create a 3x3 diagonal-ish matrix:
    // [ 2.0  0.0 -1.0 ]
    // [ 0.0  3.0  0.0 ]
    // [ 0.0  0.0  4.0 ]
    let dense_host = vec![2.0f32, 0.0, -1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
    let layout = Layout::c_contiguous([3, 3]).unwrap();
    let cpu_csr = leto_ops::CsrMatrix::from_dense(&leto::ArrayView2::new(layout, &dense_host));

    let gpu_csr = GpuCsrMatrix::from_cpu(&device, &cpu_csr).unwrap();
    assert_eq!(gpu_csr.shape(), (3, 3));
    assert_eq!(gpu_csr.nnz(), 4);

    // Round-trip back to CPU
    let cpu_csr_2 = gpu_csr.to_cpu(&device).unwrap();
    assert_eq!(cpu_csr, cpu_csr_2);

    // SpMV: y = A * x, x = [1.0, 2.0, 3.0]
    // Expected y = [ 2*1 - 3, 3*2, 4*3 ] = [ -1.0, 6.0, 12.0 ]
    let x_host = vec![1.0f32, 2.0, 3.0];
    let x_buf = device.upload(&x_host).unwrap();
    let y_buf = spmv(&device, &gpu_csr, &x_buf).unwrap();
    let mut got_y = vec![0.0f32; 3];
    device.download(&y_buf, &mut got_y).unwrap();
    assert_close_slice(&got_y, &[-1.0, 6.0, 12.0], 1.0e-4, 1.0e-4);

    let mut y_reused = device.upload(&[99.0f32, 99.0, 99.0]).unwrap();
    spmv_into(&device, &gpu_csr, &x_buf, &mut y_reused).unwrap();
    let mut got_y_reused = vec![0.0f32; 3];
    device.download(&y_reused, &mut got_y_reused).unwrap();
    assert_close_slice(&got_y_reused, &got_y, 1.0e-4, 1.0e-4);

    let y_prepared = device.upload(&[77.0f32, 77.0, 77.0]).unwrap();
    let prepared_spmv = prepare_spmv(&device, &gpu_csr, &x_buf, &y_prepared).unwrap();
    prepared_spmv.dispatch();
    let mut got_y_prepared = vec![0.0f32; 3];
    device.download(&y_prepared, &mut got_y_prepared).unwrap();
    assert_close_slice(&got_y_prepared, &got_y, 1.0e-4, 1.0e-4);

    let y_batched = device.upload(&[55.0f32, 55.0, 55.0]).unwrap();
    let batched_spmv = prepare_spmv(&device, &gpu_csr, &x_buf, &y_batched).unwrap();

    // SpMM: C = A * B, B = [ 1.0  2.0 ]
    //                      [ 3.0  4.0 ]
    //                      [ 5.0  6.0 ]
    // Expected C = [ 2*1 - 5, 2*2 - 6 ] = [ -3.0, -2.0 ]
    //              [ 3*3,     3*4     ]   [  9.0, 12.0 ]
    //              [ 4*5,     4*6     ]   [ 20.0, 24.0 ]
    let b_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_buf = device.upload(&b_host).unwrap();
    let b_layout = Layout::c_contiguous([3, 2]).unwrap();
    let b_op = StridedOperand {
        buffer: &b_buf,
        layout: &b_layout,
    };
    let c_buf = spmm(&device, &gpu_csr, &b_op).unwrap();
    let mut got_c = vec![0.0f32; 6];
    device.download(&c_buf, &mut got_c).unwrap();
    assert_close_slice(&got_c, &[-3.0, -2.0, 9.0, 12.0, 20.0, 24.0], 1.0e-4, 1.0e-4);

    let many_c_buf = spmv_many(&device, &gpu_csr, &b_op).unwrap();
    let mut got_many_c = vec![0.0f32; 6];
    device.download(&many_c_buf, &mut got_many_c).unwrap();
    assert_close_slice(&got_many_c, &got_c, 1.0e-4, 1.0e-4);

    let mut c_reused = device.upload(&[99.0f32; 6]).unwrap();
    spmm_into(&device, &gpu_csr, &b_op, &mut c_reused).unwrap();
    let mut got_c_reused = vec![0.0f32; 6];
    device.download(&c_reused, &mut got_c_reused).unwrap();
    assert_close_slice(&got_c_reused, &got_c, 1.0e-4, 1.0e-4);

    let mut many_c_reused = device.upload(&[88.0f32; 6]).unwrap();
    spmv_many_into(&device, &gpu_csr, &b_op, &mut many_c_reused).unwrap();
    let mut got_many_c_reused = vec![0.0f32; 6];
    device
        .download(&many_c_reused, &mut got_many_c_reused)
        .unwrap();
    assert_close_slice(&got_many_c_reused, &got_c, 1.0e-4, 1.0e-4);

    let mut c_prepared = device.upload(&[77.0f32; 6]).unwrap();
    let prepared_spmm = prepare_spmm(&device, &gpu_csr, &b_op, &mut c_prepared).unwrap();
    prepared_spmm.dispatch();
    let mut got_c_prepared = vec![0.0f32; 6];
    device.download(&c_prepared, &mut got_c_prepared).unwrap();
    assert_close_slice(&got_c_prepared, &got_c, 1.0e-4, 1.0e-4);

    let mut many_c_prepared = device.upload(&[66.0f32; 6]).unwrap();
    let prepared_many = prepare_spmv_many(&device, &gpu_csr, &b_op, &mut many_c_prepared).unwrap();
    prepared_many.dispatch();
    let mut got_many_c_prepared = vec![0.0f32; 6];
    device
        .download(&many_c_prepared, &mut got_many_c_prepared)
        .unwrap();
    assert_close_slice(&got_many_c_prepared, &got_c, 1.0e-4, 1.0e-4);

    let mut c_batched = device.upload(&[55.0f32; 6]).unwrap();
    let batched_spmm = prepare_spmm(&device, &gpu_csr, &b_op, &mut c_batched).unwrap();
    submit_prepared_sparse_batch(&[
        PreparedSparseDispatch::Spmv(&batched_spmv),
        PreparedSparseDispatch::Spmm(&batched_spmm),
    ])
    .unwrap();
    let mut got_y_batched = vec![0.0f32; 3];
    device.download(&y_batched, &mut got_y_batched).unwrap();
    assert_close_slice(&got_y_batched, &got_y, 1.0e-4, 1.0e-4);
    let mut got_c_batched = vec![0.0f32; 6];
    device.download(&c_batched, &mut got_c_batched).unwrap();
    assert_close_slice(&got_c_batched, &got_c, 1.0e-4, 1.0e-4);

    let empty_row_csr = leto_ops::CsrMatrix::from_parts(
        vec![2.0_f32, -1.0, 4.0],
        vec![0, 2, 2],
        vec![0, 2, 2, 3],
        3,
        3,
    )
    .unwrap();
    let empty_row_gpu = GpuCsrMatrix::from_cpu(&device, &empty_row_csr).unwrap();
    let empty_row_y = spmv(&device, &empty_row_gpu, &x_buf).unwrap();
    let mut got_empty_row_y = [0.0_f32; 3];
    device.download(&empty_row_y, &mut got_empty_row_y).unwrap();
    assert_eq!(got_empty_row_y, [-1.0, 0.0, 12.0]);

    let empty_row_c = spmm(&device, &empty_row_gpu, &b_op).unwrap();
    let mut got_empty_row_c = [0.0_f32; 6];
    device.download(&empty_row_c, &mut got_empty_row_c).unwrap();
    assert_eq!(got_empty_row_c, [-3.0, -2.0, 0.0, 0.0, 20.0, 24.0]);
}
