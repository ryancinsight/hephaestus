use super::*;

#[test]
fn prepared_sparse_dispatch_matches_reference() {
    let Some(device) = device("prepared_sparse_dispatch_matches_reference") else {
        return;
    };
    use hephaestus_metal::{
        GpuCsrMatrix, PreparedSparseDispatch, prepare_spmm, prepare_spmv, prepare_spmv_many,
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
    let gpu_csr = GpuCsrMatrix::from_cpu(&device, &cpu_csr).expect("Metal CSR upload");
    assert_eq!(gpu_csr.shape(), (3, 3));
    assert_eq!(gpu_csr.nnz(), 4);
    assert_eq!(
        gpu_csr.to_cpu(&device).expect("Metal CSR download"),
        cpu_csr
    );

    let x = device
        .upload(&[1.0_f32, 2.0, 3.0])
        .expect("SpMV input upload");
    let y = device.upload(&[77.0_f32; 3]).expect("SpMV output upload");
    let prepared_spmv = prepare_spmv(&device, &gpu_csr, &x, &y).expect("prepare Metal SpMV");
    prepared_spmv.dispatch();
    prepared_spmv.dispatch();

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
        prepare_spmm(&device, &gpu_csr, &b_operand, &mut c).expect("prepare Metal SpMM");
    let mut many = device
        .upload(&[99.0_f32; 6])
        .expect("batched output upload");
    let prepared_many = prepare_spmv_many(&device, &gpu_csr, &b_operand, &mut many)
        .expect("prepare Metal batched SpMV");
    submit_prepared_sparse_batch(&[
        PreparedSparseDispatch::Spmv(&prepared_spmv),
        PreparedSparseDispatch::Spmm(&prepared_spmm),
        PreparedSparseDispatch::Spmm(&prepared_many),
    ])
    .expect("submit Metal sparse batch");

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
