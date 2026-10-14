//! CUDA instantiation of the shared dense-product conformance clauses.

#![cfg(feature = "cuda")]

use eunomia::NumericElement;
use hephaestus_conformance::{
    assert_dense_composition_contract, assert_dense_matrix_function_contract,
    assert_dense_product_contract,
};
use hephaestus_core::{
    ComputeDevice, CudaC, DenseProductOps, DialectScalar, HephaestusError, Result, StridedView,
};
use hephaestus_cuda::{CudaDenseProductOps, CudaDevice, StridedOperand, batched_matmul, matmul};
use leto::Layout;

fn scalar_products<T: NumericElement + DialectScalar<CudaC>>(device: &CudaDevice) -> Result<()> {
    let _span =
        tracing::info_span!("scalar_products", scalar = core::any::type_name::<T>()).entered();
    let value = |n: usize| {
        T::try_from_count(n)
            .expect("invariant: every fixture constant is at most 99, exact in every tested scalar")
    };
    let lhs = device.upload(&[1, 2, 3, 4].map(value))?;
    let rhs = device.upload(&[5, 6, 7, 8].map(value))?;
    let output = device.upload(&[value(99); 4])?;
    let matrix = Layout::c_contiguous([2, 2])
        .expect("invariant: four-element matrix has representable strides");
    {
        let _span = tracing::info_span!("matmul").entered();
        CudaDenseProductOps.matmul_into(
            device,
            StridedView::new(&lhs, &matrix),
            StridedView::new(&rhs, &matrix),
            StridedView::new(&output, &matrix),
        )?;
        let mut actual = [T::ZERO; 4];
        device.download(&output, &mut actual)?;
        // Products and every partial sum are integers <= 50, exact in every tested scalar.
        assert_eq!(actual, [19, 22, 43, 50].map(value));
    }
    {
        let _span = tracing::info_span!("batched_matmul").entered();
        let left = device.upload(&[1, 2, 3, 4, 2, 0, 0, 3].map(value))?;
        let right = device.upload(&[5, 6, 7, 8, 1, 2, 3, 4].map(value))?;
        let destination = device.upload(&[value(99); 8])?;
        let batch = Layout::c_contiguous([2, 2, 2])
            .expect("invariant: eight-element batch has representable strides");
        CudaDenseProductOps.batched_matmul_into(
            device,
            StridedView::new(&left, &batch),
            StridedView::new(&right, &batch),
            StridedView::new(&destination, &batch),
        )?;
        let mut actual = [T::ZERO; 8];
        device.download(&destination, &mut actual)?;
        assert_eq!(actual, [19, 22, 43, 50, 2, 4, 9, 12].map(value));
    }
    {
        let _span = tracing::info_span!("kron").entered();
        let destination = device.upload(&[value(99); 16])?;
        let square = Layout::c_contiguous([4, 4])
            .expect("invariant: sixteen-element square has representable strides");
        CudaDenseProductOps.kron_into(
            device,
            StridedView::new(&lhs, &matrix),
            StridedView::new(&rhs, &matrix),
            StridedView::new(&destination, &square),
        )?;
        let mut actual = [T::ZERO; 16];
        device.download(&destination, &mut actual)?;
        assert_eq!(
            actual,
            [5, 6, 10, 12, 7, 8, 14, 16, 15, 18, 20, 24, 21, 24, 28, 32].map(value)
        );
    }
    Ok(())
}

fn accumulation_rounds_in_scalar<T: NumericElement + DialectScalar<CudaC>>(
    device: &CudaDevice,
    rounding_boundary: T,
) -> Result<()> {
    let _span = tracing::info_span!(
        "accumulation_rounding",
        scalar = core::any::type_name::<T>()
    )
    .entered();
    let lhs = device.upload(&[rounding_boundary, T::ONE, T::ZERO - rounding_boundary])?;
    let rhs = device.upload(&[T::ONE; 3])?;
    let output = device.upload(&[T::ONE])?;
    let row = Layout::c_contiguous([1, 3]).expect("invariant: three-element row is representable");
    let column =
        Layout::c_contiguous([3, 1]).expect("invariant: three-element column is representable");
    let scalar = Layout::c_contiguous([1, 1]).expect("invariant: scalar matrix is representable");
    CudaDenseProductOps.matmul_into(
        device,
        StridedView::new(&lhs, &row),
        StridedView::new(&rhs, &column),
        StridedView::new(&output, &scalar),
    )?;
    let mut actual = [T::ONE];
    device.download(&output, &mut actual)?;
    // At 2/epsilon, adding one is a tie rounded to the even significand.
    // Subsequent cancellation is exactly zero; wider accumulation yields one.
    assert_eq!(actual, [T::ZERO]);
    let row = Layout::c_contiguous([1, 1, 3]).expect("invariant: one batched row is representable");
    let column =
        Layout::c_contiguous([1, 3, 1]).expect("invariant: one batched column is representable");
    let scalar =
        Layout::c_contiguous([1, 1, 1]).expect("invariant: one batched scalar is representable");
    device.write_buffer(&output, &[T::ONE])?;
    CudaDenseProductOps.batched_matmul_into(
        device,
        StridedView::new(&lhs, &row),
        StridedView::new(&rhs, &column),
        StridedView::new(&output, &scalar),
    )?;
    device.download(&output, &mut actual)?;
    assert_eq!(actual, [T::ZERO]);
    Ok(())
}

#[test]
fn products_execute_supported_scalar_arithmetic() {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(
            tracing_subscriber::fmt::format::FmtSpan::ENTER
                | tracing_subscriber::fmt::format::FmtSpan::CLOSE,
        )
        .finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(HephaestusError::AdapterUnavailable { message })
            if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() =>
        {
            tracing::info!(%message, "CUDA adapter absent");
            return;
        }
        Err(error) => panic!("CUDA scalar products require an available device: {error}"),
    };
    // Evaluate every instantiation before reporting failures so one missing declaration
    // cannot conceal the other reduced-precision type's compiler diagnostics.
    let results = [
        scalar_products::<f32>(&device),
        scalar_products::<f64>(&device),
        scalar_products::<i32>(&device),
        scalar_products::<u32>(&device),
        scalar_products::<eunomia::F16>(&device),
        scalar_products::<eunomia::Bf16>(&device),
        accumulation_rounds_in_scalar(&device, 16_777_216.0f32),
        accumulation_rounds_in_scalar(&device, 9_007_199_254_740_992.0f64),
        accumulation_rounds_in_scalar(&device, eunomia::F16::from_f32(2048.0)),
        accumulation_rounds_in_scalar(&device, eunomia::Bf16::from_f32(256.0)),
    ];
    let failures: Vec<_> = results
        .into_iter()
        .filter_map(core::result::Result::err)
        .collect();
    assert!(
        failures.is_empty(),
        "scalar product failures: {failures:#?}"
    );
}

fn empty_shared_extent_writes_zero<T: NumericElement + DialectScalar<CudaC>>(
    device: &CudaDevice,
) -> Result<()> {
    let _span =
        tracing::info_span!("empty_shared_extent", scalar = core::any::type_name::<T>()).entered();
    let value = |n: usize| {
        T::try_from_count(n)
            .expect("invariant: every fixture constant is at most 99, exact in every tested scalar")
    };
    let sentinel = value(7);
    // Rank-2 `[2,0] x [0,3] -> [2,3]`, strided output over padded
    // backing: view cells {2,3,4,6,7,8} must zero, the other six backing
    // cells must keep their sentinel.
    let lhs_storage = device.upload(&[sentinel; 8])?;
    let rhs_storage = device.upload(&[sentinel; 8])?;
    let out_storage = device.upload(&[sentinel; 12])?;
    let lhs_layout =
        Layout::try_new([2, 0], [3, 1], 1).expect("invariant: empty-k row view is representable");
    let rhs_layout = Layout::try_new([0, 3], [3, 1], 1)
        .expect("invariant: empty-k column view is representable");
    let out_layout =
        Layout::try_new([2, 3], [4, 1], 2).expect("invariant: padded output view is representable");
    CudaDenseProductOps.matmul_into(
        device,
        StridedView::new(&lhs_storage, &lhs_layout),
        StridedView::new(&rhs_storage, &rhs_layout),
        StridedView::new(&out_storage, &out_layout),
    )?;
    let mut backing = [T::ZERO; 12];
    device.download(&out_storage, &mut backing)?;
    let zero = T::ZERO;
    assert_eq!(
        backing,
        [
            sentinel, sentinel, zero, zero, zero, sentinel, zero, zero, zero, sentinel, sentinel,
            sentinel
        ]
    );
    // Inputs are never read when the shared extent is zero.
    let mut lhs_backing = [T::ZERO; 8];
    device.download(&lhs_storage, &mut lhs_backing)?;
    assert_eq!(lhs_backing, [sentinel; 8]);
    let mut rhs_backing = [T::ZERO; 8];
    device.download(&rhs_storage, &mut rhs_backing)?;
    assert_eq!(rhs_backing, [sentinel; 8]);

    // Batched `[2,2,0] x [2,0,3] -> [2,2,3]`, dense sentinel output.
    let batch_lhs = device.upload(&[sentinel; 8])?;
    let batch_rhs = device.upload(&[sentinel; 8])?;
    let batch_out = device.upload(&[sentinel; 12])?;
    let batch_lhs_layout = Layout::try_new([2, 2, 0], [4, 3, 1], 0)
        .expect("invariant: batched empty-k view is representable");
    let batch_rhs_layout = Layout::try_new([2, 0, 3], [4, 1, 1], 0)
        .expect("invariant: batched empty-k view is representable");
    let batch_out_layout =
        Layout::c_contiguous([2, 2, 3]).expect("invariant: batched output is representable");
    CudaDenseProductOps.batched_matmul_into(
        device,
        StridedView::new(&batch_lhs, &batch_lhs_layout),
        StridedView::new(&batch_rhs, &batch_rhs_layout),
        StridedView::new(&batch_out, &batch_out_layout),
    )?;
    let mut batch_backing = [T::ZERO; 12];
    device.download(&batch_out, &mut batch_backing)?;
    assert_eq!(batch_backing, [T::ZERO; 12]);

    // Empty output stays a no-op and succeeds (separate buffers: the
    // alias rejection fires before the empty short-circuit by design).
    let empty_lhs = device.upload(&[sentinel; 4])?;
    let empty_rhs = device.upload(&[sentinel; 4])?;
    let empty_out = device.upload(&[sentinel; 4])?;
    let empty_layout =
        Layout::c_contiguous([0, 0]).expect("invariant: empty matrix is representable");
    CudaDenseProductOps.matmul_into(
        device,
        StridedView::new(&empty_lhs, &empty_layout),
        StridedView::new(&empty_rhs, &empty_layout),
        StridedView::new(&empty_out, &empty_layout),
    )?;
    let mut empty_backing = [T::ZERO; 4];
    device.download(&empty_out, &mut empty_backing)?;
    assert_eq!(empty_backing, [sentinel; 4]);

    // Allocating forms return initialized zeros for the same shapes.
    let out = matmul(
        device,
        StridedOperand {
            buffer: &lhs_storage,
            layout: &lhs_layout,
        },
        StridedOperand {
            buffer: &rhs_storage,
            layout: &rhs_layout,
        },
    )?;
    let mut allocated = [sentinel; 6];
    device.download(&out, &mut allocated)?;
    assert_eq!(allocated, [T::ZERO; 6]);
    let batch_allocated = batched_matmul(
        device,
        StridedOperand {
            buffer: &batch_lhs,
            layout: &batch_lhs_layout,
        },
        StridedOperand {
            buffer: &batch_rhs,
            layout: &batch_rhs_layout,
        },
    )?;
    let mut batch_allocated_backing = [sentinel; 12];
    device.download(&batch_allocated, &mut batch_allocated_backing)?;
    assert_eq!(batch_allocated_backing, [T::ZERO; 12]);
    Ok(())
}

#[test]
fn empty_products_write_zero_for_every_admitted_scalar() {
    let subscriber = tracing_subscriber::fmt().with_test_writer().finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(HephaestusError::AdapterUnavailable { message })
            if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() =>
        {
            tracing::info!(%message, "CUDA adapter absent");
            return;
        }
        Err(error) => panic!("CUDA empty-product test requires an available device: {error}"),
    };
    let results = [
        empty_shared_extent_writes_zero::<f32>(&device),
        empty_shared_extent_writes_zero::<f64>(&device),
        empty_shared_extent_writes_zero::<i32>(&device),
        empty_shared_extent_writes_zero::<u32>(&device),
        empty_shared_extent_writes_zero::<eunomia::F16>(&device),
        empty_shared_extent_writes_zero::<eunomia::Bf16>(&device),
    ];
    let failures: Vec<_> = results
        .into_iter()
        .filter_map(core::result::Result::err)
        .collect();
    assert!(failures.is_empty(), "empty-product failures: {failures:#?}");
}

#[test]
fn cuda_satisfies_the_dense_product_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA dense-product conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA dense-product conformance requires a physical device: {error}"),
    };
    assert_dense_product_contract(&device, &CudaDenseProductOps);
    assert_dense_composition_contract(&device, &CudaDenseProductOps);
    assert_dense_matrix_function_contract(&device, &CudaDenseProductOps);
}
