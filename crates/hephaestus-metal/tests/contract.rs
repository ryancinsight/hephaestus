//! Contract tests for the Metal `ComputeDevice` substrate and application operations.
//!
//! These run real device dispatch differentially against host references.
//! On a host without macOS or without a Metal device, [`MetalDevice::try_default`]
//! returns `Err` and each test skips. Hardware CI sets
//! `HEPHAESTUS_METAL_REQUIRE_DEVICE=1` so an unavailable device fails that lane
//! instead of being reported as device evidence.

use std::borrow::Cow;

use hephaestus_core::{
    BinaryStorageKernel, Binding, BindingDecl, BlockWidth, CommandStream, ComputeDevice,
    DeviceBuffer, DispatchGrid, GroupedBinding, GroupedBindingDecl, GroupedCommandStream,
    GroupedKernelDevice, GroupedKernelInterface, GroupedKernelSequence, GroupedKernelSource,
    HephaestusError, KernelDevice, KernelInterface, KernelSource, MultiStorageKernel, Result,
    UnaryStorageKernel, Wgsl,
};
use hephaestus_metal::{
    AddOp, ExpNegOp, ExpOp, GeluTanhGradOp, GeluTanhOp, MatrixFunction, MatrixNorm, MatrixProduct,
    MatrixProperties, MatrixSolve, MaxOp, MetalBinaryStorageKernel, MetalDevice,
    MetalMultiStorageKernel, MetalStorageBinding, MetalStorageBindingLayout,
    MetalUnaryStorageKernel, MinOp, MulOp, NegOp, SiluGradOp, SiluOp, SoftplusGradOp, SoftplusOp,
    SqrtOp, StridedOperand, SumOp, binary_elementwise, cumprod, cumprod_into, matmul,
    normal_with_seed, prepare_dot, prepare_max_axis_into, prepare_mean_axis_into,
    prepare_min_axis_into, prepare_norm_l2, prepare_reduction, prepare_reduction_with_width,
    prepare_sum_axis_into, prod_axis, reduce_axis_into, reduction, scalar_elementwise,
    submit_prepared_axis_reduction_batch, submit_prepared_reduction_batch, suffix_prod,
    suffix_prod_into, suffix_sum, suffix_sum_into, unary_elementwise, unary_elementwise_into,
    uniform_with_seed,
};
#[cfg(feature = "decomposition")]
use hephaestus_metal::{
    MatrixDecompose, col_piv_qr, col_piv_qr_blocked, full_piv_lu, full_piv_lu_blocked,
};
use leto::Layout;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StreamParams {
    len: u32,
    factor: f32,
}

struct StreamScaleKernel;

impl KernelInterface for StreamScaleKernel {
    type Params = StreamParams;

    const LABEL: &'static str = "hephaestus-metal-stream-scale";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [64, 1, 1];
}

impl KernelSource<Wgsl> for StreamScaleKernel {
    const ENTRY: &'static str = "main";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r#"
struct Params {
    len: u32,
    factor: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < params.len) {
        output[i] = input[i] * params.factor;
    }
}
"#,
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GroupedParams {
    len: u32,
    addend: f32,
}

struct GroupedAddKernel;

impl GroupedKernelInterface for GroupedAddKernel {
    type Params = GroupedParams;

    const LABEL: &'static str = "hephaestus-metal-grouped-add";
    const BINDINGS: &'static [GroupedBindingDecl] = &[
        GroupedBindingDecl::read_only::<f32>(0, 0),
        GroupedBindingDecl::read_only::<f32>(1, 0),
        GroupedBindingDecl::read_write::<f32>(1, 1),
    ];
    const PARAM_GROUP: u32 = 0;
    const PARAM_BINDING: u32 = 1;
    const WORKGROUP: [u32; 3] = [64, 1, 1];
}

impl GroupedKernelSource<Wgsl> for GroupedAddKernel {
    const ENTRY: &'static str = "main";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r#"
struct Params {
    len: u32,
    addend: f32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(1) @binding(0) var<storage, read> right: array<f32>;
@group(1) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < params.len) {
        output[i] = left[i] + right[i] + params.addend;
    }
}
"#,
        )
    }
}

/// Acquire a device, or `None` to skip (no Metal device).
fn device(test: &str) -> Option<MetalDevice> {
    match MetalDevice::try_default() {
        Ok(d) => Some(d),
        Err(e) => {
            if std::env::var_os("HEPHAESTUS_METAL_REQUIRE_DEVICE").is_some() {
                panic!("Metal device required for {test}: {e}");
            }
            eprintln!("skip {test}: Metal device unavailable ({e})");
            None
        }
    }
}

#[test]
fn authored_metal_stream_preserves_dispatch_copy_prefix_and_fill_order() {
    let Some(device) =
        device("authored_metal_stream_preserves_dispatch_copy_prefix_and_fill_order")
    else {
        return;
    };

    let input = device.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let scratch = device.alloc_zeroed::<f32>(4).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let copied = device.alloc_zeroed::<f32>(4).unwrap();
    let prefix = device.upload(&[99.0_f32; 4]).unwrap();
    let prepared = device.prepare(&StreamScaleKernel).unwrap();
    let grid = DispatchGrid::covering_domain([4, 1, 1], [64, 1, 1]).unwrap();

    let mut stream = device.stream().unwrap();
    stream.fill_zero(&scratch).unwrap();
    stream.copy(&input, &scratch).unwrap();
    stream
        .encode(
            &prepared,
            &[Binding::read(&scratch), Binding::read_write(&output)],
            &StreamParams {
                len: 4,
                factor: 2.5,
            },
            grid,
        )
        .unwrap();
    stream.copy(&output, &copied).unwrap();
    stream.copy_prefix(&output, &prefix, 2).unwrap();
    stream.fill_zero(&output).unwrap();
    stream.submit().unwrap();

    let mut got_output = [1.0_f32; 4];
    let mut got_copied = [0.0_f32; 4];
    let mut got_prefix = [0.0_f32; 4];
    device.download(&output, &mut got_output).unwrap();
    device.download(&copied, &mut got_copied).unwrap();
    device.download(&prefix, &mut got_prefix).unwrap();
    assert_eq!(got_output, [0.0, 0.0, 0.0, 0.0]);
    assert_eq!(got_copied, [2.5, 5.0, 7.5, 10.0]);
    assert_eq!(got_prefix, [2.5, 5.0, 99.0, 99.0]);
}

#[test]
fn grouped_metal_stream_preserves_grouped_output_and_sequence_order() {
    let Some(device) = device("grouped_metal_stream_preserves_grouped_output_and_sequence_order")
    else {
        return;
    };

    let left = device.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let right = device.upload(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
    let grid = DispatchGrid::covering_domain([4, 1, 1], [64, 1, 1]).unwrap();
    let bindings = [
        GroupedBinding::read(0, 0, &left),
        GroupedBinding::read(1, 0, &right),
        GroupedBinding::read_write(1, 1, &output),
    ];
    let params = GroupedParams {
        len: 4,
        addend: 0.5,
    };

    let mut stream = device.grouped_stream().unwrap();
    stream
        .encode_grouped(&prepared, &bindings, &params, grid)
        .unwrap();
    stream
        .encode_grouped_sequence("metal-grouped-sequence", |sequence| {
            sequence.encode_grouped(&prepared, &bindings, &params, grid)
        })
        .unwrap();
    stream.submit_grouped().unwrap();

    let mut got = [0.0_f32; 4];
    device.download(&output, &mut got).unwrap();
    assert_eq!(got, [11.5, 22.5, 33.5, 44.5]);
}

#[test]
fn metal_storage_kernels_match_values_and_reject_length_mismatches() {
    let Some(device) = device("metal_storage_kernels_match_values_and_reject_length_mismatches")
    else {
        return;
    };

    let left = device.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let right = device.upload(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let params = GroupedParams {
        len: 4,
        addend: 0.5,
    };
    let grid = DispatchGrid::new(1, 1, 1);

    let multi = MetalMultiStorageKernel::new(
        &device,
        "metal-multi-storage",
        r#"
struct Params {
    len: u32,
    addend: f32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < params.len) {
        output[i] = left[i] + right[i] + params.addend;
    }
}
"#,
        "main",
        &[
            MetalStorageBindingLayout::read_only(0),
            MetalStorageBindingLayout::read_only(1),
            MetalStorageBindingLayout::read_write(2),
        ],
        3,
    )
    .unwrap();
    multi
        .dispatch(
            &device,
            [
                MetalStorageBinding::new(0, &left),
                MetalStorageBinding::new(1, &right),
                MetalStorageBinding::new(2, &output),
            ],
            &params,
            grid,
        )
        .unwrap();

    let unary_output = device.alloc_zeroed::<f32>(4).unwrap();
    let unary = MetalUnaryStorageKernel::new(
        &device,
        "metal-unary-storage",
        r#"
struct Params {
    len: u32,
    factor: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < params.len) {
        output[i] = input[i] * params.factor;
    }
}
"#,
        "main",
    )
    .unwrap();
    unary
        .dispatch(
            &device,
            &left,
            &unary_output,
            &StreamParams {
                len: 4,
                factor: 2.0,
            },
            grid,
        )
        .unwrap();

    let binary_output = device.alloc_zeroed::<f32>(4).unwrap();
    let binary = MetalBinaryStorageKernel::new(
        &device,
        "metal-binary-storage",
        r#"
struct Params {
    len: u32,
    addend: f32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < params.len) {
        output[i] = left[i] - right[i] + params.addend;
    }
}
"#,
        "main",
    )
    .unwrap();
    binary
        .dispatch(&device, &left, &right, &binary_output, &params, grid)
        .unwrap();

    let short = device.upload(&[1.0_f32, 2.0, 3.0]).unwrap();
    assert!(matches!(
        unary.dispatch(
            &device,
            &short,
            &unary_output,
            &StreamParams {
                len: 3,
                factor: 2.0
            },
            grid,
        ),
        Err(HephaestusError::LengthMismatch {
            host_len: 3,
            device_len: 4,
        })
    ));

    let mut got_multi = [0.0_f32; 4];
    let mut got_unary = [0.0_f32; 4];
    let mut got_binary = [0.0_f32; 4];
    device.download(&output, &mut got_multi).unwrap();
    device.download(&unary_output, &mut got_unary).unwrap();
    device.download(&binary_output, &mut got_binary).unwrap();
    assert_eq!(got_multi, [11.5, 22.5, 33.5, 44.5]);
    assert_eq!(got_unary, [2.0, 4.0, 6.0, 8.0]);
    assert_eq!(got_binary, [-8.5, -17.5, -26.5, -35.5]);
}

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
    let mut y = device.upload(&[77.0_f32; 3]).expect("SpMV output upload");
    let prepared_spmv = prepare_spmv(&device, &gpu_csr, &x, &mut y).expect("prepare Metal SpMV");
    prepared_spmv.dispatch();
    prepared_spmv.dispatch();

    let b = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("SpMM input upload");
    let b_layout = Layout::new([3, 2], [1, 3], 0);
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
    let mut wrong_output = device
        .upload(&[0.0_f32; 3])
        .expect("wrong SpMV output upload");
    match prepare_spmv(&device, &gpu_csr, &wrong_x, &mut wrong_output) {
        Err(HephaestusError::LengthMismatch {
            host_len: 3,
            device_len: 2,
        }) => {}
        Err(error) => panic!("expected SpMV length rejection, got {error:?}"),
        Ok(_) => panic!("expected SpMV length rejection, got success"),
    }
    let bad_layout = Layout::new([3, 2], [2, 1], 5);
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

fn assert_elementwise_alias_rejected(result: Result<()>) {
    match result {
        Err(HephaestusError::DispatchFailed { message }) => {
            assert!(
                message.starts_with("output buffer must not alias "),
                "unexpected alias rejection message: {message}"
            );
        }
        other => panic!("expected elementwise alias rejection, got {other:?}"),
    }
}

fn assert_length_mismatch<T>(result: Result<T>, host_len: usize, device_len: usize) {
    match result {
        Err(HephaestusError::LengthMismatch {
            host_len: got_host,
            device_len: got_device,
        }) => {
            assert_eq!(got_host, host_len);
            assert_eq!(got_device, device_len);
        }
        Err(error) => panic!("expected length mismatch {host_len}->{device_len}, got {error:?}"),
        Ok(_) => panic!("expected length mismatch {host_len}->{device_len}, got success"),
    }
}

#[test]
fn upload_download_round_trips_values() {
    let Some(d) = device("upload_download_round_trips_values") else {
        return;
    };
    let host = [1.0f32, -2.0, 3.15, 0.0];
    let buf = d.upload(&host).unwrap();
    assert_eq!(buf.len(), 4);
    let mut out = [0.0f32; 4];
    d.download(&buf, &mut out).unwrap();
    assert_eq!(out, host);
}

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
fn scan_cumprod_convenience_preserves_strided_and_empty_contract() {
    let Some(device) = device("scan_cumprod_convenience_preserves_strided_and_empty_contract")
    else {
        return;
    };

    let physical = vec![1_i32, 2, 3, 4, 5, 6];
    let input = device.upload(&physical).unwrap();
    let transposed_layout = Layout::new([2, 3], [1, 2], 0);
    let output_layout = Layout::c_contiguous([2, 3]).unwrap();
    let output = device.alloc_zeroed::<i32>(6).unwrap();
    cumprod_into(
        &device,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        1,
        StridedOperand {
            buffer: &output,
            layout: &output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got = [0_i32; 6];
    device.download(&output, &mut got).unwrap();
    assert_eq!(got, [1, 3, 15, 2, 8, 48]);

    let allocated = cumprod(
        &device,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        0,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_allocated = [0_i32; 6];
    device.download(&allocated, &mut got_allocated).unwrap();
    assert_eq!(got_allocated, [1, 3, 5, 2, 12, 30]);

    let suffix_product = suffix_prod(
        &device,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_suffix_product = [0_i32; 6];
    device
        .download(&suffix_product, &mut got_suffix_product)
        .unwrap();
    assert_eq!(got_suffix_product, [15, 15, 5, 48, 24, 6]);

    let suffix_product_into = device.alloc_zeroed::<i32>(6).unwrap();
    suffix_prod_into(
        &device,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        0,
        StridedOperand {
            buffer: &suffix_product_into,
            layout: &output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_suffix_product_into = [0_i32; 6];
    device
        .download(&suffix_product_into, &mut got_suffix_product_into)
        .unwrap();
    assert_eq!(got_suffix_product_into, [2, 12, 30, 2, 4, 6]);

    let suffix = suffix_sum(
        &device,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_suffix = [0_i32; 6];
    device.download(&suffix, &mut got_suffix).unwrap();
    assert_eq!(got_suffix, [9, 8, 5, 12, 10, 6]);

    let suffix_into = device.alloc_zeroed::<i32>(6).unwrap();
    suffix_sum_into(
        &device,
        StridedOperand {
            buffer: &input,
            layout: &transposed_layout,
        },
        0,
        StridedOperand {
            buffer: &suffix_into,
            layout: &output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_suffix_into = [0_i32; 6];
    device.download(&suffix_into, &mut got_suffix_into).unwrap();
    assert_eq!(got_suffix_into, [3, 7, 11, 2, 4, 6]);

    let empty = device.alloc_zeroed::<i32>(0).unwrap();
    let empty_layout = Layout::c_contiguous([2, 0]).unwrap();
    let empty_output = cumprod(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    assert_eq!(empty_output.len(), 0);

    let invalid_layout = Layout::new([2, 3], [1, 2], 1);
    assert!(matches!(
        cumprod(
            &device,
            StridedOperand {
                buffer: &input,
                layout: &invalid_layout,
            },
            1,
            BlockWidth::DEFAULT,
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("layout rejected:")
    ));
}

#[test]
fn download_rejects_length_mismatch() {
    let Some(d) = device("download_rejects_length_mismatch") else {
        return;
    };
    let buf = d.upload(&[1.0f32, 2.0]).unwrap();
    let mut out = [0.0f32; 3];
    assert_length_mismatch(d.download(&buf, &mut out), 3, 2);
}

#[test]
fn write_buffer_rejects_length_mismatch() {
    let Some(d) = device("write_buffer_rejects_length_mismatch") else {
        return;
    };
    let buf = d.upload(&[1.0f32, 2.0]).unwrap();
    let host = [1.0f32, 2.0, 3.0];
    assert_length_mismatch(d.write_buffer(&buf, &host), 3, 2);
}

#[test]
fn write_sub_buffer_overwrites_only_requested_range() {
    let Some(d) = device("write_sub_buffer_overwrites_only_requested_range") else {
        return;
    };
    let buf = d.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    d.write_sub_buffer(&buf, 1, &[20.0f32, 30.0]).unwrap();

    let mut out = [0.0f32; 4];
    d.download(&buf, &mut out).unwrap();
    assert_eq!(out, [1.0, 20.0, 30.0, 4.0]);
}

#[test]
fn write_sub_buffer_rejects_out_of_range_write() {
    let Some(d) = device("write_sub_buffer_rejects_out_of_range_write") else {
        return;
    };
    let buf = d.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    assert_length_mismatch(d.write_sub_buffer(&buf, 2, &[4.0f32, 5.0]), 4, 3);
}

#[test]
fn write_sub_buffer_empty_tail_write_is_noop() {
    let Some(d) = device("write_sub_buffer_empty_tail_write_is_noop") else {
        return;
    };
    let buf = d.upload(&[9i32, 8, 7]).unwrap();
    d.write_sub_buffer(&buf, 3, &[] as &[i32]).unwrap();

    let mut out = [0i32; 3];
    d.download(&buf, &mut out).unwrap();
    assert_eq!(out, [9, 8, 7]);
}

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

#[test]
fn reduction_sum_matches_cpu_reference() {
    let Some(d) = device("reduction_sum_matches_cpu_reference") else {
        return;
    };
    let a = d.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let out = reduction::<SumOp, f32>(&d, &a).unwrap();
    let mut host_out = [0.0f32; 1];
    d.download(&out, &mut host_out).unwrap();
    assert_eq!(host_out[0], 10.0);
}

#[test]
fn reduction_axis_into_matches_cpu_reference() {
    let Some(d) = device("reduction_axis_into_matches_cpu_reference") else {
        return;
    };

    let input = d.upload(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let input_layout = Layout::c_contiguous([2, 3]).unwrap();
    let output = d.alloc_zeroed::<f32>(3).unwrap();
    let output_layout = Layout::c_contiguous([1, 3]).unwrap();

    reduce_axis_into::<SumOp, f32>(
        &d,
        StridedOperand {
            buffer: &input,
            layout: &input_layout,
        },
        0,
        StridedOperand {
            buffer: &output,
            layout: &output_layout,
        },
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = [0.0f32; 3];
    d.download(&output, &mut got).unwrap();
    assert_eq!(got, [5.0, 7.0, 9.0]);

    let product = prod_axis(
        &d,
        StridedOperand {
            buffer: &input,
            layout: &input_layout,
        },
        1,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got_product = [0.0f32; 2];
    d.download(&product, &mut got_product).unwrap();
    assert_eq!(got_product, [6.0, 120.0]);
}

#[test]
fn prepared_reduction_reuses_device_outputs_and_batches() {
    let Some(d) = device("prepared_reduction_reuses_device_outputs_and_batches") else {
        return;
    };
    let input = d.upload(&[3.0f32, -2.0, 7.0, 1.0, 4.0]).unwrap();
    let width = BlockWidth::new(4).unwrap();

    let sum = prepare_reduction_with_width::<SumOp, _>(&d, &input, width).unwrap();
    sum.dispatch(&d).unwrap();
    let sum_output = sum.output();
    let mut got_sum = [0.0f32];
    d.download(&sum_output, &mut got_sum).unwrap();
    assert_eq!(got_sum, [13.0]);
    sum.dispatch(&d).unwrap();
    d.download(&sum.output(), &mut got_sum).unwrap();
    assert_eq!(got_sum, [13.0]);

    let min = prepare_reduction::<MinOp, _>(&d, &input).unwrap();
    let max = prepare_reduction::<MaxOp, _>(&d, &input).unwrap();
    submit_prepared_reduction_batch(&d, &[&min, &max]).unwrap();
    let mut got_min = [0.0f32];
    let mut got_max = [0.0f32];
    d.download(&min.output(), &mut got_min).unwrap();
    d.download(&max.output(), &mut got_max).unwrap();
    assert_eq!(got_min, [-2.0]);
    assert_eq!(got_max, [7.0]);

    let empty = d.upload::<f32>(&[]).unwrap();
    let prepared_empty = prepare_reduction::<SumOp, _>(&d, &empty).unwrap();
    prepared_empty.dispatch(&d).unwrap();
    let mut got_empty = [f32::NAN];
    d.download(&prepared_empty.output(), &mut got_empty)
        .unwrap();
    assert_eq!(got_empty, [0.0]);

    let invalid_width = BlockWidth::new(3).unwrap();
    assert!(matches!(
        prepare_reduction_with_width::<SumOp, _>(&d, &input, invalid_width),
        Err(HephaestusError::DispatchFailed { message })
            if message == "reduction block width 3 must be a power of two"
    ));
}

#[test]
fn prepared_axis_reductions_reuse_plans_and_validate_contracts() {
    let Some(d) = device("prepared_axis_reductions_reuse_plans_and_validate_contracts") else {
        return;
    };

    let host: Vec<f32> = (1..=12).map(|value| value as f32).collect();
    let input = d.upload(&host).unwrap();
    let input_layout = Layout::c_contiguous([3, 4]).unwrap();
    let input_operand = StridedOperand {
        buffer: &input,
        layout: &input_layout,
    };
    let width = BlockWidth::new(2).unwrap();

    let axis0_out = d.alloc_zeroed::<f32>(4).unwrap();
    let axis0_layout = Layout::c_contiguous([1, 4]).unwrap();
    let prepared_sum_axis0 = prepare_sum_axis_into(
        &d,
        input_operand,
        0,
        StridedOperand {
            buffer: &axis0_out,
            layout: &axis0_layout,
        },
        width,
    )
    .unwrap();
    prepared_sum_axis0.dispatch(&d).unwrap();
    let mut got_axis0 = [0.0f32; 4];
    d.download(&axis0_out, &mut got_axis0).unwrap();
    assert_eq!(got_axis0, [15.0, 18.0, 21.0, 24.0]);
    prepared_sum_axis0.dispatch(&d).unwrap();
    d.download(&axis0_out, &mut got_axis0).unwrap();
    assert_eq!(got_axis0, [15.0, 18.0, 21.0, 24.0]);

    let transposed_layout = Layout::new([4, 3], [1, 4], 0);
    let transposed_input = StridedOperand {
        buffer: &input,
        layout: &transposed_layout,
    };
    let axis1_out = d.alloc_zeroed::<f32>(4).unwrap();
    let axis1_layout = Layout::c_contiguous([4, 1]).unwrap();
    let prepared_sum_axis1 = prepare_sum_axis_into(
        &d,
        transposed_input,
        1,
        StridedOperand {
            buffer: &axis1_out,
            layout: &axis1_layout,
        },
        width,
    )
    .unwrap();
    let max_axis0_out = d.alloc_zeroed::<f32>(3).unwrap();
    let max_axis0_layout = Layout::c_contiguous([1, 3]).unwrap();
    let prepared_max_axis0 = prepare_max_axis_into(
        &d,
        transposed_input,
        0,
        StridedOperand {
            buffer: &max_axis0_out,
            layout: &max_axis0_layout,
        },
        width,
    )
    .unwrap();
    submit_prepared_axis_reduction_batch(&d, &[&prepared_sum_axis1, &prepared_max_axis0]).unwrap();
    let mut got_axis1 = [0.0f32; 4];
    let mut got_max_axis0 = [0.0f32; 3];
    d.download(&axis1_out, &mut got_axis1).unwrap();
    d.download(&max_axis0_out, &mut got_max_axis0).unwrap();
    assert_eq!(got_axis1, [15.0, 18.0, 21.0, 24.0]);
    assert_eq!(got_max_axis0, [4.0, 8.0, 12.0]);

    let mean_axis1_out = d.alloc_zeroed::<f32>(4).unwrap();
    let prepared_mean_axis1 = prepare_mean_axis_into(
        &d,
        transposed_input,
        1,
        StridedOperand {
            buffer: &mean_axis1_out,
            layout: &axis1_layout,
        },
        width,
    )
    .unwrap();
    prepared_mean_axis1.dispatch(&d).unwrap();
    let mut got_mean_axis1 = [0.0f32; 4];
    d.download(&mean_axis1_out, &mut got_mean_axis1).unwrap();
    assert_eq!(got_mean_axis1, [5.0, 6.0, 7.0, 8.0]);

    let empty_input = d.upload::<f32>(&[]).unwrap();
    let empty_input_layout = Layout::c_contiguous([3, 0]).unwrap();
    let empty_output = d.upload(&[7.0f32; 3]).unwrap();
    let empty_output_layout = Layout::c_contiguous([3, 1]).unwrap();
    let prepared_empty_sum = prepare_sum_axis_into(
        &d,
        StridedOperand {
            buffer: &empty_input,
            layout: &empty_input_layout,
        },
        1,
        StridedOperand {
            buffer: &empty_output,
            layout: &empty_output_layout,
        },
        width,
    )
    .unwrap();
    prepared_empty_sum.dispatch(&d).unwrap();
    let mut got_empty = [7.0f32; 3];
    d.download(&empty_output, &mut got_empty).unwrap();
    assert_eq!(got_empty, [0.0, 0.0, 0.0]);

    let empty_product = prod_axis(
        &d,
        StridedOperand {
            buffer: &empty_input,
            layout: &empty_input_layout,
        },
        1,
        width,
    )
    .unwrap();
    let mut got_empty_product = [0.0f32; 3];
    d.download(&empty_product, &mut got_empty_product).unwrap();
    assert_eq!(got_empty_product, [1.0, 1.0, 1.0]);

    let empty_min = prepare_min_axis_into(
        &d,
        StridedOperand {
            buffer: &empty_input,
            layout: &empty_input_layout,
        },
        1,
        StridedOperand {
            buffer: &empty_output,
            layout: &empty_output_layout,
        },
        width,
    );
    assert!(matches!(
        empty_min,
        Err(HephaestusError::DispatchFailed { message })
            if message == "min_axis is undefined for empty axis 1"
    ));

    let alias_layout = Layout::c_contiguous([3, 1]).unwrap();
    let alias = prepare_sum_axis_into(
        &d,
        input_operand,
        1,
        StridedOperand {
            buffer: &input,
            layout: &alias_layout,
        },
        width,
    );
    assert!(matches!(
        alias,
        Err(HephaestusError::DispatchFailed { message })
            if message == "axis reduction output buffer must not alias input buffer"
    ));

    let invalid_width = BlockWidth::new(3).unwrap();
    let invalid = prepare_sum_axis_into(
        &d,
        input_operand,
        0,
        StridedOperand {
            buffer: &axis0_out,
            layout: &axis0_layout,
        },
        invalid_width,
    );
    assert!(matches!(
        invalid,
        Err(HephaestusError::DispatchFailed { message })
            if message == "reduction block width 3 must be a power of two"
    ));
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

#[test]
fn fluent_linalg_traits_match_value_contracts() {
    let Some(device) = device("fluent_linalg_traits_match_value_contracts") else {
        return;
    };

    let matrix = device.upload(&[4.0_f32, 1.0, 1.0, 3.0]).unwrap();
    let matrix_layout = Layout::c_contiguous([2, 2]).unwrap();
    let operand = StridedOperand {
        buffer: &matrix,
        layout: &matrix_layout,
    };

    let product = operand.matmul(&device, &operand).unwrap();
    let mut got_product = [0.0_f32; 4];
    device.download(&product, &mut got_product).unwrap();
    assert_eq!(got_product, [17.0, 7.0, 7.0, 10.0]);

    let l1 = operand.norm_l1(&device).unwrap();
    let l2 = operand.norm_l2(&device).unwrap();
    let max = operand.norm_max(&device).unwrap();
    let trace = operand.trace(&device).unwrap();
    let det = operand.det(&device).unwrap();
    let mut got_scalar = [0.0_f32];
    device.download(&l1, &mut got_scalar).unwrap();
    assert_eq!(got_scalar, [9.0]);
    device.download(&l2, &mut got_scalar).unwrap();
    assert!((got_scalar[0] - 27.0_f32.sqrt()).abs() <= 8.0 * f32::EPSILON);
    device.download(&max, &mut got_scalar).unwrap();
    assert_eq!(got_scalar, [4.0]);
    device.download(&trace, &mut got_scalar).unwrap();
    assert_eq!(got_scalar, [7.0]);
    device.download(&det, &mut got_scalar).unwrap();
    assert_eq!(got_scalar, [11.0]);
    assert_eq!(operand.rank(&device).unwrap(), 2);
    assert_eq!(operand.rank_with_tolerance(&device, 1.0e-6).unwrap(), 2);

    let power = operand.matpow(&device, 2).unwrap();
    let mut got_power = [0.0_f32; 4];
    device.download(&power, &mut got_power).unwrap();
    assert_eq!(got_power, got_product);

    let rhs = device.upload(&[5.0_f32, 7.0]).unwrap();
    let solution = operand.solve(&device, &rhs).unwrap();
    let mut got_solution = [0.0_f32; 2];
    device.download(&solution, &mut got_solution).unwrap();
    assert!((got_solution[0] - 8.0 / 11.0).abs() <= 8.0 * f32::EPSILON);
    assert!((got_solution[1] - 23.0 / 11.0).abs() <= 8.0 * f32::EPSILON);

    let inverse = operand.inv(&device).unwrap();
    let pseudoinverse = operand.pinv(&device).unwrap();
    let mut got_inverse = [0.0_f32; 4];
    let mut got_pseudoinverse = [0.0_f32; 4];
    device.download(&inverse, &mut got_inverse).unwrap();
    device
        .download(&pseudoinverse, &mut got_pseudoinverse)
        .unwrap();
    let expected_inverse = [3.0 / 11.0, -1.0 / 11.0, -1.0 / 11.0, 4.0 / 11.0];
    for (got, expected) in got_inverse.iter().zip(expected_inverse) {
        assert!((got - expected).abs() <= 16.0 * f32::EPSILON);
    }
    for (got, expected) in got_pseudoinverse.iter().zip(expected_inverse) {
        assert!((got - expected).abs() <= 16.0 * f32::EPSILON);
    }

    let diagonal = device.upload(&[1.0_f32, 0.0, 0.0, 2.0]).unwrap();
    let diagonal_operand = StridedOperand {
        buffer: &diagonal,
        layout: &matrix_layout,
    };
    let exponential = diagonal_operand.matexp(&device).unwrap();
    let mut got_exponential = [0.0_f32; 4];
    device.download(&exponential, &mut got_exponential).unwrap();
    assert!((got_exponential[0] - 1.0_f32.exp()).abs() <= 16.0 * f32::EPSILON);
    assert!((got_exponential[3] - 2.0_f32.exp()).abs() <= 16.0 * f32::EPSILON);
}

#[cfg(feature = "decomposition")]
#[test]
fn fluent_matrix_decomposition_uses_metal_selected_device() {
    let Some(device) = device("fluent_matrix_decomposition_uses_metal_selected_device") else {
        return;
    };

    let matrix = device.upload(&[4.0_f32, 1.0, 1.0, 3.0]).unwrap();
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let operand = StridedOperand {
        buffer: &matrix,
        layout: &layout,
    };
    let lu = operand.lu(&device).unwrap();
    assert!((lu.det() - 11.0).abs() <= 8.0 * f32::EPSILON);
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_pivoted_decompositions_match_ordinary_contracts() {
    let Some(device) = device("blocked_pivoted_decompositions_match_ordinary_contracts") else {
        return;
    };

    let square = device.upload(&[2.0f32, 5.0, 1.0, 2.0]).unwrap();
    let square_layout = Layout::c_contiguous([2, 2]).unwrap();
    let square_operand = StridedOperand {
        buffer: &square,
        layout: &square_layout,
    };
    let ordinary_lu = full_piv_lu(&device, square_operand).unwrap();
    let blocked_lu = full_piv_lu_blocked(&device, square_operand).unwrap();
    assert_eq!(blocked_lu.rank(), ordinary_lu.rank());
    assert_eq!(blocked_lu.row_permutation(), ordinary_lu.row_permutation());
    assert_eq!(blocked_lu.col_permutation(), ordinary_lu.col_permutation());
    assert!((blocked_lu.det() - ordinary_lu.det()).abs() <= 1.0e-5);

    let tall = device.upload(&[1.0f32, 0.0, 0.0, 2.0, 0.0, 0.0]).unwrap();
    let tall_layout = Layout::c_contiguous([3, 2]).unwrap();
    let tall_operand = StridedOperand {
        buffer: &tall,
        layout: &tall_layout,
    };
    let ordinary_qr = col_piv_qr(&device, tall_operand).unwrap();
    let blocked_qr = col_piv_qr_blocked(&device, tall_operand).unwrap();
    assert_eq!(blocked_qr.rank(), ordinary_qr.rank());
    assert_eq!(blocked_qr.permutation(), ordinary_qr.permutation());
}

#[cfg(feature = "decomposition")]
#[test]
fn blocked_pivoted_decompositions_reject_non_dense_operands() {
    let Some(device) = device("blocked_pivoted_decompositions_reject_non_dense_operands") else {
        return;
    };
    let dense = device.upload(&[1.0f32; 16]).unwrap();
    let small = device.upload(&[1.0f32; 4]).unwrap();
    let transposed = Layout::new([4, 4], [1, 4], 0);
    let broadcast = Layout::new([4, 4], [0, 1], 0);

    let lu_result = full_piv_lu_blocked(
        &device,
        StridedOperand {
            buffer: &dense,
            layout: &transposed,
        },
    );
    assert!(
        matches!(
            lu_result,
            Err(HephaestusError::DispatchFailed { message })
                if message.contains("dense C-contiguous")
        ),
        "complete-pivoted LU must reject non-dense operands"
    );

    let qr_result = col_piv_qr_blocked(
        &device,
        StridedOperand {
            buffer: &small,
            layout: &broadcast,
        },
    );
    assert!(
        matches!(
            qr_result,
            Err(HephaestusError::DispatchFailed { message })
                if message.contains("dense C-contiguous")
        ),
        "column-pivoted QR must reject non-dense operands"
    );
}

#[test]
fn prepared_map_reductions_reuse_resources_and_validate_layouts() {
    let Some(device) = device("prepared_map_reductions_reuse_resources_and_validate_layouts")
    else {
        return;
    };

    let lhs = device.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let rhs = device.upload(&[5.0_f32, 6.0, 7.0, 8.0]).unwrap();
    let contiguous = Layout::c_contiguous([4]).unwrap();
    let prepared_dot = prepare_dot(
        &device,
        StridedOperand {
            buffer: &lhs,
            layout: &contiguous,
        },
        StridedOperand {
            buffer: &rhs,
            layout: &contiguous,
        },
    )
    .unwrap();
    let dot_output = prepared_dot.output().wgpu_buffer().raw().clone();
    prepared_dot.dispatch(&device).unwrap();
    let mut got = [0.0_f32];
    let dot_buffer = prepared_dot.output();
    device.download(&dot_buffer, &mut got).unwrap();
    assert_eq!(got, [70.0]);
    assert_eq!(&dot_output, dot_buffer.wgpu_buffer().raw());

    device
        .write_buffer(&lhs, &[2.0_f32, 2.0, 2.0, 2.0])
        .unwrap();
    prepared_dot.dispatch(&device).unwrap();
    let dot_buffer = prepared_dot.output();
    device.download(&dot_buffer, &mut got).unwrap();
    assert_eq!(got, [52.0]);
    assert_eq!(&dot_output, dot_buffer.wgpu_buffer().raw());

    let reversed = Layout::new([4], [-1], 3);
    let reversed_dot = prepare_dot(
        &device,
        StridedOperand {
            buffer: &rhs,
            layout: &reversed,
        },
        StridedOperand {
            buffer: &lhs,
            layout: &contiguous,
        },
    )
    .unwrap();
    reversed_dot.dispatch(&device).unwrap();
    let reversed_buffer = reversed_dot.output();
    device.download(&reversed_buffer, &mut got).unwrap();
    assert_eq!(got, [52.0]);

    let norm_input = device.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let transposed = Layout::new([2, 2], [1, 2], 0);
    let prepared_norm = prepare_norm_l2(
        &device,
        StridedOperand {
            buffer: &norm_input,
            layout: &transposed,
        },
    )
    .unwrap();
    let norm_output = prepared_norm.output().wgpu_buffer().raw().clone();
    prepared_norm.dispatch(&device).unwrap();
    let norm_buffer = prepared_norm.output();
    device.download(&norm_buffer, &mut got).unwrap();
    let expected = 30.0_f32.sqrt();
    assert!((got[0] - expected).abs() <= 2.0 * f32::EPSILON * expected.max(1.0));
    assert_eq!(&norm_output, norm_buffer.wgpu_buffer().raw());

    device.write_buffer(&norm_input, &[4.0_f32; 4]).unwrap();
    prepared_norm.dispatch(&device).unwrap();
    let norm_buffer = prepared_norm.output();
    device.download(&norm_buffer, &mut got).unwrap();
    assert_eq!(got, [8.0]);
    assert_eq!(&norm_output, norm_buffer.wgpu_buffer().raw());

    let empty = device.upload::<f32>(&[]).unwrap();
    let empty_layout = Layout::c_contiguous([0]).unwrap();
    let empty_dot = prepare_dot(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    )
    .unwrap();
    empty_dot.dispatch(&device).unwrap();
    let empty_dot_buffer = empty_dot.output();
    device.download(&empty_dot_buffer, &mut got).unwrap();
    assert_eq!(got, [0.0]);

    let empty_norm = prepare_norm_l2(
        &device,
        StridedOperand {
            buffer: &empty,
            layout: &empty_layout,
        },
    )
    .unwrap();
    empty_norm.dispatch(&device).unwrap();
    let empty_norm_buffer = empty_norm.output();
    device.download(&empty_norm_buffer, &mut got).unwrap();
    assert_eq!(got, [0.0]);

    let wrong_shape = Layout::c_contiguous([3]).unwrap();
    assert!(matches!(
        prepare_dot(
            &device,
            StridedOperand {
                buffer: &lhs,
                layout: &contiguous,
            },
            StridedOperand {
                buffer: &rhs,
                layout: &wrong_shape,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("dot product shape mismatch:")
    ));

    let invalid_layout = Layout::new([3], [1], 2);
    assert!(matches!(
        prepare_norm_l2(
            &device,
            StridedOperand {
                buffer: &lhs,
                layout: &invalid_layout,
            },
        ),
        Err(HephaestusError::DispatchFailed { message })
            if message.starts_with("layout rejected:")
    ));
}
