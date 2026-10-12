use super::*;

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
fn device_local_copy_preserves_values_and_rejects_mismatch() {
    let Some(device) = device("device_local_copy_preserves_values_and_rejects_mismatch") else {
        return;
    };
    let host: Vec<f32> = (0..1027).map(|index| index as f32 * 0.25 - 17.0).collect();
    let source = device.upload(&host).unwrap();
    let destination = device.alloc_zeroed::<f32>(host.len()).unwrap();

    device.copy_buffer(&source, &destination).unwrap();
    let mut copied = vec![0.0_f32; host.len()];
    device.download(&destination, &mut copied).unwrap();
    assert_eq!(copied, host);

    let short = device.alloc_zeroed::<f32>(host.len() - 1).unwrap();
    assert_length_mismatch(
        device.copy_buffer(&source, &short),
        source.len(),
        short.len(),
    );
}

#[test]
fn uninitialized_allocation_is_fully_overwritten_before_read() {
    let Some(device) = device("uninitialized_allocation_is_fully_overwritten_before_read") else {
        return;
    };
    let expected = [1.0_f32, -2.5, 3.25, 8.0];
    let buffer = device.alloc_uninitialized::<f32>(expected.len()).unwrap();
    device.write_buffer(&buffer, &expected).unwrap();
    let mut actual = [0.0_f32; 4];
    device.download(&buffer, &mut actual).unwrap();
    assert_eq!(actual, expected);
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
