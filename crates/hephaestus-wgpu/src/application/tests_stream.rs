use super::*;
use std::borrow::Cow;

use hephaestus_core::{
    Access, BindingDecl, ComputeDevice, GroupedBindingDecl, GroupedKernelInterface, KernelInterface,
};

#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct ScaleParams {
    len: u32,
    factor: f32,
}

struct ScaleKernel;

impl KernelInterface for ScaleKernel {
    type Params = ScaleParams;
    const LABEL: &'static str = "hephaestus-wgpu-stream-scale";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [64, 1, 1];
}

#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
struct GroupedParams {
    len: u32,
    addend: f32,
}

struct GroupedAddKernel;

impl GroupedKernelInterface for GroupedAddKernel {
    type Params = GroupedParams;
    const LABEL: &'static str = "hephaestus-wgpu-grouped-add";
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

const OVERSIZED_PARAMETER_WORDS: usize = 16_385;
static OVERSIZED_PARAMETERS: [u32; OVERSIZED_PARAMETER_WORDS] = [0; OVERSIZED_PARAMETER_WORDS];

struct OversizedGroupedKernel;

impl GroupedKernelInterface for OversizedGroupedKernel {
    type Params = [u32; OVERSIZED_PARAMETER_WORDS];
    const LABEL: &'static str = "hephaestus-wgpu-oversized-grouped";
    const BINDINGS: &'static [GroupedBindingDecl] = GroupedAddKernel::BINDINGS;
    const PARAM_GROUP: u32 = GroupedAddKernel::PARAM_GROUP;
    const PARAM_BINDING: u32 = GroupedAddKernel::PARAM_BINDING;
    const WORKGROUP: [u32; 3] = GroupedAddKernel::WORKGROUP;
}

impl GroupedKernelSource<Wgsl> for OversizedGroupedKernel {
    const ENTRY: &'static str = GroupedAddKernel::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        GroupedAddKernel.source()
    }
}

impl KernelSource<Wgsl> for ScaleKernel {
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

fn try_device() -> Option<WgpuDevice> {
    WgpuDevice::try_with_power_preference_and_adapter_config(
        "hephaestus-wgpu-stream-test",
        wgpu::PowerPreference::HighPerformance,
        |_| wgpu::Features::empty(),
        |_| wgpu::Limits::default(),
    )
    .ok()
}

#[test]
fn module_cases_share_process_state() {
    crate::test_support::run_cases(&[
        (
            "command_stream_dispatches_prepared_wgsl_kernel",
            command_stream_dispatches_prepared_wgsl_kernel as fn(),
        ),
        (
            "command_stream_preserves_order_for_fill_copy_and_dispatch",
            command_stream_preserves_order_for_fill_copy_and_dispatch as fn(),
        ),
        (
            "command_stream_rejects_binding_contract_mismatch",
            command_stream_rejects_binding_contract_mismatch as fn(),
        ),
        (
            "command_stream_copy_rejects_length_mismatch",
            command_stream_copy_rejects_length_mismatch as fn(),
        ),
        (
            "command_stream_copy_prefix_preserves_the_destination_suffix",
            command_stream_copy_prefix_preserves_the_destination_suffix as fn(),
        ),
        (
            "grouped_command_stream_dispatches_multi_group_wgsl_kernel",
            grouped_command_stream_dispatches_multi_group_wgsl_kernel as fn(),
        ),
        (
            "grouped_sequence_preserves_order_in_one_wgpu_pass",
            grouped_sequence_preserves_order_in_one_wgpu_pass as fn(),
        ),
        (
            "bound_grouped_sequence_reuses_fixed_resources",
            bound_grouped_sequence_reuses_fixed_resources as fn(),
        ),
        (
            "bound_grouped_dispatch_updates_retained_parameters",
            bound_grouped_dispatch_updates_retained_parameters as fn(),
        ),
        (
            "bound_grouped_parameter_update_rejects_foreign_device",
            bound_grouped_parameter_update_rejects_foreign_device as fn(),
        ),
        (
            "bound_grouped_dispatch_rejects_invalid_parameter_size",
            bound_grouped_dispatch_rejects_invalid_parameter_size as fn(),
        ),
        (
            "bound_grouped_sequence_rejects_foreign_device",
            bound_grouped_sequence_rejects_foreign_device as fn(),
        ),
        (
            "bound_grouped_dispatch_rejects_group_mismatch",
            bound_grouped_dispatch_rejects_group_mismatch as fn(),
        ),
        (
            "grouped_command_stream_rejects_group_mismatch",
            grouped_command_stream_rejects_group_mismatch as fn(),
        ),
        (
            "kernel_interface_declares_wgpu_parameter_binding_after_storage",
            kernel_interface_declares_wgpu_parameter_binding_after_storage as fn(),
        ),
    ]);
}

fn command_stream_dispatches_prepared_wgsl_kernel() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping command stream test");
        return;
    };
    let input = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let prepared = device.prepare(&ScaleKernel).unwrap();

    device
        .dispatch(
            &prepared,
            &[Binding::read(&input), Binding::read_write(&output)],
            &ScaleParams {
                len: 4,
                factor: 2.5,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap();

    let mut host = [0.0f32; 4];
    device.download(&output, &mut host).unwrap();
    assert_eq!(host, [2.5, 5.0, 7.5, 10.0]);
}

fn command_stream_preserves_order_for_fill_copy_and_dispatch() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping command stream order test");
        return;
    };
    let input = device.upload(&[3.0f32, 4.0, 5.0, 6.0]).unwrap();
    let scratch = device.alloc_zeroed::<f32>(4).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let prepared = device.prepare(&ScaleKernel).unwrap();

    let mut stream = device.stream().unwrap();
    stream.fill_zero(&scratch).unwrap();
    stream.copy(&input, &scratch).unwrap();
    stream
        .encode(
            &prepared,
            &[Binding::read(&scratch), Binding::read_write(&output)],
            &ScaleParams {
                len: 4,
                factor: 3.0,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap();
    stream.submit().unwrap();

    let mut host = [0.0f32; 4];
    device.download(&output, &mut host).unwrap();
    assert_eq!(host, [9.0, 12.0, 15.0, 18.0]);
}

fn command_stream_rejects_binding_contract_mismatch() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping binding contract test");
        return;
    };
    let input = device.upload(&[1.0f32, 2.0]).unwrap();
    let prepared = device.prepare(&ScaleKernel).unwrap();

    let err = device
        .dispatch(
            &prepared,
            &[Binding::read(&input)],
            &ScaleParams {
                len: 2,
                factor: 1.0,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap_err();

    match err {
        HephaestusError::DispatchFailed { message } => {
            assert!(message.contains("declares 2 storage bindings, got 1"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

fn command_stream_copy_rejects_length_mismatch() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping copy contract test");
        return;
    };
    let src = device.upload(&[1u32, 2, 3]).unwrap();
    let dst = device.alloc_zeroed::<u32>(2).unwrap();
    let mut stream = device.stream().unwrap();

    let err = stream.copy(&src, &dst).unwrap_err();

    assert!(matches!(
        err,
        HephaestusError::LengthMismatch {
            host_len: 3,
            device_len: 2
        }
    ));
}

fn command_stream_copy_prefix_preserves_the_destination_suffix() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping command stream prefix-copy test");
        return;
    };
    let src = device.upload(&[3u32, 5, 7, 11]).unwrap();
    let dst = device.alloc_zeroed::<u32>(4).unwrap();
    let mut stream = device.stream().unwrap();
    stream.copy_prefix(&src, &dst, 2).unwrap();
    stream.submit().unwrap();

    let mut host = [0u32; 4];
    device.download(&dst, &mut host).unwrap();
    assert_eq!(host, [3, 5, 0, 0]);
}

fn grouped_command_stream_dispatches_multi_group_wgsl_kernel() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping grouped command stream test");
        return;
    };
    let left = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let right = device.upload(&[10.0f32, 20.0, 30.0, 40.0]).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();

    device
        .dispatch_grouped(
            &prepared,
            &[
                GroupedBinding::read(0, 0, &left),
                GroupedBinding::read(1, 0, &right),
                GroupedBinding::read_write(1, 1, &output),
            ],
            &GroupedParams {
                len: 4,
                addend: 0.5,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap();

    let mut host = [0.0f32; 4];
    device.download(&output, &mut host).unwrap();
    assert_eq!(host, [11.5, 22.5, 33.5, 44.5]);
}

fn grouped_sequence_preserves_order_in_one_wgpu_pass() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping grouped sequence test");
        return;
    };
    let left = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let right = device.upload(&[10.0f32, 20.0, 30.0, 40.0]).unwrap();
    let scratch = device.alloc_zeroed::<f32>(4).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();

    let mut stream = device.grouped_stream().unwrap();
    stream
        .encode_grouped_sequence("hephaestus-wgpu-grouped-sequence", |sequence| {
            sequence.encode_grouped(
                &prepared,
                &[
                    GroupedBinding::read(0, 0, &left),
                    GroupedBinding::read(1, 0, &right),
                    GroupedBinding::read_write(1, 1, &scratch),
                ],
                &GroupedParams {
                    len: 4,
                    addend: 0.5,
                },
                DispatchGrid::new(1, 1, 1),
            )?;
            sequence.encode_grouped(
                &prepared,
                &[
                    GroupedBinding::read(0, 0, &scratch),
                    GroupedBinding::read(1, 0, &right),
                    GroupedBinding::read_write(1, 1, &output),
                ],
                &GroupedParams {
                    len: 4,
                    addend: 1.0,
                },
                DispatchGrid::new(1, 1, 1),
            )
        })
        .unwrap();
    stream.submit_grouped().unwrap();

    let mut host = [0.0f32; 4];
    device.download(&output, &mut host).unwrap();
    assert_eq!(host, [22.5, 43.5, 64.5, 85.5]);
}

fn bound_grouped_sequence_reuses_fixed_resources() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping bound grouped stream test");
        return;
    };
    let left = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let right = device.upload(&[10.0f32, 20.0, 30.0, 40.0]).unwrap();
    let scratch = device.alloc_zeroed::<f32>(4).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
    let first = device
        .bind_grouped_dispatch(
            &prepared,
            &[
                GroupedBinding::read(0, 0, &left),
                GroupedBinding::read(1, 0, &right),
                GroupedBinding::read_write(1, 1, &scratch),
            ],
            &GroupedParams {
                len: 4,
                addend: 0.5,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap();
    let second = device
        .bind_grouped_dispatch(
            &prepared,
            &[
                GroupedBinding::read(0, 0, &scratch),
                GroupedBinding::read(1, 0, &right),
                GroupedBinding::read_write(1, 1, &output),
            ],
            &GroupedParams {
                len: 4,
                addend: 1.0,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap();

    for (input, expected) in [
        ([1.0, 2.0, 3.0, 4.0], [22.5, 43.5, 64.5, 85.5]),
        ([2.0, 4.0, 6.0, 8.0], [23.5, 45.5, 67.5, 89.5]),
    ] {
        device.write_buffer(&left, &input).unwrap();
        let mut stream = device.grouped_stream().unwrap();
        stream
            .encode_grouped_sequence("hephaestus-wgpu-bound-grouped", |sequence| {
                first.encode_in_sequence(sequence)?;
                second.encode_in_sequence(sequence)
            })
            .unwrap();
        stream.submit_grouped().unwrap();
        let mut host = [0.0f32; 4];
        device.download(&output, &mut host).unwrap();
        assert_eq!(host, expected);
    }

    let other = try_device().expect("second WGPU device");
    let error = other
        .bind_grouped_dispatch(
            &prepared,
            &[
                GroupedBinding::read(0, 0, &left),
                GroupedBinding::read(1, 0, &right),
                GroupedBinding::read_write(1, 1, &output),
            ],
            &GroupedParams {
                len: 4,
                addend: 0.0,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .expect_err("bound grouped preparation must preserve device provenance");
    assert!(error.to_string().contains("different device"));
}

fn bound_grouped_dispatch_updates_retained_parameters() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping bound parameter-update test");
        return;
    };
    let left = device.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let right = device.upload(&[10.0f32, 20.0, 30.0, 40.0]).unwrap();
    let output = device.alloc_zeroed::<f32>(4).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
    let mut bound = device
        .bind_grouped_dispatch(
            &prepared,
            &[
                GroupedBinding::read(0, 0, &left),
                GroupedBinding::read(1, 0, &right),
                GroupedBinding::read_write(1, 1, &output),
            ],
            &GroupedParams {
                len: 4,
                addend: 0.5,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap();

    let execute = |bound: &WgpuBoundGroupedDispatch<GroupedAddKernel>| {
        let mut stream = device.grouped_stream().unwrap();
        stream
            .encode_grouped_sequence("hephaestus-wgpu-bound-parameter-update", |sequence| {
                bound.encode_in_sequence(sequence)
            })
            .unwrap();
        stream.submit_grouped().unwrap();
    };
    execute(&bound);
    let mut host = [0.0f32; 4];
    device.download(&output, &mut host).unwrap();
    assert_eq!(host, [11.5, 22.5, 33.5, 44.5]);

    bound
        .update_params(
            &device,
            &GroupedParams {
                len: 2,
                addend: -1.0,
            },
        )
        .unwrap();
    execute(&bound);
    device.download(&output, &mut host).unwrap();
    assert_eq!(host, [10.0, 21.0, 33.5, 44.5]);
}

fn bound_grouped_parameter_update_rejects_foreign_device() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping foreign parameter-update test");
        return;
    };
    let Some(other) = try_device() else {
        eprintln!("No second WGPU device available; skipping foreign parameter-update test");
        return;
    };
    let left = device.upload(&[1.0f32]).unwrap();
    let right = device.upload(&[2.0f32]).unwrap();
    let output = device.alloc_zeroed::<f32>(1).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
    let mut bound = device
        .bind_grouped_dispatch(
            &prepared,
            &[
                GroupedBinding::read(0, 0, &left),
                GroupedBinding::read(1, 0, &right),
                GroupedBinding::read_write(1, 1, &output),
            ],
            &GroupedParams {
                len: 1,
                addend: 0.0,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap();

    let error = bound
        .update_params(
            &other,
            &GroupedParams {
                len: 1,
                addend: 9.0,
            },
        )
        .expect_err("parameter update must reject a foreign device");
    match error {
        HephaestusError::DispatchFailed { message } => assert_eq!(
            message,
            "prepared WGPU bound grouped dispatch belongs to a different device"
        ),
        other => panic!("unexpected error: {other:?}"),
    }
}

fn bound_grouped_dispatch_rejects_invalid_parameter_size() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping bound parameter-size test");
        return;
    };
    let left = device.upload(&[1.0f32]).unwrap();
    let right = device.upload(&[2.0f32]).unwrap();
    let output = device.alloc_zeroed::<f32>(1).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
    let oversized = WgpuGroupedPrepared::<OversizedGroupedKernel> {
        pipeline: prepared.pipeline.clone(),
        bind_group_layouts: prepared.bind_group_layouts.clone(),
        parameter_group: prepared.parameter_group,
        parameter_binding: prepared.parameter_binding,
        owner: prepared.owner.clone(),
        label: OversizedGroupedKernel::LABEL,
        marker: PhantomData,
    };

    let error = device
        .bind_grouped_dispatch(
            &oversized,
            &[
                GroupedBinding::read(0, 0, &left),
                GroupedBinding::read(1, 0, &right),
                GroupedBinding::read_write(1, 1, &output),
            ],
            &OVERSIZED_PARAMETERS,
            DispatchGrid::new(1, 1, 1),
        )
        .expect_err("oversized retained parameters must fail before WGPU allocation");

    match error {
        HephaestusError::AllocationFailed { message } => {
            assert!(message.contains("65540 bytes"));
            assert!(message.contains("max_uniform_buffer_binding_size=65536"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

fn bound_grouped_sequence_rejects_foreign_device() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping foreign bound-sequence test");
        return;
    };
    let Some(other) = try_device() else {
        eprintln!("No second WGPU device available; skipping foreign bound-sequence test");
        return;
    };
    let left = device.upload(&[1.0f32]).unwrap();
    let right = device.upload(&[2.0f32]).unwrap();
    let output = device.alloc_zeroed::<f32>(1).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
    let bound = device
        .bind_grouped_dispatch(
            &prepared,
            &[
                GroupedBinding::read(0, 0, &left),
                GroupedBinding::read(1, 0, &right),
                GroupedBinding::read_write(1, 1, &output),
            ],
            &GroupedParams {
                len: 1,
                addend: 0.0,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap();
    let mut stream = other.grouped_stream().unwrap();

    let error = stream
        .encode_grouped_sequence("foreign-bound-grouped-sequence", |sequence| {
            bound.encode_in_sequence(sequence)
        })
        .expect_err("bound dispatch must reject a foreign sequence");

    assert_eq!(
        error.to_string(),
        "kernel dispatch failed: prepared WGPU bound grouped dispatch belongs to a different device"
    );
    let mut host = [f32::NAN; 1];
    device.download(&output, &mut host).unwrap();
    assert_eq!(host, [0.0]);
}

fn bound_grouped_dispatch_rejects_group_mismatch() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping bound grouped mismatch test");
        return;
    };
    let left = device.upload(&[1.0f32]).unwrap();
    let right = device.upload(&[2.0f32]).unwrap();
    let output = device.alloc_zeroed::<f32>(1).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();

    let error = device
        .bind_grouped_dispatch(
            &prepared,
            &[
                GroupedBinding::read(0, 0, &left),
                GroupedBinding::read(0, 0, &right),
                GroupedBinding::read_write(1, 1, &output),
            ],
            &GroupedParams {
                len: 1,
                addend: 0.0,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .expect_err("bound grouped preparation must validate binding slots");

    assert_eq!(
        error.to_string(),
        "kernel dispatch failed: hephaestus-wgpu-grouped-add: binding 1 declared group 1 binding 0, bound group 0 binding 0"
    );
}

fn grouped_command_stream_rejects_group_mismatch() {
    let Some(device) = try_device() else {
        eprintln!("No WGPU adapter available; skipping grouped mismatch test");
        return;
    };
    let left = device.upload(&[1.0f32]).unwrap();
    let right = device.upload(&[2.0f32]).unwrap();
    let output = device.alloc_zeroed::<f32>(1).unwrap();
    let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();

    let err = device
        .dispatch_grouped(
            &prepared,
            &[
                GroupedBinding::read(0, 0, &left),
                GroupedBinding::read(0, 0, &right),
                GroupedBinding::read_write(1, 1, &output),
            ],
            &GroupedParams {
                len: 1,
                addend: 0.0,
            },
            DispatchGrid::new(1, 1, 1),
        )
        .unwrap_err();

    match err {
        HephaestusError::DispatchFailed { message } => {
            assert!(message.contains("declared group 1 binding 0"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

fn kernel_interface_declares_wgpu_parameter_binding_after_storage() {
    assert_eq!(ScaleKernel::BINDINGS.len(), 2);
    assert_eq!(ScaleKernel::BINDINGS[0].access, Access::ReadOnly);
    assert_eq!(ScaleKernel::BINDINGS[1].access, Access::ReadWrite);
}
