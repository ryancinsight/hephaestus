//! Test module for this operation family, split out of the parent
//! source file to keep it under the 500-line conformance budget.

    use super::*;
    use hephaestus_core::{
        BindingDecl, ComputeDevice, GroupedBindingDecl, GroupedKernelInterface, KernelInterface,
    };
    use std::borrow::Cow;

    #[repr(C)]
    #[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
    struct ScaleParams {
        len: u32,
        factor: f32,
    }

    #[derive(Clone, Copy, Debug)]
    struct ScaleKernel;

    impl KernelInterface for ScaleKernel {
        type Params = ScaleParams;

        const LABEL: &'static str = "hephaestus-cuda-stream-scale";
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

    #[derive(Clone, Copy, Debug)]
    struct GroupedAddKernel;

    impl GroupedKernelInterface for GroupedAddKernel {
        type Params = GroupedParams;

        const LABEL: &'static str = "hephaestus-cuda-grouped-add";
        const BINDINGS: &'static [GroupedBindingDecl] = &[
            GroupedBindingDecl::read_only::<f32>(0, 0),
            GroupedBindingDecl::read_only::<f32>(1, 0),
            GroupedBindingDecl::read_write::<f32>(1, 1),
        ];
        const PARAM_GROUP: u32 = 0;
        const PARAM_BINDING: u32 = 1;
        const WORKGROUP: [u32; 3] = [64, 1, 1];
    }

    impl GroupedKernelSource<CudaC> for GroupedAddKernel {
        const ENTRY: &'static str = "grouped_add_kernel";

        fn source(&self) -> Cow<'static, str> {
            Cow::Borrowed(
                r#"
struct GroupedParams {
    unsigned int len;
    float addend;
};

extern "C" __global__ void grouped_add_kernel(
    const float* left,
    const float* right,
    float* output,
    GroupedParams params
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < params.len) {
        output[idx] = left[idx] + right[idx] + params.addend;
    }
}
"#,
            )
        }
    }

    impl KernelSource<CudaC> for ScaleKernel {
        const ENTRY: &'static str = "scale_kernel";

        fn source(&self) -> Cow<'static, str> {
            Cow::Borrowed(
                r#"
struct ScaleParams {
    unsigned int len;
    float factor;
};

extern "C" __global__ void scale_kernel(
    const float* input,
    float* output,
    ScaleParams params
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < params.len) {
        output[idx] = input[idx] * params.factor;
    }
}
"#,
            )
        }
    }

    #[test]
    fn launch_scratch_reuses_capacity_for_steady_state_encodes() {
        let mut scratch = CudaLaunchScratch::default();
        let mut params = 7_u32;

        assert_eq!(
            scratch
                .prepare([1_u64 as DevicePtr, 2_u64 as DevicePtr], &mut params)
                .len(),
            3
        );
        scratch.prepare(
            [
                1_u64 as DevicePtr,
                2_u64 as DevicePtr,
                3_u64 as DevicePtr,
                4_u64 as DevicePtr,
            ],
            &mut params,
        );
        let pointer_capacity = scratch.device_ptrs.capacity();
        let argument_capacity = scratch.args.capacity();

        assert_eq!(
            scratch
                .prepare([1_u64 as DevicePtr, 2_u64 as DevicePtr], &mut params)
                .len(),
            3
        );
        assert_eq!(scratch.device_ptrs.capacity(), pointer_capacity);
        assert_eq!(scratch.args.capacity(), argument_capacity);
    }

    #[test]
    fn cuda_command_stream_dispatches_prepared_kernel_when_available() {
        let Ok(device) = CudaDevice::try_default() else {
            eprintln!("CUDA device unavailable, skipping command stream test");
            return;
        };

        let input = device.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let output = device.alloc_zeroed::<f32>(4).unwrap();
        let prepared = device.prepare(&ScaleKernel).unwrap();
        let bindings = [Binding::read(&input), Binding::read_write(&output)];
        let mut stream = device.stream().unwrap();
        stream
            .encode(
                &prepared,
                &bindings,
                &ScaleParams {
                    len: 4,
                    factor: 2.5,
                },
                DispatchGrid::new(1, 1, 1),
            )
            .unwrap();
        stream.submit().unwrap();
        device.synchronize().unwrap();

        let mut out = [0.0_f32; 4];
        device.download(&output, &mut out).unwrap();
        assert_eq!(out, [2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn cuda_command_stream_preserves_fill_copy_dispatch_order_when_available() {
        let Ok(device) = CudaDevice::try_default() else {
            eprintln!("CUDA device unavailable, skipping command stream order test");
            return;
        };

        let input = device.upload(&[2.0_f32, 4.0, 6.0, 8.0]).unwrap();
        let scratch = device.upload(&[9.0_f32, 9.0, 9.0, 9.0]).unwrap();
        let output = device.alloc_zeroed::<f32>(4).unwrap();
        let prepared = device.prepare(&ScaleKernel).unwrap();
        let mut stream = device.stream().unwrap();
        stream.fill_zero(&scratch).unwrap();
        stream.copy(&input, &scratch).unwrap();
        let bindings = [Binding::read(&scratch), Binding::read_write(&output)];
        stream
            .encode(
                &prepared,
                &bindings,
                &ScaleParams {
                    len: 4,
                    factor: 3.0,
                },
                DispatchGrid::new(1, 1, 1),
            )
            .unwrap();
        stream.submit().unwrap();
        device.synchronize().unwrap();

        let mut out = [0.0_f32; 4];
        device.download(&output, &mut out).unwrap();
        assert_eq!(out, [6.0, 12.0, 18.0, 24.0]);
    }

    #[test]
    fn cuda_command_stream_rejects_binding_contract_mismatch_when_available() {
        let Ok(device) = CudaDevice::try_default() else {
            eprintln!("CUDA device unavailable, skipping binding mismatch test");
            return;
        };

        let input = device.upload(&[1.0_f32]).unwrap();
        let output = device.alloc_zeroed::<f32>(1).unwrap();
        let prepared = device.prepare(&ScaleKernel).unwrap();
        let bindings = [Binding::read_write(&input), Binding::read_write(&output)];
        let mut stream = device.stream().unwrap();
        let err = stream
            .encode(
                &prepared,
                &bindings,
                &ScaleParams {
                    len: 1,
                    factor: 1.0,
                },
                DispatchGrid::new(1, 1, 1),
            )
            .unwrap_err();
        assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
    }

    #[test]
    fn cuda_grouped_command_stream_dispatches_kernel_when_available() {
        let Ok(device) = CudaDevice::try_default() else {
            eprintln!("CUDA device unavailable, skipping grouped command stream test");
            return;
        };

        let left = device.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let right = device.upload(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();
        let output = device.alloc_zeroed::<f32>(4).unwrap();
        let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
        let bindings = [
            GroupedBinding::read(0, 0, &left),
            GroupedBinding::read(1, 0, &right),
            GroupedBinding::read_write(1, 1, &output),
        ];
        let mut stream = device.grouped_stream().unwrap();
        stream
            .encode_grouped(
                &prepared,
                &bindings,
                &GroupedParams {
                    len: 4,
                    addend: 0.5,
                },
                DispatchGrid::new(1, 1, 1),
            )
            .unwrap();
        stream.submit_grouped().unwrap();
        device.synchronize().unwrap();

        let mut out = [0.0_f32; 4];
        device.download(&output, &mut out).unwrap();
        assert_eq!(out, [11.5, 22.5, 33.5, 44.5]);
    }

    #[test]
    fn cuda_grouped_command_stream_rejects_group_mismatch_when_available() {
        let Ok(device) = CudaDevice::try_default() else {
            eprintln!("CUDA device unavailable, skipping grouped mismatch test");
            return;
        };

        let left = device.upload(&[1.0_f32]).unwrap();
        let right = device.upload(&[2.0_f32]).unwrap();
        let output = device.alloc_zeroed::<f32>(1).unwrap();
        let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();
        let bindings = [
            GroupedBinding::read(0, 0, &left),
            GroupedBinding::read(0, 0, &right),
            GroupedBinding::read_write(1, 1, &output),
        ];
        let mut stream = device.grouped_stream().unwrap();
        let err = stream
            .encode_grouped(
                &prepared,
                &bindings,
                &GroupedParams {
                    len: 1,
                    addend: 0.0,
                },
                DispatchGrid::new(1, 1, 1),
            )
            .unwrap_err();
        assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
    }

    #[test]
    fn cuda_grouped_sequence_preserves_order_when_available() {
        let Ok(device) = CudaDevice::try_default() else {
            eprintln!("CUDA device unavailable, skipping grouped sequence test");
            return;
        };

        let left = device.upload(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let right = device.upload(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();
        let scratch = device.alloc_zeroed::<f32>(4).unwrap();
        let output = device.alloc_zeroed::<f32>(4).unwrap();
        let prepared = device.prepare_grouped(&GroupedAddKernel).unwrap();

        let mut stream = device.grouped_stream().unwrap();
        stream
            .encode_grouped_sequence("hephaestus-cuda-grouped-sequence", |sequence| {
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
        device.synchronize().unwrap();

        let mut out = [0.0_f32; 4];
        device.download(&output, &mut out).unwrap();
        assert_eq!(out, [22.5, 43.5, 64.5, 85.5]);
    }
