//! Guards the removal of the Windows per-launch `cuCtxSynchronize` drain.
//!
//! The drain existed because WDDM does not support concurrent host/device
//! access to `cuMemAllocManaged` ranges: a host touchpoint issued while a
//! kernel was in flight faulted with STATUS_IN_PAGE_ERROR (0xc0000006). The
//! backend now allocates only `cuMemAlloc_v2`, so no managed range exists to
//! fault on, and the drain was removed on run evidence — it cost 4.9x on
//! launch throughput.
//!
//! This reproduces the exact cited scenario so the removal is defended by a
//! test rather than by a one-time measurement: host allocation, upload, and
//! free issued with a launch outstanding and no intervening synchronization.

#![cfg(feature = "cuda")]

use hephaestus_cuda::{
    AddOp, BlockWidth, ComputeDevice, CudaDevice, NegOp, binary_elementwise_into,
    unary_elementwise_into,
};

const LEN: usize = 1024;
const ROUNDS: usize = 1500;

#[test]
fn cuda_launch_survives_host_allocation_in_flight() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA in-flight host access: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA in-flight host access requires a physical device: {error}"),
    };

    let lhs_host: Vec<f32> = (0..LEN).map(|index| index as f32 * 1.0e-3).collect();
    let rhs_host: Vec<f32> = (0..LEN).map(|index| index as f32 * 2.0e-3).collect();
    let lhs = device.upload(&lhs_host).expect("upload lhs");
    let rhs = device.upload(&rhs_host).expect("upload rhs");
    let out = device.alloc_zeroed::<f32>(LEN).expect("allocate output");

    let mut retained = Vec::new();
    for round in 0..ROUNDS {
        binary_elementwise_into::<AddOp, f32>(&device, &lhs, &rhs, &out, BlockWidth::DEFAULT)
            .expect("launch");

        // No synchronization here: the launch is still outstanding on the null
        // stream while the host allocates, uploads, and frees. This is the
        // access pattern the drain was written to prevent.
        let scratch = device
            .alloc_zeroed::<f32>(4096 + round % 512)
            .expect("in-flight device allocation");
        let uploaded = device
            .upload(&[1.0f32; 256])
            .expect("in-flight host-to-device upload");

        if round % 8 == 0 {
            retained.push(scratch);
        } else {
            drop(scratch);
        }
        drop(uploaded);
        if retained.len() > 32 {
            retained.clear();
        }
    }

    // Value semantics: the kernel must still be computing, so a fault-free run
    // cannot be a run that silently did nothing.
    let mut actual = vec![0.0f32; LEN];
    device.download(&out, &mut actual).expect("download result");
    for index in [0usize, 1, 7, LEN / 2, LEN - 1] {
        let expected = lhs_host[index] + rhs_host[index];
        assert!(
            (actual[index] - expected).abs() <= 1.0e-6,
            "index {index}: {} != {expected}",
            actual[index]
        );
    }
}

/// The drop-soundness claim behind stream-ordered frees.
///
/// A buffer may still be referenced by a kernel in flight when it is dropped.
/// `cuMemFree_v2` made that safe by synchronizing the whole device;
/// `cuMemFreeAsync` on the null stream makes it safe by ordering the release
/// behind work already submitted to that stream. This drops the kernel's *input*
/// with the launch outstanding and then checks the output, so a free that took
/// effect early would show as corruption or a fault rather than passing quietly.
#[test]
fn cuda_input_buffer_may_drop_while_its_kernel_is_in_flight() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA in-flight buffer drop: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA in-flight buffer drop requires a physical device: {error}"),
    };

    // Large enough that the kernel is plausibly still running when the input is
    // dropped; the ordering guarantee is what must hold, not the timing.
    const BIG: usize = 1 << 20;
    const DROP_ROUNDS: usize = 200;

    let host: Vec<f32> = (0..BIG).map(|index| (index % 1000) as f32 * 0.5).collect();
    let out = device.alloc_zeroed::<f32>(BIG).expect("allocate output");

    for _ in 0..DROP_ROUNDS {
        let input = device.upload(&host).expect("upload input");
        unary_elementwise_into::<NegOp, f32>(&device, &input, &out, BlockWidth::DEFAULT)
            .expect("launch");
        // No synchronization: the kernel still references `input`'s device
        // pointer when this frees it.
        drop(input);
    }

    let mut actual = vec![0.0f32; BIG];
    device.download(&out, &mut actual).expect("download result");
    for index in [0usize, 1, 999, BIG / 2, BIG - 1] {
        let expected = -host[index];
        assert!(
            (actual[index] - expected).abs() <= 1.0e-6,
            "index {index}: {} != {expected}",
            actual[index]
        );
    }
}

/// The allocator and the free must be chosen by the same flag, or the driver
/// rejects the mismatched pair. This pins that structurally, mirroring
/// `device_local_copy_uses_one_synchronous_copy_without_context_barrier`.
#[test]
fn cuda_allocation_and_free_select_the_same_allocator() {
    let device_source = include_str!("../src/infrastructure/device.rs");
    let buffer_source = include_str!("../src/infrastructure/buffer.rs");

    // Allocation branches on the context flag and offers both allocators.
    assert!(device_source.contains("if self.context.stream_ordered"));
    assert_eq!(device_source.matches("cuMemAllocAsync(").count(), 1);
    assert_eq!(device_source.matches("cuMemAlloc_v2(").count(), 1);

    // The free branches on the same flag and offers exactly the matching pair.
    assert!(buffer_source.contains("if context.stream_ordered"));
    assert_eq!(buffer_source.matches("cuMemFreeAsync(").count(), 1);
    assert_eq!(buffer_source.matches("cuMemFree_v2(").count(), 1);

    // The capability is probed, never assumed.
    assert!(device_source.contains("CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED"));
}
