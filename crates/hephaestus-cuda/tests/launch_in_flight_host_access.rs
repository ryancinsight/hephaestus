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

use hephaestus_cuda::{AddOp, BlockWidth, ComputeDevice, CudaDevice, binary_elementwise_into};

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
