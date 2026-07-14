//! Empirical WGPU dispatch benchmark for allocating vs caller-owned output.
//!
//! This benchmark intentionally avoids synthetic mocks: it acquires a real
//! adapter, dispatches real kernels, waits for device completion, and validates
//! the final output values. It prints timing only; no speedup threshold is
//! claimed without a stored Criterion baseline.

use std::time::{Duration, Instant};

use hephaestus_core::BlockWidth;
use hephaestus_wgpu::{
    AddOp, ComputeDevice, WgpuDevice, binary_elementwise, binary_elementwise_into,
};

const LEN: usize = 1 << 20;
const ITERS: usize = 20;

fn wait(device: &WgpuDevice) {
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("invariant: benchmark device poll succeeds");
}

fn elapsed_per_iter(elapsed: Duration) -> Duration {
    elapsed / u32::try_from(ITERS).expect("invariant: benchmark iteration count fits u32")
}

fn main() {
    let device = match WgpuDevice::try_default("hephaestus-elementwise-bench") {
        Ok(device) => device,
        Err(e) => {
            eprintln!("skipping wgpu benchmark: {e}");
            return;
        }
    };

    let a_host: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.25).collect();
    let b_host: Vec<f32> = (0..LEN).map(|i| 1000.0 - i as f32 * 0.5).collect();
    let expected_tail = a_host[LEN - 1] + b_host[LEN - 1];
    let a = device.upload(&a_host).expect("upload a");
    let b = device.upload(&b_host).expect("upload b");
    let out = device.alloc_zeroed::<f32>(LEN).expect("allocate output");

    let warm = binary_elementwise::<AddOp, f32>(&device, &a, &b).expect("warm allocating dispatch");
    wait(&device);
    binary_elementwise_into::<AddOp, f32>(&device, &a, &b, &out, BlockWidth::DEFAULT)
        .expect("warm caller-owned dispatch");
    wait(&device);
    drop(warm);

    let allocating_start = Instant::now();
    for _ in 0..ITERS {
        let _result =
            binary_elementwise::<AddOp, f32>(&device, &a, &b).expect("allocating dispatch");
    }
    wait(&device);
    let allocating = allocating_start.elapsed();

    let caller_owned_start = Instant::now();
    for _ in 0..ITERS {
        binary_elementwise_into::<AddOp, f32>(&device, &a, &b, &out, BlockWidth::DEFAULT)
            .expect("caller-owned dispatch");
    }
    wait(&device);
    let caller_owned = caller_owned_start.elapsed();

    let mut tail = vec![0.0f32; LEN];
    device.download(&out, &mut tail).expect("download output");
    assert_eq!(tail[LEN - 1], expected_tail);

    println!("len={LEN} iters={ITERS}");
    println!(
        "allocating_total_ns={} allocating_per_iter_ns={}",
        allocating.as_nanos(),
        elapsed_per_iter(allocating).as_nanos()
    );
    println!(
        "caller_owned_total_ns={} caller_owned_per_iter_ns={}",
        caller_owned.as_nanos(),
        elapsed_per_iter(caller_owned).as_nanos()
    );
}
