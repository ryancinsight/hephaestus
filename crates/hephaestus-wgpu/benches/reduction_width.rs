//! Empirical WGPU reduction benchmark for default vs caller-selected width.
//!
//! This benchmark uses a real adapter, dispatches real reduction kernels, waits
//! for completion, and validates both outputs against a host-side exact `u32`
//! sum. It prints timing only; no speedup threshold is claimed without a
//! stored Criterion baseline.

use std::time::{Duration, Instant};

use hephaestus_core::BlockWidth;
use hephaestus_wgpu::{reduction, reduction_with_width, ComputeDevice, SumOp, WgpuDevice};

const LEN: usize = 1 << 16;
const ITERS: usize = 20;

fn wait(device: &WgpuDevice) {
    device
        .inner()
        .poll(wgpu::PollType::Wait)
        .expect("invariant: benchmark device poll succeeds");
}

fn elapsed_per_iter(elapsed: Duration) -> Duration {
    elapsed / u32::try_from(ITERS).expect("invariant: benchmark iteration count fits u32")
}

fn download_one(device: &WgpuDevice, buffer: &hephaestus_wgpu::WgpuBuffer<u32>) -> u32 {
    let mut out = [0u32; 1];
    device
        .download(buffer, &mut out)
        .expect("download reduction");
    out[0]
}

fn main() {
    let device = match WgpuDevice::try_default("hephaestus-reduction-bench") {
        Ok(device) => device,
        Err(e) => {
            eprintln!("skipping wgpu benchmark: {e}");
            return;
        }
    };

    let host: Vec<u32> = (0..u32::try_from(LEN).expect("invariant: LEN fits u32")).collect();
    let expected: u32 = host.iter().sum();
    let input = device.upload(&host).expect("upload input");
    let narrow = BlockWidth::new(128).expect("invariant: benchmark width is non-zero");

    let warm_default = reduction::<SumOp, u32>(&device, &input).expect("warm default reduction");
    wait(&device);
    let warm_narrow =
        reduction_with_width::<SumOp, u32>(&device, &input, narrow).expect("warm width reduction");
    wait(&device);
    assert_eq!(download_one(&device, &warm_default), expected);
    assert_eq!(download_one(&device, &warm_narrow), expected);

    let default_start = Instant::now();
    for _ in 0..ITERS {
        let _result = reduction::<SumOp, u32>(&device, &input).expect("default reduction");
    }
    wait(&device);
    let default = default_start.elapsed();

    let narrow_start = Instant::now();
    for _ in 0..ITERS {
        let _result =
            reduction_with_width::<SumOp, u32>(&device, &input, narrow).expect("width reduction");
    }
    wait(&device);
    let narrow_elapsed = narrow_start.elapsed();

    println!("len={LEN} iters={ITERS}");
    println!(
        "default_total_ns={} default_per_iter_ns={}",
        default.as_nanos(),
        elapsed_per_iter(default).as_nanos()
    );
    println!(
        "width128_total_ns={} width128_per_iter_ns={}",
        narrow_elapsed.as_nanos(),
        elapsed_per_iter(narrow_elapsed).as_nanos()
    );
}
