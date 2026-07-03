//! Empirical WGPU axis-scan benchmark: cumsum along axis 1 of a 512x4096 f32
//! matrix.
//!
//! This benchmark uses a real adapter, dispatches real scan kernels, waits for
//! completion, and validates the output against a host-side sequential prefix
//! sum (bitwise-equal: the kernel accumulates in the same left-to-right
//! order). It prints timing only; no speedup threshold is claimed without a
//! stored baseline.

use std::time::{Duration, Instant};

use hephaestus_core::BlockWidth;
use hephaestus_wgpu::{cumsum_into, ComputeDevice, StridedOperand, WgpuDevice};
use leto::Layout;

const ROWS: usize = 512;
const COLS: usize = 4096;
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

fn host_cumsum_axis1(input: &[f32]) -> Vec<f32> {
    let mut expected = vec![0.0f32; input.len()];
    for row in 0..ROWS {
        let mut acc = 0.0f32;
        for col in 0..COLS {
            acc += input[row * COLS + col];
            expected[row * COLS + col] = acc;
        }
    }
    expected
}

fn main() {
    let device = match WgpuDevice::try_default("hephaestus-scan-bench") {
        Ok(device) => device,
        Err(e) => {
            eprintln!("skipping wgpu benchmark: {e}");
            return;
        }
    };

    // Values are quarter-integers so the correctness oracle is bitwise
    // identity of the left-to-right addition order, not an epsilon bound.
    let host: Vec<f32> = (0..ROWS * COLS)
        .map(|i| f32::from(u8::try_from(i % 7).expect("invariant: i % 7 < 7")) * 0.25)
        .collect();
    let expected = host_cumsum_axis1(&host);

    let layout = Layout::c_contiguous([ROWS, COLS]).expect("invariant: benchmark layout is valid");
    let input = device.upload(&host).expect("upload input");
    let output = device
        .alloc_zeroed::<f32>(ROWS * COLS)
        .expect("alloc output");
    let input_operand = StridedOperand {
        buffer: &input,
        layout: &layout,
    };
    let output_operand = StridedOperand {
        buffer: &output,
        layout: &layout,
    };

    cumsum_into(
        &device,
        input_operand,
        1,
        output_operand,
        BlockWidth::DEFAULT,
    )
    .expect("warm cumsum");
    wait(&device);
    let mut got = vec![0.0f32; ROWS * COLS];
    device.download(&output, &mut got).expect("download cumsum");
    assert_eq!(got, expected, "cumsum output diverges from host reference");

    let start = Instant::now();
    for _ in 0..ITERS {
        cumsum_into(
            &device,
            input_operand,
            1,
            output_operand,
            BlockWidth::DEFAULT,
        )
        .expect("cumsum");
    }
    wait(&device);
    let elapsed = start.elapsed();

    println!("rows={ROWS} cols={COLS} iters={ITERS}");
    println!(
        "cumsum_axis1_total_ns={} cumsum_axis1_per_iter_ns={}",
        elapsed.as_nanos(),
        elapsed_per_iter(elapsed).as_nanos()
    );
}
