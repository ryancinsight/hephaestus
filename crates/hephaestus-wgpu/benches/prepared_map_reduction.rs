//! Criterion comparison of one-shot and prepared fused map reductions.
//!
//! Local evidence host (2026-07-21): Intel Core Ultra 9 285K and NVIDIA
//! GeForce RTX 5080, driver 610.47. Every timed iteration waits for device
//! completion; operation inputs and Criterion settings are identical within
//! each prepared/one-shot pair.

use criterion::{Criterion, criterion_group, criterion_main};
use hephaestus_wgpu::{
    ComputeDevice, StridedOperand, WgpuDevice, dot, norm_l2, prepare_dot, prepare_norm_l2,
};
use leto::Layout;
use std::hint::black_box;

const LEN: usize = 1 << 16;
const EXPECTED_DOT: f32 = 131_072.0;
const EXPECTED_NORM: f32 = 256.0;

fn wait(device: &WgpuDevice) {
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("invariant: benchmark device poll succeeds");
}

fn download_scalar(device: &WgpuDevice, buffer: &hephaestus_wgpu::WgpuBuffer<f32>) -> f32 {
    let mut value = [0.0f32; 1];
    device
        .download(buffer, &mut value)
        .expect("benchmark scalar download");
    value[0]
}

fn prepared_map_reduction(c: &mut Criterion) {
    let device = match WgpuDevice::try_default("hephaestus-prepared-map-reduction-bench") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping WGPU benchmark: {error}");
            return;
        }
    };
    let lhs_host = vec![1.0f32; LEN];
    let rhs_host = vec![2.0f32; LEN];
    let lhs = device.upload(&lhs_host).expect("upload lhs");
    let rhs = device.upload(&rhs_host).expect("upload rhs");
    let layout = Layout::c_contiguous([LEN]).expect("invariant: fixed shape is valid");
    let lhs_view = StridedOperand {
        buffer: &lhs,
        layout: &layout,
    };
    let rhs_view = StridedOperand {
        buffer: &rhs,
        layout: &layout,
    };
    let prepared_dot = prepare_dot(&device, lhs_view, rhs_view).expect("prepare dot");
    let prepared_norm = prepare_norm_l2(&device, lhs_view).expect("prepare L2 norm");

    prepared_dot.dispatch(&device).expect("warm prepared dot");
    prepared_norm.dispatch(&device).expect("warm prepared norm");
    wait(&device);
    assert_eq!(
        download_scalar(&device, prepared_dot.output()),
        EXPECTED_DOT
    );
    assert_eq!(
        download_scalar(&device, prepared_norm.output()),
        EXPECTED_NORM
    );

    let mut group = c.benchmark_group("map_reduction_dispatch");
    group.bench_function("dot_one_shot", |b| {
        b.iter(|| {
            let output = dot(&device, black_box(lhs_view), black_box(rhs_view)).expect("dot");
            wait(&device);
            black_box(output)
        });
    });
    group.bench_function("dot_prepared", |b| {
        b.iter(|| {
            prepared_dot.dispatch(&device).expect("prepared dot");
            wait(&device);
            black_box(prepared_dot.output().raw())
        });
    });
    group.bench_function("l2_one_shot", |b| {
        b.iter(|| {
            let output = norm_l2(&device, black_box(lhs_view)).expect("L2 norm");
            wait(&device);
            black_box(output)
        });
    });
    group.bench_function("l2_prepared", |b| {
        b.iter(|| {
            prepared_norm.dispatch(&device).expect("prepared L2 norm");
            wait(&device);
            black_box(prepared_norm.output().raw())
        });
    });
    group.finish();
}

criterion_group!(benches, prepared_map_reduction);
criterion_main!(benches);
