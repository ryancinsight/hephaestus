//! Synchronization-floor profiling for hybrid blocked decomposition paths.
//!
//! The blocked LU/QR algorithms already have end-to-end comparative benchmark
//! rows. This harness isolates the host/device transfer pattern those paths
//! impose at the measured benchmark shapes, so follow-up kernel work targets
//! the synchronization component rather than guessing from total time.

use std::hint::black_box;
use std::time::{Duration, Instant};

use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use hephaestus_wgpu::WgpuDevice;

const ITERS: usize = 100;
const QR_REFLECTORS: usize = 32;

fn wait_wgpu(device: &WgpuDevice) {
    device
        .inner()
        .poll(wgpu::PollType::Wait)
        .expect("invariant: benchmark device poll succeeds");
}

fn elapsed_per_iter(elapsed: Duration) -> Duration {
    elapsed / u32::try_from(ITERS).expect("invariant: iteration count fits u32")
}

fn assert_close_slice(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (index, (&got, &expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - expected).abs() <= f32::EPSILON,
            "sync profile mismatch at {index}: got {got}, expected {expected}"
        );
    }
}

fn profile_blocked_lu_sync(device: &WgpuDevice) {
    let n = 66usize;
    let len = n * n;
    let mut host = vec![0.0f32; len];
    for row in 0..n {
        for col in 0..n {
            host[row * n + col] = if row == col {
                n as f32 + 4.0
            } else {
                0.1 / (1.0 + row.abs_diff(col) as f32)
            };
        }
    }

    let buffer = device.upload(&host).expect("upload LU profile matrix");
    let mut out = vec![0.0f32; len];
    let l21 = vec![0.25f32; 2 * 64];
    let u12 = vec![0.125f32; 64 * 2];
    let trail = vec![0.5f32; 4];
    let trail_buf = device.upload(&trail).expect("upload LU trailing tile");
    let mut trail_out = vec![0.0f32; 4];

    let start = Instant::now();
    for _ in 0..ITERS {
        device
            .download(&buffer, &mut out)
            .expect("download LU input");
        assert_close_slice(&out, &host);

        device
            .write_buffer(&buffer, &host)
            .expect("write LU panel state");
        let l21_buf = device.upload(black_box(&l21)).expect("upload LU L21");
        let u12_buf = device.upload(black_box(&u12)).expect("upload LU U12");
        black_box((&l21_buf, &u12_buf));

        device
            .download(&trail_buf, &mut trail_out)
            .expect("download LU trailing tile");
        assert_close_slice(&trail_out, &trail);

        device
            .write_buffer(&trail_buf, &trail)
            .expect("write LU final panel tile");
    }
    wait_wgpu(device);

    println!(
        "Blocked LU 66x66 sync floor: {} ns/iter",
        elapsed_per_iter(start.elapsed()).as_nanos()
    );
}

fn profile_blocked_qr_sync(device: &WgpuDevice) {
    let rows = 70usize;
    let cols = 35usize;
    let len = rows * cols;
    let mut host = vec![0.0f32; len];
    for row in 0..rows {
        for col in 0..cols {
            host[row * cols + col] = if row == col {
                5.0
            } else {
                0.01 / (1.0 + row.abs_diff(col) as f32)
            };
        }
    }

    let buffer = device.upload(&host).expect("upload QR profile matrix");
    let mut out = vec![0.0f32; len];
    let trail_cols = 3usize;
    let trailing = vec![0.25f32; rows * trail_cols];
    let trailing_buf = device
        .upload(&trailing)
        .expect("upload QR trailing columns");
    let mut trailing_out = vec![0.0f32; trailing.len()];
    let vectors: Vec<Vec<f32>> = (0..QR_REFLECTORS)
        .map(|j| vec![1.0f32 / (1.0 + j as f32); rows - j])
        .collect();
    let packed_vectors: Vec<f32> = vectors.iter().flatten().copied().collect();

    let start = Instant::now();
    for _ in 0..ITERS {
        device
            .download(&buffer, &mut out)
            .expect("download QR input");
        assert_close_slice(&out, &host);

        device
            .write_buffer(&trailing_buf, &trailing)
            .expect("write QR trailing columns");
        let uploaded_vectors = device
            .upload(black_box(&packed_vectors))
            .expect("upload packed QR reflectors");
        black_box(&uploaded_vectors);

        device
            .download(&trailing_buf, &mut trailing_out)
            .expect("download QR trailing columns");
        assert_close_slice(&trailing_out, &trailing);
    }
    wait_wgpu(device);

    println!(
        "Blocked QR 70x35 sync floor: {} ns/iter",
        elapsed_per_iter(start.elapsed()).as_nanos()
    );
}

fn profile_blocked_qr_timestamp_queries() {
    let device = match WgpuDevice::try_default_with_features_and_limits(
        "hephaestus-qr-timestamp-profile",
        wgpu::Features::TIMESTAMP_QUERY,
        wgpu::Limits::downlevel_defaults(),
    ) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("Skipping QR timestamp profile: timestamp queries unavailable: {error}");
            return;
        }
    };
    match profile_blocked_qr_launch_timestamps(&device) {
        Ok(profile) => {
            println!(
                "Blocked QR {QR_REFLECTORS}-reflector timestamp launch total: {:.1} ns",
                profile.total_ns
            );
            println!(
                "Blocked QR reflector timestamp launch median: {:.1} ns",
                profile.median_ns
            );
        }
        Err(error) => {
            eprintln!("Skipping QR timestamp profile: {error}");
        }
    }
}

struct TimestampProfile {
    total_ns: f64,
    median_ns: f64,
}

fn profile_blocked_qr_launch_timestamps(device: &WgpuDevice) -> Result<TimestampProfile> {
    let timestamp_period = f64::from(device.queue().get_timestamp_period());
    if timestamp_period == 0.0 {
        return Err(HephaestusError::DispatchFailed {
            message: "timestamp period is zero; timestamp queries are unsupported".to_string(),
        });
    }

    let query_count =
        u32::try_from(QR_REFLECTORS * 2).expect("invariant: timestamp query count fits u32");
    let query_set = device.inner().create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("hephaestus-qr-launch-timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: query_count,
    });
    let query_bytes = u64::from(query_count)
        * u64::try_from(std::mem::size_of::<u64>()).expect("invariant: u64 byte width fits u64");
    let resolve = device.inner().create_buffer(&wgpu::BufferDescriptor {
        label: Some("hephaestus-qr-launch-timestamp-resolve"),
        size: query_bytes,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.inner().create_buffer(&wgpu::BufferDescriptor {
        label: Some("hephaestus-qr-launch-timestamp-staging"),
        size: query_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let module = device
        .inner()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hephaestus-qr-launch-noop"),
            source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
        });
    let pipeline = device
        .inner()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hephaestus-qr-launch-noop"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-qr-launch-timestamps"),
        });
    for reflector in 0..QR_REFLECTORS {
        let start = u32::try_from(reflector * 2).expect("invariant: query index fits u32");
        let end = start + 1;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hephaestus-qr-launch-timestamp-pass"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &query_set,
                    beginning_of_pass_write_index: Some(start),
                    end_of_pass_write_index: Some(end),
                }),
            });
            pass.set_pipeline(&pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
    encoder.resolve_query_set(&query_set, 0..query_count, &resolve, 0);
    encoder.copy_buffer_to_buffer(&resolve, 0, &staging, 0, query_bytes);
    device.queue().submit(Some(encoder.finish()));

    let slice = staging.slice(..query_bytes);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .inner()
        .poll(wgpu::PollType::Wait)
        .map_err(|e| HephaestusError::TransferFailed {
            message: format!("device poll failed: {e:?}"),
        })?;
    receiver
        .recv()
        .map_err(|_| HephaestusError::TransferFailed {
            message: "timestamp map_async callback dropped".to_string(),
        })?
        .map_err(|e| HephaestusError::TransferFailed {
            message: format!("timestamp buffer mapping failed: {e:?}"),
        })?;

    let mapped = slice.get_mapped_range();
    let timestamps: &[u64] = bytemuck::cast_slice(&mapped);
    let mut durations: Vec<f64> = timestamps
        .chunks_exact(2)
        .map(|pair| pair[1].saturating_sub(pair[0]) as f64 * timestamp_period)
        .collect();
    drop(mapped);
    staging.unmap();

    durations.sort_by(f64::total_cmp);
    let total_ns = durations.iter().sum();
    let median_ns = durations[durations.len() / 2];
    Ok(TimestampProfile {
        total_ns,
        median_ns,
    })
}

fn main() {
    let device = match WgpuDevice::try_default("hephaestus-decomposition-sync-bench") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("Skipping decomposition sync profile: WGPU device unavailable: {error}");
            return;
        }
    };

    println!("=== Hybrid decomposition synchronization profile ===");
    println!("Iterations: {ITERS}");
    println!("WGPU GPU Backend: {}", device.backend_name());
    profile_blocked_lu_sync(&device);
    profile_blocked_qr_sync(&device);
    profile_blocked_qr_timestamp_queries();
}
