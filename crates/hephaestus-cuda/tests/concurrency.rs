//! Diagnoses intra-process concurrency safety of CUDA device acquisition.
//!
//! Spawns many threads that each acquire a `CudaDevice` and run a small
//! transfer concurrently. If acquisition is not safe under concurrency this
//! faults (0xc0000005) or returns errors; a clean pass means intra-process
//! acquisition is serialized/shared correctly. Skips without a CUDA device.

use hephaestus_core::ComputeDevice;
use hephaestus_cuda::CudaDevice;

#[test]
fn concurrent_device_acquisition_is_safe() {
    if CudaDevice::try_default().is_err() {
        eprintln!("skip concurrent_device_acquisition_is_safe: no CUDA device");
        return;
    }

    const THREADS: usize = 16;
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            std::thread::spawn(move || {
                // Each thread independently acquires the device and round-trips
                // a buffer — exercises concurrent acquisition + transfer.
                let dev = CudaDevice::try_default().expect("acquire device on worker thread");
                let host: Vec<f32> = (0..256).map(|i| (i + t) as f32).collect();
                let buf = dev.upload(&host).expect("upload");
                let mut out = vec![0.0f32; host.len()];
                dev.download(&buf, &mut out).expect("download");
                assert_eq!(out, host, "round-trip identity on thread {t}");
            })
        })
        .collect();

    for h in handles {
        h.join().expect("worker thread must not panic or fault");
    }
}
