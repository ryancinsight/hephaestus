//! Capacity-query and preparation parity on a logical WGPU device.

use hephaestus_core::{
    ComputeDevice, FftDirection, FftLimit, FftOperands, FftOps, FftResource, HephaestusError,
    StridedView,
};
use leto::Layout;

use crate::{WgpuDevice, WgpuFftOps};

#[test]
fn preflight_and_preparation_agree_on_real_buffer_limits() {
    crate::test_support::run_cases(&[("fft_exact_device_capacity", exact_device_capacity as fn())]);
}

fn exact_device_capacity() {
    let shape = [17, 16, 16];
    let axes = [0, 1, 2];
    let layout = Layout::c_contiguous(shape).expect("dense FFT layout");
    let mut impulse = vec![0.0f32; shape.into_iter().product()];
    impulse[0] = 1.0;
    for max_buffer_size in [65_535, 65_536] {
        let limits = wgpu::Limits {
            max_buffer_size,
            ..wgpu::Limits::downlevel_defaults()
        };
        let device = match WgpuDevice::try_default_with_limits("fft-exact-capacity", limits) {
            Ok(device) => device,
            Err(HephaestusError::AdapterUnavailable { .. })
                if std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_none() =>
            {
                return;
            }
            Err(error) => panic!("FFT capacity test requires a WGPU device: {error}"),
        };
        let real = device
            .upload(&impulse)
            .expect("real operand fits the test device");
        let imaginary = device
            .alloc_zeroed(impulse.len())
            .expect("imaginary operand fits");
        let operands = || FftOperands {
            real: StridedView::new(&real, &layout),
            imaginary: StridedView::new(&imaginary, &layout),
        };
        let ops = WgpuFftOps;
        let requirements = ops.requirements::<f32, 3>(&device, shape, &axes);
        let prepared = ops.prepare_fft_axes(&device, operands(), FftDirection::Forward, &axes);
        if max_buffer_size == 65_535 {
            let refused = |error| {
                matches!(
                    error,
                    HephaestusError::FftCapacityExceeded {
                        resource: FftResource::Workspace,
                        required: 65_536,
                        limit: 65_535,
                        limiting_quantity: FftLimit::BufferBytes,
                    }
                )
            };
            assert!(requirements.is_err_and(refused));
            assert!(prepared.is_err_and(refused));
            assert!(
                ops.prepare_fft(&device, operands(), FftDirection::Forward)
                    .is_err_and(refused)
            );
        } else {
            assert_eq!(
                requirements.expect("inclusive byte cap").workspace_bytes,
                Some(65_536)
            );
            ops.dispatch_fft(&device, &prepared.expect("same inclusive byte cap"))
                .expect("forward FFT dispatch");
            let mut readback = vec![0.0; impulse.len()];
            device
                .download(&real, &mut readback)
                .expect("read computed FFT");
            // The independent impulse oracle is exactly one over the spectrum.
            // Carry the existing per-operation 128-ulp bound across the 64
            // terms of the padded Bluestein transform.
            let tolerance = 128.0 * 64.0 * f32::EPSILON;
            assert!(
                readback
                    .iter()
                    .all(|value| (value - 1.0).abs() <= tolerance)
            );
            device
                .download(&imaginary, &mut readback)
                .expect("read imaginary FFT");
            assert!(readback.iter().all(|value| value.abs() <= tolerance));
        }
    }
}
