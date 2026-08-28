use hephaestus_core::{ComputeDevice, FftDirection, FftOperands, FftOps, StridedView};
use hephaestus_wgpu::{HephaestusError, WgpuDevice, WgpuFftOps};
use leto::Layout;

use super::device_or_skip;

pub(super) fn prepared_fft_device_preflight_is_public_and_typed() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let other = WgpuDevice::try_default("hephaestus-fft-public-preflight")
        .expect("a second logical WGPU device must be acquired");
    let real = device.upload(&[1.0, 2.0]).expect("real upload");
    let imaginary = device.upload(&[0.0, 0.0]).expect("imaginary upload");
    let layout = Layout::c_contiguous([2]).expect("dense layout");
    let prepared = WgpuFftOps
        .prepare_fft(
            &device,
            FftOperands {
                real: StridedView::new(&real, &layout),
                imaginary: StridedView::new(&imaginary, &layout),
            },
            FftDirection::Forward,
        )
        .expect("FFT preparation");

    let error = prepared
        .validate_device(&other)
        .expect_err("foreign public preflight must fail");
    let HephaestusError::DispatchFailed { message } = error else {
        panic!("foreign public preflight returned the wrong error variant")
    };
    assert_eq!(message, "prepared WGPU FFT belongs to a different device");
}
