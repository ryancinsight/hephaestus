mod backward;
mod forward;
mod validation;

use crate::WgpuDevice;

fn device_or_skip() -> Option<WgpuDevice> {
    match WgpuDevice::try_default("hephaestus-attention-test") {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping WGPU attention test: {error}");
            None
        }
    }
}

fn assert_close(actual: &[f32], expected: &[f32], operation_count: u16) {
    assert_eq!(actual.len(), expected.len());
    let roundoff = f32::from(operation_count) * f32::EPSILON;
    let gamma = roundoff / (1.0 - roundoff);
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let scale = expected.abs().max(1.0);
        let bound = 2.0 * gamma * scale;
        assert!(
            (actual - expected).abs() <= bound,
            "element {index}: actual {actual}, expected {expected}, derived bound {bound}"
        );
    }
}
