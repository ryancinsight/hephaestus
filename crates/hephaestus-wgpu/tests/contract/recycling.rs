use super::device_or_skip;
use hephaestus_core::{ComputeDevice, Result};
use hephaestus_wgpu::{AddOp, WgpuDevice, scalar_elementwise};

#[test]
fn transient_reuse_preserves_each_devices_values() -> Result<()> {
    let Some(first) = device_or_skip() else {
        return Ok(());
    };
    let second = WgpuDevice::try_default("second transient device")?;
    let sources = [
        first.upload(&[7.0_f32, 11.0])?,
        second.upload(&[31.0_f32, 47.0])?,
    ];
    // Each pass reuses the preceding pass's staging and uniform allocations;
    // distinct device inputs and changing scalars expose cross-device routing.
    for scalar in [3.0, 5.0, -2.0] {
        for (device, source, values) in [
            (&first, &sources[0], [7.0, 11.0]),
            (&second, &sources[1], [31.0, 47.0]),
        ] {
            let output = scalar_elementwise::<AddOp, f32>(device, source, scalar)?;
            assert_eq!(
                device.download_owned(&output)?,
                values.map(|value| value + scalar)
            );
            assert_eq!(device.download_owned(source)?, values);
        }
    }
    Ok(())
}
