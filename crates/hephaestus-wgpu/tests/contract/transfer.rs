use super::device_or_skip;
use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use hephaestus_wgpu::WgpuDevice;

#[test]
fn transfer_writes_reject_foreign_buffers_without_mutation() -> Result<()> {
    let Some(owner) = device_or_skip() else {
        return Ok(());
    };
    let writer = WgpuDevice::try_default("foreign transfer writer")?;
    let sentinel = [17_u32, 23, 31, 47];
    let destination = owner.upload(&sentinel)?;
    for outcome in [
        writer.write_buffer(&destination, &[2, 3, 5, 7]),
        writer.write_sub_buffer(&destination, 1, &[11, 13]),
    ] {
        match outcome {
            Err(HephaestusError::DispatchFailed { message }) => {
                assert!(message.contains("device"), "{message}");
            }
            other => panic!("expected a device ownership error, got {other:?}"),
        }
        assert_eq!(owner.download_owned(&destination)?, sentinel);
    }
    owner.write_buffer(&destination, &[2, 3, 5, 7])?;
    owner.write_sub_buffer(&destination, 1, &[11, 13])?;
    assert_eq!(owner.download_owned(&destination)?, [2, 11, 13, 7]);
    Ok(())
}

#[test]
fn transfer_write_captures_destroyed_buffer_error() -> Result<()> {
    let Some(device) = device_or_skip() else {
        return Ok(());
    };
    let destroyed = device.upload(&[17_u32, 23])?;
    destroyed.raw().destroy();
    for outcome in [
        device.write_buffer(&destroyed, &[2, 3]),
        device.write_sub_buffer(&destroyed, 1, &[5]),
    ] {
        match outcome {
            Err(HephaestusError::DispatchFailed { message }) => {
                assert!(message.contains("hephaestus-buffer-write"), "{message}");
                assert!(message.contains("destroyed"), "{message}");
            }
            other => panic!("expected the real destroyed-buffer error, got {other:?}"),
        }
    }
    let destination = device.upload(&[17_u32, 23])?;
    device.write_buffer(&destination, &[2, 3])?;
    assert_eq!(device.download_owned(&destination)?, [2, 3]);
    Ok(())
}

#[test]
fn transfer_subranges_preserve_bytes_on_rejection() -> Result<()> {
    let Some(device) = device_or_skip() else {
        return Ok(());
    };
    let original = [3_u8, 5, 7, 11, 13, 17, 19];
    let destination = device.upload(&original)?;
    for (offset, values) in [(1, &[23_u8; 4][..]), (0, &[23; 3]), (8, &[23; 4])] {
        match device.write_sub_buffer(&destination, offset, values) {
            Err(HephaestusError::TransferFailed { message })
                if offset + values.len() <= original.len() =>
            {
                assert!(message.contains("multiple of 4"), "{message}");
            }
            Err(HephaestusError::LengthMismatch {
                host_len,
                device_len,
            }) if offset + values.len() > original.len() => {
                assert_eq!(
                    (host_len, device_len),
                    (offset + values.len(), original.len())
                );
            }
            other => panic!("expected rejected subrange, got {other:?}"),
        }
        assert_eq!(device.download_owned(&destination)?, original);
    }
    device.write_sub_buffer(&destination, 0, &[23, 29, 31, 37])?;
    device.write_sub_buffer(&destination, 4, &[41, 43, 47])?;
    assert_eq!(
        device.download_owned(&destination)?,
        [23, 29, 31, 37, 41, 43, 47]
    );
    Ok(())
}
