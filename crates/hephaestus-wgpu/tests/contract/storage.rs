use super::device_or_skip;
use eunomia::{Pod, Zeroable};
use hephaestus_core::{CommandStream, ComputeDevice, HephaestusError, KernelDevice};
use hephaestus_wgpu::WgpuDevice;

fn storage_extents<T: Pod + Zeroable + PartialEq + core::fmt::Debug>(
    device: &WgpuDevice,
    source_value: T,
    sentinel: T,
) -> hephaestus_core::Result<()> {
    for len in 0..=9 {
        let original: Vec<_> = (0..len)
            .map(|index| match index % 3 {
                0 => source_value,
                1 => sentinel,
                _ => T::zeroed(),
            })
            .collect();
        let source = device.upload(&original)?;
        let destination = device.upload(&vec![sentinel; len])?;
        device.copy_buffer(&source, &source)?;
        let mut alias_stream = device.stream()?;
        alias_stream.copy_prefix(&source, &source, len)?;
        alias_stream.submit()?;
        assert_eq!(device.download_owned(&source)?, original);
        device.copy_buffer(&source, &destination)?;
        assert_eq!(device.download_owned(&destination)?, original);
        let mut stream = device.stream()?;
        stream.fill_zero(&destination)?;
        stream.submit()?;
        assert_eq!(device.download_owned(&destination)?, vec![T::zeroed(); len]);
        assert_eq!(device.download_owned(&source)?, original);

        for prefix in 0..=len {
            let destination = device.upload(&vec![sentinel; len + 2])?;
            let mut stream = device.stream()?;
            stream.copy_prefix(&source, &destination, prefix)?;
            stream.submit()?;
            let mut expected = vec![sentinel; len + 2];
            expected[..prefix].copy_from_slice(&original[..prefix]);
            assert_eq!(
                device.download_owned(&destination)?,
                expected,
                "length {len}, prefix {prefix}"
            );
            assert_eq!(device.download_owned(&source)?, original);
        }
        let destination = device.upload(&vec![sentinel; len + 2])?;
        let mut stream = device.stream()?;
        match stream.copy_prefix(&source, &destination, len + 1) {
            Err(HephaestusError::LengthMismatch {
                host_len,
                device_len,
            }) => {
                assert_eq!((host_len, device_len), (len + 1, len));
            }
            Err(error) => panic!("unexpected prefix rejection: {error}"),
            Ok(()) => panic!("an oversized prefix must fail"),
        }
        stream.submit()?;
        assert_eq!(
            device.download_owned(&destination)?,
            vec![sentinel; len + 2]
        );
        if len != 0 {
            let short = device.upload(&vec![sentinel; len - 1])?;
            let mut stream = device.stream()?;
            match stream.copy_prefix(&source, &short, len) {
                Err(HephaestusError::LengthMismatch {
                    host_len,
                    device_len,
                }) => {
                    assert_eq!((host_len, device_len), (len, len - 1));
                }
                Err(error) => panic!("unexpected short-destination rejection: {error}"),
                Ok(()) => panic!("a prefix cannot exceed destination storage"),
            }
            stream.submit()?;
            assert_eq!(device.download_owned(&short)?, vec![sentinel; len - 1]);
        }
    }
    Ok(())
}

#[test]
fn buffer_copy_and_clear_preserve_byte_extents() -> hephaestus_core::Result<()> {
    let Some(device) = device_or_skip() else {
        return Ok(());
    };
    storage_extents(&device, 0xa5_u8, 0x5a)?;
    storage_extents(&device, 0xa531_u16, 0x5ac7)?;
    storage_extents(&device, 0xa531_c748_u32, 0x5ac7_48ed)?;
    storage_extents(&device, 0xa531_c748_0294_bedf_u64, 0x5ac7_48ed_f039_ac57)?;
    storage_extents(&device, 1.25_f32, -2.5)?;
    storage_extents(&device, 1.25_f64, -2.5)?;
    storage_extents(&device, 0x0134_7523_i32, -0x2356_7102)?;
    storage_extents(
        &device,
        eunomia::F16::from_bits(0x3c01),
        eunomia::F16::from_bits(0x4003),
    )?;
    storage_extents(
        &device,
        eunomia::Bf16::from_bits(0x3f81),
        eunomia::Bf16::from_bits(0x4003),
    )
}
