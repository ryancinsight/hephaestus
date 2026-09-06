//! CUDA instantiation of the shared transfer conformance clauses.

#[cfg(feature = "cuda")]
use hephaestus_conformance::assert_transfer_contract;
#[cfg(feature = "cuda")]
use hephaestus_cuda::CudaDevice;

#[cfg(feature = "cuda")]
#[test]
fn cuda_satisfies_the_transfer_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error @ hephaestus_core::HephaestusError::AdapterUnavailable { .. })
            if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() =>
        {
            eprintln!("skip CUDA transfer conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA transfer conformance requires a physical device: {error}"),
    };
    assert_transfer_contract(&device);
}

#[test]
fn device_local_copy_uses_one_synchronous_copy_without_context_barrier() {
    let device_source = include_str!("../src/infrastructure/device/compute.rs");
    let copy_buffer = function_body(device_source, "fn copy_buffer<T: Pod>");
    assert_eq!(copy_buffer.matches("stream.copy(src, dst)?").count(), 1);
    assert_eq!(copy_buffer.matches("stream.submit()").count(), 1);
    assert_eq!(copy_buffer.matches("self.synchronize()").count(), 0);
    assert_eq!(
        copy_buffer
            .matches("self.synchronize_default_stream()")
            .count(),
        1
    );

    let stream_source = include_str!("../src/application/stream.rs");
    let copy = function_body(stream_source, "fn copy<T: Pod>");
    assert_eq!(
        copy.matches("memory.copy)(dst.raw(), src.raw(), byte_len)")
            .count(),
        1
    );
    assert_eq!(copy.matches("cuMemcpyDtoDAsync").count(), 0);

    let stream_sync = function_body(device_source, "fn synchronize_default_stream");
    assert_eq!(
        stream_sync
            .matches("context.synchronize_stream)(core::ptr::null_mut())")
            .count(),
        1
    );
    assert_eq!(stream_sync.matches("cuCtxSynchronize").count(), 0);
}

fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source.find(signature).expect("function signature");
    let body_start = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .expect("function body start");
    let mut depth = 0_usize;
    for (offset, byte) in source[body_start..].bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[body_start..=body_start + offset];
                }
            }
            _ => {}
        }
    }
    panic!("function body end");
}

#[cfg(feature = "cuda")]
#[test]
fn memory_capacity_uses_pointer_sized_driver_outputs() {
    use hephaestus_core::{ComputeDevice, ComputeDeviceCapabilities, HephaestusError};
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(HephaestusError::AdapterUnavailable { .. })
            if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() =>
        {
            return;
        }
        Err(error) => panic!("CUDA memory ABI requires a working device: {error}"),
    };
    let total = device.device_limits().max_buffer_size;
    let free = device.free_memory_bytes().expect("query free memory");
    assert!(total > 0, "a real CUDA device has memory capacity");
    assert!(free <= total, "free bytes cannot exceed physical capacity");
    let expected = [0x1234_5678_u64, u64::MAX, 0, 1 << 63];
    let buffer = device
        .upload(&expected)
        .expect("allocate after memory query");
    let retained = device.clone();
    drop(device);
    let mut actual = [0; 4];
    retained
        .download(&buffer, &mut actual)
        .expect("retained context transfer");
    assert_eq!(actual, expected);
}
