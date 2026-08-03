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
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA transfer conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA transfer conformance requires a physical device: {error}"),
    };
    assert_transfer_contract(&device);
}

#[test]
fn device_local_copy_uses_one_synchronous_copy_without_context_barrier() {
    let device_source = include_str!("../src/infrastructure/device.rs");
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
        copy.matches("cuMemcpyDtoD_v2(dst.raw(), src.raw(), byte_count)")
            .count(),
        1
    );
    assert_eq!(copy.matches("cuMemcpyDtoDAsync").count(), 0);

    let stream_sync = function_body(device_source, "fn synchronize_default_stream");
    assert_eq!(
        stream_sync
            .matches("cuda_oxide::sys::cuStreamSynchronize(core::ptr::null_mut())")
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
