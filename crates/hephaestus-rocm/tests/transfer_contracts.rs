//! ROCm instantiation of the shared transfer conformance clauses.

#[cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_transfer_contract;
#[cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_rocm::RocmDevice;

const MEMORY_SOURCE: &str = include_str!("../src/infrastructure/memory.rs");
const STREAM_SOURCE: &str = include_str!("../src/application/stream.rs");

#[test]
fn device_copy_uses_one_synchronous_copy_without_a_global_barrier() {
    let (_, copy_and_rest) = MEMORY_SOURCE
        .split_once("fn copy_buffer<T: Pod>")
        .expect("ROCm memory provider must implement copy_buffer");
    let (copy_body, _) = copy_and_rest
        .split_once("fn topology(")
        .expect("copy_buffer must remain before topology");

    assert_eq!(
        copy_body.matches("stream.copy(src, dst)?;").count(),
        1,
        "copy_buffer must issue exactly one device-local copy"
    );
    assert_eq!(
        copy_body.matches("stream.submit()").count(),
        1,
        "copy_buffer must preserve command-stream submission"
    );
    assert_eq!(
        copy_body.matches("synchronize").count(),
        0,
        "synchronous hipMemcpyDtoD must not be followed by a global barrier"
    );

    let (_, device_copy_and_rest) = STREAM_SOURCE
        .split_once("fn copy_device(")
        .expect("ROCm command stream must implement copy_device");
    let (device_copy_body, _) = device_copy_and_rest
        .split_once("fn fill_device(")
        .expect("copy_device must remain before fill_device");
    assert_eq!(
        device_copy_body
            .matches("cubecl_hip_sys::hipMemcpyDtoD(dst, src, bytes)")
            .count(),
        1,
        "copy_device must retain exactly one synchronous HIP copy"
    );
    assert_eq!(
        device_copy_body.matches("hipMemcpyDtoDAsync").count(),
        0,
        "copy_device must not weaken copy_buffer to asynchronous completion"
    );
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
#[test]
fn rocm_satisfies_the_transfer_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm transfer conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm transfer conformance requires a physical device: {error}"),
    };
    assert_transfer_contract(&device);
}
