//! External-backend compile contract for the open [`DeviceApi`] seam.

use hephaestus_core::DeviceApi;
use hephaestus_cuda::CudaDevice;

fn assert_device_api<D: DeviceApi>() {}

#[test]
fn cuda_backend_implements_the_external_device_api_contract() {
    assert_device_api::<CudaDevice>();
}
