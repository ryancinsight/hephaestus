//! Compile-time contract for the provider-owned WGPU ABI re-export.
//!
//! This test does not acquire a GPU. It verifies that downstream crates can
//! name WGPU descriptor types through `hephaestus_wgpu::wgpu`, keeping the ABI
//! version tied to the Hephaestus provider instead of a separate direct
//! dependency.

#[test]
fn provider_exports_wgpu_abi_types() {
    let usage = hephaestus_wgpu::wgpu::BufferUsages::STORAGE
        | hephaestus_wgpu::wgpu::BufferUsages::COPY_SRC
        | hephaestus_wgpu::wgpu::BufferUsages::COPY_DST;
    assert!(usage.contains(hephaestus_wgpu::wgpu::BufferUsages::STORAGE));

    let _descriptor = hephaestus_wgpu::wgpu::BufferDescriptor {
        label: Some("provider-owned-wgpu-abi-contract"),
        size: 16,
        usage,
        mapped_at_creation: false,
    };
}
