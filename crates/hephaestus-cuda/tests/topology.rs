//! Verifies `CudaDevice` reports real device properties into the themis
//! `GpuTopology` snapshot (atlas ADR 0002), queried from the driver rather than
//! hardcoded. Skips when no CUDA device / `cuda` feature is available.

use hephaestus_cuda::CudaDevice;

#[test]
fn topology_reflects_real_device_properties() {
    let Ok(dev) = CudaDevice::try_default() else {
        eprintln!("skip topology_reflects_real_device_properties: no CUDA device");
        return;
    };
    let topo = dev
        .topology()
        .expect("acquired CUDA device must report a topology snapshot");

    // Real driver-queried values: every modern CUDA device reports every
    // capacity, so each accessor is Some (an unreported capacity would be
    // None by type, never a fabricated number).
    assert!(
        topo.compute_units().is_some(),
        "compute_units must be queried, got None"
    );
    assert_eq!(
        topo.warp_width().map(core::num::NonZeroU32::get),
        Some(32),
        "every NVIDIA GPU has a 32-lane warp, got {:?}",
        topo.warp_width()
    );
    assert!(
        topo.max_threads_per_unit().is_some(),
        "max_threads_per_unit must be queried, got None"
    );
    assert!(
        topo.registers_per_unit().is_some(),
        "registers_per_unit must be queried, got None"
    );
    assert!(
        topo.shared_mem_per_unit_bytes().is_some(),
        "shared_mem_per_unit_bytes must be queried, got None"
    );
    assert!(
        topo.memory_bytes().is_some(),
        "device global memory must be queried, got None"
    );
    // Derived occupancy figure must follow from the real capacities.
    assert!(
        topo.max_resident_warps().is_some_and(|warps| warps > 0),
        "max_resident_warps must follow from real capacities, got {:?}",
        topo.max_resident_warps()
    );
}
