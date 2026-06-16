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

    // Real driver-queried values: every modern CUDA device has at least one SM,
    // a 32-lane warp, resident threads, registers, shared memory, and nonzero
    // global memory. Zeros would mean the snapshot was fabricated rather than
    // queried (the prior placeholder reported compute_units = 0, memory = 0).
    assert!(
        topo.compute_units() > 0,
        "compute_units must be queried (>0), got {}",
        topo.compute_units()
    );
    assert_eq!(
        topo.warp_width(),
        32,
        "every NVIDIA GPU has a 32-lane warp, got {}",
        topo.warp_width()
    );
    assert!(
        topo.max_threads_per_unit() > 0,
        "max_threads_per_unit must be queried (>0), got {}",
        topo.max_threads_per_unit()
    );
    assert!(
        topo.registers_per_unit() > 0,
        "registers_per_unit must be queried (>0), got {}",
        topo.registers_per_unit()
    );
    assert!(
        topo.shared_mem_per_unit_bytes() > 0,
        "shared_mem_per_unit_bytes must be queried (>0), got {}",
        topo.shared_mem_per_unit_bytes()
    );
    assert!(
        topo.memory_bytes() > 0,
        "device global memory must be queried (>0), got {}",
        topo.memory_bytes()
    );
    // Derived occupancy figure must follow from the real capacities.
    assert!(
        topo.max_resident_warps() > 0,
        "max_resident_warps must follow from real capacities, got {}",
        topo.max_resident_warps()
    );
}
