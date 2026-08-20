#![expect(
    clippy::unwrap_used,
    reason = "ratchet HEPH-UNWRAP-1: pre-existing debt"
)]

//! Host device: allocation, upload, and download.
//!
//! [`HostDevice`] implements the full [`ComputeDevice`] contract over plain
//! host memory — no GPU driver, no vendor toolkit.  It exists so:
//! (a) conformance tests run everywhere, and (b) Atlas consumers can write
//! generic code that compiles and passes without hardware.
//!
//! This example shows the core interop pattern:
//! 1. Allocate a typed device buffer.
//! 2. Upload host data via `write_buffer`.
//! 3. Modify the buffer in-place via `write_sub_buffer`.
//! 4. Download back to a host slice with `download`.

extern crate hephaestus_core;
extern crate hephaestus_host;
extern crate themis;

use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError};
use hephaestus_host::HostDevice;
use themis::PlacementHint;

fn main() {
    let device = HostDevice::new();

    println!("backend: {}", device.backend_name()); // "host"

    // ── Allocate a zero-initialised f32 buffer ──
    let buf = device
        .alloc_zeroed_with_hint::<f32>(8, PlacementHint::Current)
        .expect("allocation");
    assert_eq!(buf.len(), 8);
    println!("allocated {} f32 slots", buf.len());

    // ── Upload host data ──
    let source: Vec<f32> = (0..8).map(|i| i as f32).collect();
    device.write_buffer(&buf, &source).expect("write");

    // ── Download and verify ──
    let mut dst = vec![0.0_f32; 8];
    device.download(&buf, &mut dst).expect("download");
    assert_eq!(dst, source, "round-trip upload/download mismatch");
    println!("round-trip: {:?}", dst);

    // ── Sub-buffer write: overwrite elements 2..5 ──
    device
        .write_sub_buffer(&buf, 2, &[99.0_f32, 88.0, 77.0])
        .expect("sub-write");
    device.download(&buf, &mut dst).expect("download");
    assert_eq!(dst[2], 99.0);
    assert_eq!(dst[4], 77.0);
    assert_eq!(dst[5], 5.0, "elements outside the window are unchanged");
    println!("after sub-write: {:?}", dst);

    // ── Length mismatch is an error, never UB ──
    let mut too_short = vec![0.0_f32; 4];
    let err = device.download(&buf, &mut too_short).unwrap_err();
    assert!(matches!(err, HephaestusError::LengthMismatch { .. }));
    println!("length mismatch returned {:?}", err);

    // ── Copy buffer shares no aliasing ──
    let copy = device
        .alloc_zeroed_with_hint::<f32>(8, PlacementHint::Current)
        .expect("alloc copy");
    device.copy_buffer(&buf, &copy).expect("copy");
    let mut copy_out = vec![0.0_f32; 8];
    device
        .download(&copy, &mut copy_out)
        .expect("download copy");
    assert_eq!(copy_out[2], 99.0);
    println!("copy verified: {:?}", copy_out);

    println!("all host-device assertions passed");
}
