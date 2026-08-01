//! Contract clauses for device transfer and buffer initialization.
//!
//! Every conformance module exercises `upload`/`download` incidentally;
//! these clauses pin the transfer contract itself: bitwise round-trip
//! fidelity (transfers move representations, so NaN payload bits survive),
//! zero-initialization, sub-range writes, device-to-device copies, and
//! length-mismatch rejection. The storage-kernel and command-stream layers
//! beneath the operation seams are exercised transitively by every kernel
//! clause in this crate and carry no separate module.

use hephaestus_core::{ComputeDevice, DeviceBuffer};

/// Run every transfer clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_transfer_contract<D: ComputeDevice>(device: &D) {
    let name = device.backend_name();

    // Round-trip is bitwise: transfers move representations, not values,
    // so a NaN payload and signed zero survive exactly.
    let pattern = [
        1.5f32,
        -0.0,
        f32::from_bits(0x7fc0_1234),
        f32::MIN_POSITIVE,
        -3.25,
    ];
    let buffer = device.upload(&pattern).expect("upload");
    assert_eq!(buffer.len(), pattern.len(), "{name}: uploaded length");
    let mut got = [0.0f32; 5];
    device.download(&buffer, &mut got).expect("download");
    for (index, (host, device_value)) in pattern.iter().zip(&got).enumerate() {
        assert_eq!(
            host.to_bits(),
            device_value.to_bits(),
            "{name}: round-trip element {index} must be bitwise identical"
        );
    }

    // Provider-owned download preserves the same bitwise transfer contract.
    let owned = device.download_owned(&buffer).expect("owned download");
    assert_eq!(owned.len(), pattern.len(), "{name}: owned download length");
    for (index, (host, device_value)) in pattern.iter().zip(&owned).enumerate() {
        assert_eq!(
            host.to_bits(),
            device_value.to_bits(),
            "{name}: owned round-trip element {index} must be bitwise identical"
        );
    }

    // Zero allocation is observable zeros.
    let zeros = device.alloc_zeroed::<f32>(4).expect("alloc_zeroed");
    let mut got = [1.0f32; 4];
    device.download(&zeros, &mut got).expect("download");
    assert_eq!(got, [0.0; 4], "{name}: alloc_zeroed contents");

    // write_buffer replaces the full contents in place.
    device
        .write_buffer(&zeros, &[9.0f32, 8.0, 7.0, 6.0])
        .expect("write_buffer");
    device.download(&zeros, &mut got).expect("download");
    assert_eq!(got, [9.0, 8.0, 7.0, 6.0], "{name}: write_buffer contents");

    // write_sub_buffer touches exactly the addressed range.
    device
        .write_sub_buffer(&zeros, 1, &[0.5f32, 0.25])
        .expect("write_sub_buffer");
    device.download(&zeros, &mut got).expect("download");
    assert_eq!(
        got,
        [9.0, 0.5, 0.25, 6.0],
        "{name}: write_sub_buffer must touch only its range"
    );

    // copy_buffer duplicates device contents without a host round-trip.
    let copy = device.alloc_zeroed::<f32>(4).expect("copy alloc");
    device.copy_buffer(&zeros, &copy).expect("copy_buffer");
    let mut got_copy = [0.0f32; 4];
    device.download(&copy, &mut got_copy).expect("download");
    assert_eq!(got, got_copy, "{name}: copy_buffer contents");

    // Length mismatches are typed rejections, not truncation.
    let mut short = [0.0f32; 3];
    assert!(
        device.download(&copy, &mut short).is_err(),
        "{name}: a short download target must be rejected"
    );
    let long = device.alloc_zeroed::<f32>(6).expect("long alloc");
    assert!(
        device.copy_buffer(&copy, &long).is_err(),
        "{name}: a copy between different lengths must be rejected"
    );
    assert!(
        device.write_buffer(&copy, &[1.0f32; 2]).is_err(),
        "{name}: a short write_buffer source must be rejected"
    );
    assert!(
        device.write_sub_buffer(&copy, 3, &[1.0f32, 1.0]).is_err(),
        "{name}: an out-of-range write_sub_buffer must be rejected"
    );

    // Rejections leave the target untouched.
    device.download(&copy, &mut got_copy).expect("download");
    assert_eq!(
        got, got_copy,
        "{name}: rejected writes must not mutate the buffer"
    );

    // A probed hardware topology's REPORTED fields are mutually
    // consistent. Zero means unreported, never fabricated (wgpu's pinned
    // convention: WebGPU exposes no SM/register introspection), so the
    // clause constrains only fields the backend actually reported; `None`
    // (no snapshot at all) is equally valid. Encoding unknowability in
    // the type instead of a zero sentinel is filed upstream against
    // themis (ATLAS-THEMIS-TOPOLOGY-OPTION-1).
    if let Some(topology) = device.topology() {
        let warp = topology.warp_width();
        assert!(
            warp == 0 || warp.is_power_of_two(),
            "{name}: a reported warp width must be a power of two, got {warp}"
        );
        let threads = topology.max_threads_per_unit();
        if warp > 0 && threads > 0 {
            assert!(
                threads >= warp,
                "{name}: a compute unit must hold at least one warp"
            );
        }
    }

    // Zero-length transfers are valid no-ops.
    let empty = device.upload(&[] as &[f32]).expect("empty upload");
    assert_eq!(empty.len(), 0, "{name}: empty buffer length");
    let mut nothing: [f32; 0] = [];
    device
        .download(&empty, &mut nothing)
        .expect("empty download");
    let owned_empty = device.download_owned(&empty).expect("empty owned download");
    assert!(
        owned_empty.is_empty(),
        "{name}: empty owned download length"
    );
    assert_eq!(
        owned_empty.capacity(),
        0,
        "{name}: empty owned download must not allocate"
    );
}
