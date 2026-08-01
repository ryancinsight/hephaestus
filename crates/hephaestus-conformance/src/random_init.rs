//! Contract clauses for device-neutral seeded random initialization.
//!
//! Every backend delegates seeded generation to leto-ops on the host and
//! uploads the result, so the host generator is an exact bitwise oracle:
//! the device buffer must equal the host sequence for the same
//! `(shape, parameters, seed)`. Determinism and seed sensitivity are part
//! of the seam contract.

use hephaestus_core::{ComputeDevice, RandomInitOps};

/// Run every random-initialization clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_random_init_contract<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: RandomInitOps<D, f32>,
{
    let name = device.backend_name();

    // Uniform: matches the host oracle bitwise, lands in [low, high).
    let buffer = ops
        .uniform_with_seed(device, [3, 4], -1.0f32, 2.0, 42)
        .expect("uniform generation");
    let mut got = [0.0f32; 12];
    device.download(&buffer, &mut got).expect("download");
    let host = leto_ops::uniform_with_seed([3usize, 4], -1.0f32, 2.0, 42).expect("host uniform");
    let host_values = leto::Storage::as_slice(host.storage());
    assert_eq!(
        got.as_slice(),
        host_values,
        "{name}: seeded uniform must equal the host generator bitwise"
    );
    for (index, value) in got.iter().enumerate() {
        assert!(
            (-1.0..2.0).contains(value),
            "{name}: uniform sample {index} = {value} outside [-1, 2)"
        );
    }

    // Determinism: the same seed reproduces the same buffer.
    let again = ops
        .uniform_with_seed(device, [3, 4], -1.0f32, 2.0, 42)
        .expect("uniform regeneration");
    let mut got_again = [0.0f32; 12];
    device.download(&again, &mut got_again).expect("download");
    assert_eq!(
        got, got_again,
        "{name}: identical seeds must reproduce identical buffers"
    );

    // Seed sensitivity: a different seed changes the sequence.
    let other = ops
        .uniform_with_seed(device, [3, 4], -1.0f32, 2.0, 43)
        .expect("uniform other seed");
    let mut got_other = [0.0f32; 12];
    device.download(&other, &mut got_other).expect("download");
    assert_ne!(
        got, got_other,
        "{name}: a different seed must change the sequence"
    );

    // Normal: matches the host oracle bitwise and is finite.
    let buffer = ops
        .normal_with_seed(device, [3, 4], 1.0f32, 0.5, 7)
        .expect("normal generation");
    device.download(&buffer, &mut got).expect("download");
    let host = leto_ops::normal_with_seed([3usize, 4], 1.0f32, 0.5, 7).expect("host normal");
    let host_values = leto::Storage::as_slice(host.storage());
    assert_eq!(
        got.as_slice(),
        host_values,
        "{name}: seeded normal must equal the host generator bitwise"
    );
    assert!(
        got.iter().all(|value| value.is_finite()),
        "{name}: normal samples must be finite"
    );
}
