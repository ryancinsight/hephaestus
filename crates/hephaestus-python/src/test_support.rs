//! Shared test scaffolding: one-time Python interpreter initialisation and
//! the device-lifecycle case runner for the binding test modules.

use std::sync::{Once, OnceLock};

use crate::device::PyDevice;

static INIT_PYTHON: Once = Once::new();

pub(crate) type TestCase = (&'static str, fn());

pub(crate) fn prepare_python() {
    INIT_PYTHON.call_once(pyo3::Python::initialize);
}

pub(crate) fn default_wgpu_device() -> &'static PyDevice {
    static DEVICE: OnceLock<PyDevice> = OnceLock::new();
    DEVICE.get_or_init(|| {
        PyDevice::new(None).expect("invariant: WGPU binding tests require a working device")
    })
}

#[test]
fn wgpu_bindings_share_device_lifecycle() {
    run_cases(&[
        (
            "array_tolist_and_numpy_preserve_values",
            crate::array::tests::array_tolist_and_numpy_preserve_values as fn(),
        ),
        (
            "random_initializers_preserve_seeded_contracts",
            crate::random::tests::random_initializers_preserve_seeded_contracts as fn(),
        ),
        (
            "sparse_roundtrip_and_products_preserve_values",
            crate::sparse::tests::sparse_roundtrip_and_products_preserve_values as fn(),
        ),
    ]);
}

fn run_cases(cases: &[TestCase]) {
    let failures = cases
        .iter()
        .filter_map(|(name, case)| std::panic::catch_unwind(case).is_err().then_some(*name))
        .collect::<Vec<_>>();

    assert!(
        failures.is_empty(),
        "Python WGPU binding cases failed: {}",
        failures.join(", ")
    );
}
