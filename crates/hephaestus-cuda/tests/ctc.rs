//! CUDA instantiation of the shared CTC clauses.
//!
//! The shared conformance clause runs the dispatch against
//! `leto_ops::CtcState` — the CPU implementation this kernel mirrors — over
//! a problem matrix covering mixed lengths, empty targets, suppressed and
//! allowed skip transitions, a nonzero blank, and the empty-frame edges, so
//! every branch of both recurrences is exercised on the device.

use hephaestus_conformance::assert_ctc_contract;
use hephaestus_cuda::{CudaCtcOps, CudaDevice};

fn device(test: &str) -> Option<CudaDevice> {
    match CudaDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) => {
            if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_some() {
                panic!("CUDA device required for {test}: {error}");
            }
            eprintln!("skipping CUDA CTC contract {test}: {error}");
            None
        }
    }
}

#[test]
fn cuda_satisfies_the_ctc_contract() {
    let Some(device) = device("cuda_satisfies_the_ctc_contract") else {
        return;
    };
    assert_ctc_contract(&device, &CudaCtcOps);
}
