//! Host instantiation of the shared roll conformance clause.

use hephaestus_conformance::{assert_roll_contract, assert_roll_contract_for_scalar};
use hephaestus_host::{HostDevice, HostRollOps};

#[test]
fn host_satisfies_the_roll_contract() {
    assert_roll_contract(&HostDevice::new(), &HostRollOps);
}

#[test]
fn host_satisfies_roll_contract_for_shipped_scalars() {
    let device = HostDevice::new();
    assert_roll_contract_for_scalar(
        &device,
        &HostRollOps,
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        &[4.0, 5.0, 1.0, 2.0, 3.0, 40.0, 50.0, 10.0, 20.0, 30.0],
        &[2.0, 3.0, 4.0, 5.0, 1.0, 20.0, 30.0, 40.0, 50.0, 10.0],
        &[10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        &[0.0, 10.0, 20.0, 0.0, 30.0, 40.0, 0.0, 50.0, 60.0],
        &[0.0, 50.0, 60.0, 0.0, 10.0, 20.0, 0.0, 30.0, 40.0],
    );
    assert_roll_contract_for_scalar(
        &device,
        &HostRollOps,
        &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        &[4.0, 5.0, 1.0, 2.0, 3.0, 40.0, 50.0, 10.0, 20.0, 30.0],
        &[2.0, 3.0, 4.0, 5.0, 1.0, 20.0, 30.0, 40.0, 50.0, 10.0],
        &[10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        &[0.0, 10.0, 20.0, 0.0, 30.0, 40.0, 0.0, 50.0, 60.0],
        &[0.0, 50.0, 60.0, 0.0, 10.0, 20.0, 0.0, 30.0, 40.0],
    );
    assert_roll_contract_for_scalar(
        &device,
        &HostRollOps,
        &[1_u32, 2, 3, 4, 5, 10, 20, 30, 40, 50],
        &[4, 5, 1, 2, 3, 40, 50, 10, 20, 30],
        &[2, 3, 4, 5, 1, 20, 30, 40, 50, 10],
        &[10, 20, 30, 40, 50, 1, 2, 3, 4, 5],
        &[0, 10, 20, 0, 30, 40, 0, 50, 60],
        &[0, 50, 60, 0, 10, 20, 0, 30, 40],
    );
}
