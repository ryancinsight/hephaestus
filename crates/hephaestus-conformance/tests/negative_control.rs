//! Proves the clauses can fail, and that they fail for the right reason.
//!
//! Every other test in this workspace runs a clause against a backend that
//! should satisfy it, and passes. That is the shape a broken clause hides in:
//! invert a comparison, replace a condition with a constant, drop the loop that
//! walks the elements — and the clause still passes wherever the device is
//! correct. Nothing in a green run distinguishes "the contract holds" from "the
//! clause stopped checking".
//!
//! The negatives here run a clause against
//! [`FaultyDevice`](hephaestus_core::test_support::FaultyDevice), whose
//! behaviour violates the contract in one named way, and require the clause to
//! panic with the message of the *specific assertion that fault should trip*.
//!
//! The clauses name the device (`backend_name()`) and the violated property in
//! their panic messages, but never their own function name — so the property
//! text is what identifies the assertion. Keying on it is what makes this a
//! control rather than a smoke test: a clause that failed for an unrelated
//! reason would satisfy "it panicked" while proving nothing.

use hephaestus_conformance::assert_transfer_contract;
use hephaestus_core::ComputeDevice;
use hephaestus_core::test_support::{Fault, FaultyDevice};

/// Run `clause` against `device` and require the panic message to name
/// `expected_property`.
///
/// # Panics
///
/// Panics when `clause` returns normally — a clause that passes against a
/// device built to violate it is not testing the contract — or when the panic
/// message does not contain `expected_property`, which would mean some other
/// assertion tripped first and the fault under test was never reached.
#[track_caller]
fn assert_clause_fails_on(device: &FaultyDevice, expected_property: &str, clause: impl FnOnce()) {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(clause));
    let Err(payload) = outcome else {
        panic!(
            "the clause passed against a device that violates it (fault: {:?}), \
             so it is not checking the contract",
            device.fault()
        );
    };
    let message = panic_message(payload.as_ref());
    assert!(
        message.contains(expected_property),
        "the clause failed, but on a different assertion: expected the message \
         to contain {expected_property:?}, got {message:?}. A clause that fails \
         for an unrelated reason is not evidence that it checks this property."
    );
    assert!(
        message.contains(device.backend_name()),
        "the clause's message must name the device it ran against; got {message:?}"
    );
}

/// Extract a panic payload's message, for payloads that carry one.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        // A non-string payload still carries a distinct type; report that
        // rather than an empty message, so a failure here is diagnosable.
        "<panic payload was not a string>".to_owned()
    }
}

// The transfer clause carries several assertions, and each fault below is
// aimed at a different one. Four separate cases rather than one, because a
// single passing case cannot show that the other assertions still run.

/// A device that reverses transfers fails the bitwise round-trip assertion.
///
/// The input is deliberately not a palindrome: a reversed palindrome
/// round-trips correctly, so a palindromic operand would let a device that
/// genuinely corrupts transfers pass this clause.
#[test]
fn transfer_reversal_fails_the_round_trip_assertion() {
    let device = FaultyDevice::new(Fault::TransferReverses);
    assert_clause_fails_on(
        &device,
        "round-trip element 0 must be bitwise identical",
        || assert_transfer_contract(&device),
    );
}

/// A device that accepts a short download target fails the rejection assertion.
///
/// This is the failure a caller cannot see: the elements the device did copy
/// look correct, and there is no error to handle.
#[test]
fn accepted_length_mismatch_fails_the_rejection_assertion() {
    let device = FaultyDevice::new(Fault::LengthMismatchAccepted);
    assert_clause_fails_on(&device, "a short download target must be rejected", || {
        assert_transfer_contract(&device)
    });
}

/// A device that accepts a copy between unequal lengths fails its assertion.
#[test]
fn accepted_copy_length_mismatch_fails_the_rejection_assertion() {
    let device = FaultyDevice::new(Fault::CopyLengthMismatchAccepted);
    assert_clause_fails_on(
        &device,
        "a copy between different lengths must be rejected",
        || assert_transfer_contract(&device),
    );
}

/// The faults are distinguishable from one another.
///
/// Guards the guard: if two `Fault` variants produced the same observable
/// behaviour, the cases above would be one case repeated, and a clause could
/// stop checking one of them without any test noticing.
#[test]
fn each_fault_trips_a_different_assertion() {
    let properties = [
        (
            Fault::TransferReverses,
            "round-trip element 0 must be bitwise identical",
        ),
        (
            Fault::LengthMismatchAccepted,
            "a short download target must be rejected",
        ),
        (
            Fault::CopyLengthMismatchAccepted,
            "a copy between different lengths must be rejected",
        ),
    ];
    for (fault, property) in properties {
        let device = FaultyDevice::new(fault);
        assert_clause_fails_on(&device, property, || assert_transfer_contract(&device));
    }
}
/// Why no case here reaches a clause other than the transfer one.
///
/// `assert_transfer_contract` is the only clause in this crate that takes no
/// operation seam — it is generic over `ComputeDevice` alone. Every other
/// clause needs an `*Ops` implementor to be callable, so a control for those
/// clauses means implementing the seam first.
///
/// That is deliberately not done. The convolution seam alone declares twelve
/// generic methods with const rank parameters and explicit lifetimes; stubbing
/// them to forward to a faulty device would put a second, weaker implementation
/// of a real seam in the tree, and its correctness would become a thing to
/// maintain. The transfer faults below are all transfer concerns, and one
/// clause reaches every one of them.
///
/// What this therefore does *not* cover: a clause in another seam that ignores
/// its inputs. Catching that needs a correct seam implementor plus one fault per
/// clause in that seam — the same shape as the cases below, applied seam by
/// seam. See `ATLAS-HEPHAESTUS-HOST-SEAM-COVERAGE` on the atlas board: eighteen
/// of the nineteen seams have no implementor anywhere in the stack, so most of
/// that coverage cannot exist yet.
#[test]
fn the_seam_free_clause_is_the_one_a_device_without_a_seam_can_be_held_to() {
    // Structural rather than behavioural, and asserted rather than commented so
    // that a future signature change making this clause seam-generic breaks a
    // test instead of silently removing the only seam-free control.
    let device = FaultyDevice::new(Fault::TransferReverses);
    assert_clause_fails_on(
        &device,
        "round-trip element 0 must be bitwise identical",
        || assert_transfer_contract(&device),
    );
}
