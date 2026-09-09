//! Assertions that name the constraint a call violated.
//!
//! An assertion that only checks a result is an error passes whenever the call
//! fails, including for a reason unrelated to the one the test exists to
//! catch. That is sharper here than in most crates: a conformance clause runs
//! against every backend, so an error-only assertion also hides the case where
//! two backends reject the same input with different diagnostics.

use core::fmt::Display;

/// Assert that `result` is an error whose rendered form contains `fragment`.
///
/// Generic over the error type, and taking the result by reference so a caller
/// that inspects the value afterwards keeps its binding.
///
/// # Panics
///
/// Panics when `result` is `Ok`, or when its error does not contain
/// `fragment`.
#[track_caller]
pub fn assert_rejects<T, E: Display>(result: &Result<T, E>, fragment: &str) {
    match result {
        Err(error) => {
            let rendered = format!("{error}");
            assert!(
                rendered.contains(fragment),
                "rejection must name the violated constraint: expected a \
                 message containing {fragment:?}, got {rendered:?}"
            );
        }
        Ok(_) => panic!("expected a rejection containing {fragment:?}, got Ok"),
    }
}
