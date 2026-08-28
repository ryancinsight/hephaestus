//! Process-local unit-test suite support.

pub(crate) type TestCase = (&'static str, fn());

pub(crate) fn run_cases(cases: &[TestCase]) {
    let failures = cases
        .iter()
        .filter_map(|(name, case)| std::panic::catch_unwind(case).is_err().then_some(*name))
        .collect::<Vec<_>>();

    assert!(
        failures.is_empty(),
        "unit-test cases failed: {}",
        failures.join(", ")
    );
}
