/// The 3x4 fixture, row-major.
pub(crate) fn fixture() -> Vec<f32> {
    (1..=12).map(|value| value as f32).collect()
}
