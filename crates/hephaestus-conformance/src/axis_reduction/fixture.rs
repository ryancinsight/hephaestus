/// The 3x4 fixture, row-major.
pub(crate) fn fixture() -> Vec<f32> {
    (1_u8..=12).map(f32::from).collect()
}
