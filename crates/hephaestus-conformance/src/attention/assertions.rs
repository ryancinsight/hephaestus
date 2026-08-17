//! Shared value assertions for attention contract clauses.

use hephaestus_core::ComputeDevice;

pub(super) fn assert_download_eq<D, T>(
    device: &D,
    buffer: &D::Buffer<T>,
    expected: &[T],
    clause: &str,
) where
    D: ComputeDevice,
    T: bytemuck::Pod + Default + Copy + PartialEq + core::fmt::Debug,
{
    let mut actual = vec![T::default(); expected.len()];
    device.download(buffer, &mut actual).expect(clause);
    assert_eq!(actual, expected, "{clause}");
}
