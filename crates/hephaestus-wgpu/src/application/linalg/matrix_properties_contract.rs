use super::with_row_major_matrix;
use leto::Layout;

#[test]
fn canonical_contiguous_matrix_reuses_downloaded_storage() {
    let layout = Layout::c_contiguous([2, 3]).expect("test layout must be valid");
    let mut downloaded = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let downloaded_pointer = downloaded.as_mut_ptr();

    let (scratch_pointer, observed) = with_row_major_matrix(&mut downloaded, &layout, |scratch| {
        (scratch.as_mut_ptr(), scratch.to_vec())
    })
    .expect("canonical matrix must expose its downloaded prefix");

    assert_eq!(scratch_pointer, downloaded_pointer);
    assert_eq!(observed, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn strided_matrix_compacts_into_independent_row_major_storage() {
    let layout = Layout::new([2, 2], [3, 1], 1);
    let mut downloaded = vec![99.0_f32, 1.0, 2.0, 99.0, 3.0, 4.0];
    let downloaded_pointer = downloaded.as_mut_ptr();

    let (scratch_pointer, observed) = with_row_major_matrix(&mut downloaded, &layout, |scratch| {
        scratch[0] = 7.0;
        (scratch.as_mut_ptr(), scratch.to_vec())
    })
    .expect("valid strided matrix must compact");

    assert_ne!(scratch_pointer, downloaded_pointer);
    assert_eq!(observed, [7.0, 2.0, 3.0, 4.0]);
    assert_eq!(downloaded, [99.0, 1.0, 2.0, 99.0, 3.0, 4.0]);
}
