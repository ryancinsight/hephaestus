use super::{StridedMeta, pad_shape, pad_strides};
use hephaestus_core::HephaestusError;
use leto::Layout;

fn decoded_addresses(meta: &StridedMeta, linear: u32) -> [i64; 3] {
    let mut remaining = linear;
    let mut addresses = [meta.offsets[0], meta.offsets[1], meta.offsets[2]].map(i64::from);
    for axis in (0..8).rev() {
        let coordinate = i64::from(remaining % meta.shape[axis]);
        remaining /= meta.shape[axis];
        for (address, stride) in addresses.iter_mut().zip([
            meta.a_strides[axis],
            meta.b_strides[axis],
            meta.out_strides[axis],
        ]) {
            *address += coordinate * i64::from(stride);
        }
    }
    addresses
}

fn packed_layout<const N: usize>() {
    let shape = [2; N];
    let output =
        Layout::c_contiguous(shape).expect("invariant: at most eight axes of extent two fit usize");
    let mut strides = output.strides();
    strides.reverse();
    // A reversed first axis, transposed stride order, and a nonzero origin
    // exercise all metadata lanes without acquiring a device.
    strides[0] = -strides[0];
    let first = Layout::try_new(shape, strides, 3)
        .expect("invariant: origin three covers the reversed unit stride");
    let second =
        Layout::try_new(shape, [0; N], 7).expect("invariant: zero strides address origin seven");
    let len = output
        .checked_size()
        .expect("invariant: at most 256 logical elements");
    let meta = StridedMeta::new(&first, Some(&second), &output, len)
        .expect("invariant: bounded rank and layout values fit the metadata ABI");
    let unary = StridedMeta::new(&first, None, &output, len)
        .expect("invariant: bounded unary layout values fit the metadata ABI");
    assert_eq!(
        meta.offsets,
        [
            3,
            7,
            0,
            u32::try_from(len).expect("invariant: length is at most 256")
        ]
    );
    assert_eq!(
        unary.offsets,
        [
            3,
            0,
            0,
            u32::try_from(len).expect("invariant: length is at most 256")
        ]
    );
    assert_eq!(unary.b_strides, [0; 8]);
    assert_eq!(meta.a_strides, unary.a_strides);
    assert_eq!(meta.out_strides, unary.out_strides);
    assert_eq!(meta.shape, unary.shape);

    for linear in 0..len {
        let index = core::array::from_fn(|axis| (linear >> (N - 1 - axis)) & 1);
        let addresses = decoded_addresses(
            &meta,
            u32::try_from(linear).expect("invariant: logical index is below 256"),
        );
        assert_eq!(
            addresses,
            [
                i64::try_from(
                    first
                        .offset_of(index)
                        .expect("invariant: the index lies within every fixture dimension")
                )
                .expect("invariant: fixture physical offsets are below i64::MAX"),
                i64::try_from(
                    second
                        .offset_of(index)
                        .expect("invariant: the index lies within every fixture dimension")
                )
                .expect("invariant: fixture physical offsets are below i64::MAX"),
                i64::try_from(
                    output
                        .offset_of(index)
                        .expect("invariant: the index lies within every fixture dimension")
                )
                .expect("invariant: fixture physical offsets are below i64::MAX"),
            ]
        );
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn sparse_views_keep_unsigned_origins_and_wide_stride_products() {
    let origin =
        usize::try_from(u32::MAX).expect("invariant: this test requires a 64-bit pointer width");
    let stride =
        isize::try_from(i32::MAX).expect("invariant: this test requires a 64-bit pointer width");
    let first = Layout::try_new([3], [stride], origin)
        .expect("invariant: maximum sparse offset is 8589934589, below isize::MAX");
    let second = Layout::try_new([3], [-stride], origin)
        .expect("invariant: reversed sparse offsets range from one to u32::MAX");
    let output = Layout::try_new([3], [stride], 1 << 31)
        .expect("invariant: maximum sparse output offset is 6442450942");
    let meta = StridedMeta::new(&first, Some(&second), &output, 3)
        .expect("invariant: rank-one extents, strides and origins fit their ABI fields");
    assert_eq!(meta.offsets, [u32::MAX, u32::MAX, 1 << 31, 3]);
    assert_eq!(
        decoded_addresses(&meta, 0),
        [4_294_967_295, 4_294_967_295, 2_147_483_648]
    );
    assert_eq!(
        decoded_addresses(&meta, 2),
        [8_589_934_589, 1, 6_442_450_942]
    );
    for index in 0..3 {
        assert_eq!(
            decoded_addresses(
                &meta,
                u32::try_from(index).expect("invariant: index is below three")
            ),
            [
                i64::try_from(
                    first
                        .offset_of([index])
                        .expect("invariant: the index lies within every fixture dimension")
                )
                .expect("invariant: fixture physical offsets are below i64::MAX"),
                i64::try_from(
                    second
                        .offset_of([index])
                        .expect("invariant: the index lies within every fixture dimension")
                )
                .expect("invariant: fixture physical offsets are below i64::MAX"),
                i64::try_from(
                    output
                        .offset_of([index])
                        .expect("invariant: the index lies within every fixture dimension")
                )
                .expect("invariant: fixture physical offsets are below i64::MAX"),
            ]
        );
    }
}

#[test]
fn packed_addresses_match_layouts_at_every_supported_rank() {
    packed_layout::<1>();
    packed_layout::<2>();
    packed_layout::<3>();
    packed_layout::<4>();
    packed_layout::<5>();
    packed_layout::<6>();
    packed_layout::<7>();
    packed_layout::<8>();
}

#[test]
fn packing_preserves_all_axes_and_rejects_the_first_unsupported_rank() {
    assert_eq!(
        pad_shape([2, 3, 5, 7, 11, 13, 17, 19])
            .expect("invariant: eight positive dimensions fit u32"),
        [2, 3, 5, 7, 11, 13, 17, 19]
    );
    assert_eq!(
        pad_strides([-2, 3, 0, 7, 11, 13, 17, 19])
            .expect("invariant: eight signed strides fit i32"),
        [-2, 3, 0, 7, 11, 13, 17, 19]
    );
    assert_eq!(
        pad_shape([2, 3]).expect("invariant: rank two is within the metadata capacity"),
        [1, 1, 1, 1, 1, 1, 2, 3]
    );
    assert_eq!(
        pad_strides([3, -1]).expect("invariant: two signed strides fit i32"),
        [0, 0, 0, 0, 0, 0, 3, -1]
    );
    for result in [
        pad_shape([1; 9]).map(|_| ()),
        pad_strides([0; 9]).map(|_| ()),
    ] {
        match result {
            Err(HephaestusError::DispatchFailed { message }) => {
                assert_eq!(message, "strided dispatch rank 9 exceeds maximum 8");
            }
            other => panic!("expected unsupported-rank error, got {other:?}"),
        }
    }
}

#[test]
fn packed_abi_has_the_same_field_order_as_hip() {
    assert_eq!(core::mem::size_of::<StridedMeta>(), 144);
    assert_eq!(core::mem::offset_of!(StridedMeta, shape), 0);
    assert_eq!(core::mem::offset_of!(StridedMeta, a_strides), 32);
    assert_eq!(core::mem::offset_of!(StridedMeta, b_strides), 64);
    assert_eq!(core::mem::offset_of!(StridedMeta, out_strides), 96);
    assert_eq!(core::mem::offset_of!(StridedMeta, offsets), 128);
}
