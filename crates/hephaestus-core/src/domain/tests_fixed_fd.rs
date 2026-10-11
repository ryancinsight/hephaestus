use super::*;

const SCHEMES: [FixedFd3DScheme; 5] = [
    FixedFd3DScheme::CentralSecondOrder,
    FixedFd3DScheme::CentralFourthOrder,
    FixedFd3DScheme::CentralSixthOrder,
    FixedFd3DScheme::StaggeredForward,
    FixedFd3DScheme::StaggeredBackward,
];

#[test]
fn minimum_extents_match_the_provider_thresholds() {
    let minima = [3, 1, 7, 2, 2];
    for (scheme, minimum) in SCHEMES.into_iter().zip(minima) {
        assert_eq!(scheme.min_axis_extent(), minimum, "{scheme:?}");
        for axis in [StaggeredAxis::X, StaggeredAxis::Y, StaggeredAxis::Z] {
            let mut dims = [8, 8, 8];
            dims[axis.index()] = minimum - 1;
            if minimum == 1 {
                continue;
            }
            let Err(HephaestusError::InvalidConfiguration { message }) =
                FixedFd3DParams::new(dims[0], dims[1], dims[2], axis, scheme, [0.5; 3])
            else {
                panic!("{scheme:?} on {axis:?} accepts below minimum")
            };
            assert_eq!(
                message,
                format!(
                    "fixed-fd {scheme:?} needs at least {minimum} points on the differentiated axis, got {}",
                    minimum - 1
                )
            );
            dims[axis.index()] = minimum;
            let grid = FixedFd3DParams::new(dims[0], dims[1], dims[2], axis, scheme, [0.5; 3])
                .expect("the provider minimum extent is accepted");
            assert_eq!(grid.dims(), dims);
        }
    }
}

#[test]
fn scales_follow_the_provider_parenthesization() {
    let params = FixedFd3DParams::new(8, 8, 8, StaggeredAxis::Y, SCHEMES[1], [0.5, 0.25, 1.0])
        .expect("valid block");
    let h = 0.25_f32;
    assert_eq!(
        params.scales,
        [1.0 / h, 1.0 / (2.0 * h), 1.0 / (12.0 * h), 1.0 / (60.0 * h)]
    );
}

#[test]
fn forward_shrinks_the_output_on_its_axis_only() {
    for axis in [StaggeredAxis::X, StaggeredAxis::Y, StaggeredAxis::Z] {
        let params =
            FixedFd3DParams::new(8, 6, 10, axis, FixedFd3DScheme::StaggeredForward, [0.5; 3])
                .expect("valid block");
        let mut expected = [8, 6, 10];
        expected[axis.index()] -= 1;
        assert_eq!(params.output_dims(), expected);
        let cells: usize = expected.iter().map(|extent| *extent as usize).product();
        assert_eq!(params.output_cell_count().expect("count"), cells);
        assert!(params.validate_storage(8 * 6 * 10, cells).is_ok());
        assert!(params.validate_storage(8 * 6 * 10, 8 * 6 * 10).is_err());
    }
}

#[test]
fn adjoint_storage_swaps_the_forward_grids() {
    let shrunk = FixedFd3DParams::new(
        8,
        6,
        10,
        StaggeredAxis::X,
        FixedFd3DScheme::StaggeredForward,
        [0.5; 3],
    )
    .expect("valid block");
    // The upstream has the forward output shape (one plane short), the
    // gradient the full input grid.
    assert!(
        shrunk
            .validate_adjoint_storage(7 * 6 * 10, 8 * 6 * 10)
            .is_ok()
    );
    assert!(
        shrunk
            .validate_adjoint_storage(8 * 6 * 10, 8 * 6 * 10)
            .is_err()
    );
    assert!(
        shrunk
            .validate_adjoint_storage(7 * 6 * 10, 7 * 6 * 10)
            .is_err()
    );

    let kept = FixedFd3DParams::new(
        8,
        8,
        8,
        StaggeredAxis::X,
        FixedFd3DScheme::CentralSixthOrder,
        [0.5; 3],
    )
    .expect("valid block");
    assert!(kept.validate_adjoint_storage(512, 512).is_ok());
    assert!(kept.validate_adjoint_storage(511, 512).is_err());
}

#[test]
fn central_and_backward_preserve_the_grid() {
    for scheme in [
        FixedFd3DScheme::CentralSecondOrder,
        FixedFd3DScheme::CentralFourthOrder,
        FixedFd3DScheme::CentralSixthOrder,
        FixedFd3DScheme::StaggeredBackward,
    ] {
        let params =
            FixedFd3DParams::new(8, 8, 8, StaggeredAxis::X, scheme, [0.5; 3]).expect("valid block");
        assert_eq!(params.output_dims(), [8, 8, 8]);
        assert!(params.validate_storage(512, 512).is_ok());
        assert!(params.validate_storage(512, 511).is_err());
    }
}
