use super::*;

#[test]
fn params_valid_grid() {
    let params = Laplacian2DParams::new(
        4,
        5,
        Length::from_unit::<Meter>(0.1),
        Length::from_unit::<Meter>(0.2),
        BoundaryCondition::Dirichlet,
    )
    .expect("valid metre spacing");

    assert_eq!(params.dims_bc, [4, 5, 0, 0]);
    assert!((params.inv2[0] - 100.0).abs() < f32::EPSILON);
    assert!((params.inv2[1] - 25.0).abs() < f32::EPSILON);
}

#[test]
fn params_rejects_too_small_axes() {
    for dimensions in [(1, 4), (4, 1)] {
        assert!(matches!(
            Laplacian2DParams::new(
                dimensions.0,
                dimensions.1,
                Length::from_unit::<Meter>(1.0),
                Length::from_unit::<Meter>(1.0),
                BoundaryCondition::Neumann,
            ),
            Err(HephaestusError::InvalidConfiguration { .. })
        ));
    }
}

#[test]
fn params_rejects_bad_spacing() {
    for bad in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, 0.0, -1.0] {
        for spacing in [
            (
                Length::from_unit::<Meter>(bad),
                Length::from_unit::<Meter>(1.0),
            ),
            (
                Length::from_unit::<Meter>(1.0),
                Length::from_unit::<Meter>(bad),
            ),
        ] {
            assert!(matches!(
                Laplacian2DParams::new(4, 4, spacing.0, spacing.1, BoundaryCondition::Periodic,),
                Err(HephaestusError::InvalidConfiguration { .. })
            ));
        }
    }
}
