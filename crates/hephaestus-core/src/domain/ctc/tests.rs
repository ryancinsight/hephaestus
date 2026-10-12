use super::*;

fn problem() -> CtcProblem {
    CtcProblem::new(4, 2, 6, 0, &[4, 3], &[2, 0], &[1, 2]).expect("a valid problem")
}

#[test]
fn upload_vectors_expand_labels_and_metadata() {
    let problem = problem();
    // Sample 0 interleaves blanks around [1, 2]; sample 1 is blank-only.
    assert_eq!(problem.labels(), &[0, 1, 0, 2, 0, 0]);
    assert_eq!(problem.meta(), &[4, 5, 0, 0, 3, 1, 20, 5]);
    assert_eq!(problem.divisors(), &[2.0, 1.0]);
    // Four frames of five states plus three frames of one state.
    assert_eq!(problem.total_cells(), 23);
    assert_eq!(problem.max_states(), 5);
    assert_eq!(problem.max_frames(), 4);
}

#[test]
fn validation_mirrors_the_provider_rejections() {
    assert!(CtcProblem::new(4, 0, 6, 0, &[], &[], &[]).is_err());
    assert!(CtcProblem::new(4, 2, 0, 0, &[4, 3], &[2, 0], &[1, 2]).is_err());
    assert!(CtcProblem::new(4, 2, 6, 0, &[4], &[2, 0], &[1, 2]).is_err());
    assert!(CtcProblem::new(4, 2, 6, 6, &[4, 3], &[2, 0], &[1, 2]).is_err());
    assert!(CtcProblem::new(4, 2, 6, 0, &[5, 3], &[2, 0], &[1, 2]).is_err());
    assert!(CtcProblem::new(4, 2, 6, 0, &[4, 3], &[2, 0], &[1]).is_err());
    assert!(CtcProblem::new(4, 2, 6, 0, &[4, 3], &[2, 0], &[1, 0]).is_err());
    assert!(CtcProblem::new(4, 2, 6, 0, &[4, 3], &[2, 0], &[1, 6]).is_err());
    assert_eq!(problem().batch_divisor(), 2.0);
}

#[test]
fn storage_validation_covers_both_grids() {
    let problem = problem();
    assert_eq!(problem.grid_len().expect("a grid length"), 4 * 2 * 6);
    assert!(problem.validate_storage(4 * 2 * 6, 4 * 2 * 6).is_ok());
    assert!(problem.validate_storage(4 * 2 * 6 - 1, 4 * 2 * 6).is_err());
    assert!(problem.validate_storage(4 * 2 * 6, 4 * 2 * 6 + 1).is_err());
}

#[test]
fn backward_params_reject_a_non_finite_seed() {
    let problem = problem();
    assert!(CtcBackwardParams::new(&problem, 1.0).is_ok());
    assert!(CtcBackwardParams::new(&problem, f32::NAN).is_err());
    assert!(CtcBackwardParams::new(&problem, f32::INFINITY).is_err());
}

#[test]
fn both_param_blocks_share_the_dims_convention() {
    let problem = problem();
    let step = CtcStepParams::for_frame(&problem, 3);
    let backward = CtcBackwardParams::new(&problem, 1.5).expect("a seed");
    // Every entry reads dims.y as the batch and dims.z as the classes;
    // the posterior's frame slot is zero.
    assert_eq!(step.dims, [3, 2, 6, 0]);
    assert_eq!(backward.dims, [0, 2, 6, 0]);
    assert_eq!(backward.scalars[0], 1.5);
}
