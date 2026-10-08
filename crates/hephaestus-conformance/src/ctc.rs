//! Contract clauses for the device-neutral CTC seam.
//!
//! Four oracles, each catching what the others cannot:
//!
//! - **Provider forward.** The device loss runs against `leto_ops::CtcState`
//!   on the same log-probabilities over a problem matrix — mixed lengths,
//!   empty targets, repeated labels that suppress the skip transition,
//!   all-distinct labels that allow it, a nonzero blank, an empty-frame
//!   sample, and an empty grid — so every branch of both recurrences and
//!   the likelihood finish is exercised. The kernels reproduce the
//!   provider's operation order exactly, so the bound is elementary-function
//!   rounding plus denormal handling, not a modeling tolerance.
//! - **Provider backward.** The device gradient runs lane-by-lane against
//!   `backward_accumulate` over the same matrix, with a non-unit seed so a
//!   dropped or misplaced seed cannot hide.
//! - **Structural.** The posterior is a distribution: each active frame's
//!   gradient sums to the seeded, doubly normalized unit mass, which falls
//!   out of the forward-backward identities rather than of any
//!   implementation. Padded lanes are additionally proven untouched under a
//!   distinctive initial gradient, which a zero-initialized oracle
//!   comparison cannot see.
//! - **Edge.** An impossible alignment — a nonempty target with zero valid
//!   frames — yields positive infinity on the device exactly as on the
//!   provider, and its backward pass is skipped: the provider errors there,
//!   and the device contract requires callers to check the loss first.

use hephaestus_core::{ComputeDevice, CtcOps, CtcProblem, CtcStateBuffers};
use leto::{ArrayView3, ArrayViewMut3, Layout as LetoLayout};
use leto_ops::ctc::CtcState;

const UPSTREAM: f32 = 1.5;
const PADDING_SENTINEL: f32 = 2.25;

/// Loss and gradient bound in ULPs of the reference magnitude: the rational
/// TwoSum arithmetic is bit-identical, and only `exp`/`log` rounding plus
/// denormal handling separate the device from the host libm. Structural
/// defects — a wrong predecessor set, skip condition, or terminal init —
/// move results macroscopically and cannot hide under this bound; finer
/// operation-order effects live below what any cross-driver bound can pin.
const BOUND_ULPS: f32 = 64.0;

struct Case {
    frames: usize,
    batch: usize,
    classes: usize,
    blank: usize,
    input_lengths: Vec<usize>,
    target_lengths: Vec<usize>,
    targets: Vec<usize>,
    /// One emission lane forced to negative infinity, or none.
    neg_inf_lane: Option<[usize; 3]>,
    /// Whether every sample's alignment is possible.
    reachable: bool,
}

fn cases() -> Vec<Case> {
    vec![
        // Mixed lengths, a repeated label (skip suppressed), a single label,
        // and an empty sample: every recurrence branch is live somewhere.
        Case {
            frames: 6,
            batch: 3,
            classes: 5,
            blank: 0,
            input_lengths: vec![6, 5, 0],
            target_lengths: vec![2, 1, 0],
            targets: vec![2, 2, 1],
            neg_inf_lane: None,
            reachable: true,
        },
        // Minimum grid with a nonzero blank.
        Case {
            frames: 1,
            batch: 1,
            classes: 2,
            blank: 1,
            input_lengths: vec![1],
            target_lengths: vec![1],
            targets: vec![0],
            neg_inf_lane: None,
            reachable: true,
        },
        // All-distinct adjacent labels, so every skip transition is allowed.
        Case {
            frames: 5,
            batch: 2,
            classes: 4,
            blank: 0,
            input_lengths: vec![5, 5],
            target_lengths: vec![2, 2],
            targets: vec![1, 2, 3, 1],
            neg_inf_lane: None,
            reachable: true,
        },
        // A zero-probability lane on the blank path: the alignment survives
        // through the non-blank state.
        Case {
            frames: 4,
            batch: 1,
            classes: 3,
            blank: 0,
            input_lengths: vec![4],
            target_lengths: vec![1],
            targets: vec![1],
            neg_inf_lane: Some([2, 0, 0]),
            reachable: true,
        },
        // An empty grid: the drive loop takes no steps and the posterior
        // dispatches over an empty domain.
        Case {
            frames: 0,
            batch: 2,
            classes: 3,
            blank: 0,
            input_lengths: vec![0, 0],
            target_lengths: vec![0, 0],
            targets: vec![],
            neg_inf_lane: None,
            reachable: true,
        },
        // A nonempty target with zero valid frames: the loss is positive
        // infinity and backward is out of contract.
        Case {
            frames: 4,
            batch: 2,
            classes: 3,
            blank: 0,
            input_lengths: vec![4, 0],
            target_lengths: vec![1, 1],
            targets: vec![1, 2],
            neg_inf_lane: None,
            reachable: false,
        },
    ]
}

fn dense_layout(shape: [usize; 3]) -> LetoLayout<3> {
    let [frames, batch, classes] = shape;
    let strides = [
        isize::try_from(batch * classes).expect("a stride fitting isize"),
        isize::try_from(classes).expect("a stride fitting isize"),
        1,
    ];
    LetoLayout::<3>::try_new([frames, batch, classes], strides, 0).expect("a contiguous layout")
}

fn log_probs(case: &Case, seed: u64) -> Vec<f32> {
    let mut values = vec![0.0_f32; case.frames * case.batch * case.classes];
    for t in 0..case.frames {
        for b in 0..case.batch {
            for c in 0..case.classes {
                let hash = (t * 13 + b * 7 + c * 3 + seed as usize) % 97;
                values[(t * case.batch + b) * case.classes + c] = -(hash as f32 * 0.05 + 0.01);
            }
        }
    }
    if let Some([t, b, c]) = case.neg_inf_lane {
        values[(t * case.batch + b) * case.classes + c] = f32::NEG_INFINITY;
    }
    values
}

fn bound(reference: f32) -> f32 {
    BOUND_ULPS * f32::EPSILON * reference.abs().max(1.0)
}

/// Run every CTC clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a device.
pub fn assert_ctc_contract<D, S>(device: &D, ops: &S)
where
    D: ComputeDevice,
    S: CtcOps<D>,
{
    let name = device.backend_name();
    let kernel = ops.prepare_ctc(device).expect("ctc kernel compile");

    for (case_index, case) in cases().into_iter().enumerate() {
        let tag = format!(
            "{name}: case {case_index} over [{}, {}, {}]",
            case.frames, case.batch, case.classes
        );
        let problem = CtcProblem::new(
            case.frames,
            case.batch,
            case.classes,
            case.blank,
            &case.input_lengths,
            &case.target_lengths,
            &case.targets,
        )
        .expect("the device validates a problem the provider accepts");
        let input = log_probs(&case, 11 + case_index as u64);

        // Provider forward: the device loss matches the CPU loss.
        let view = ArrayView3::try_new(
            dense_layout([case.frames, case.batch, case.classes]),
            &input,
        )
        .expect("an input view over the lanes");
        let oracle = CtcState::<f32>::forward(
            &view,
            &case.targets,
            &case.input_lengths,
            &case.target_lengths,
            case.blank,
        )
        .expect("the provider runs a valid problem");
        let device_input = device.upload(&input).expect("input upload");
        let state = CtcStateBuffers::allocate(device, problem).expect("state alloc");
        let loss = ops
            .ctc_forward_into(device, &kernel, &device_input, &state)
            .expect("forward dispatch");
        let expected = oracle.loss();
        if case.reachable {
            let deviation = (loss - expected).abs();
            assert!(
                deviation <= bound(expected),
                "{tag}: loss is {loss:e}, provider says {expected:e} (bound {:e})",
                bound(expected)
            );
        } else {
            assert_eq!(expected, f32::INFINITY, "{tag}: the oracle pins +inf");
            assert_eq!(loss, f32::INFINITY, "{tag}: impossible loss is {loss:e}");
            continue;
        }

        // Provider backward: the device gradient matches lane by lane.
        let grid = case.frames * case.batch * case.classes;
        let mut reference = vec![0.0_f32; grid];
        let mut grad_view = ArrayViewMut3::try_new(
            dense_layout([case.frames, case.batch, case.classes]),
            &mut reference,
        )
        .expect("a gradient view over the lanes");
        oracle
            .backward_accumulate(UPSTREAM, &mut grad_view)
            .expect("the provider differentiates a reachable problem");
        let grad = device.alloc_zeroed::<f32>(grid).expect("gradient alloc");
        ops.ctc_backward_into(device, &kernel, UPSTREAM, &grad, &state)
            .expect("backward dispatch");
        let mut got = vec![0.0_f32; grid];
        device.download(&grad, &mut got).expect("gradient readback");
        for (index, (value, lane)) in got.iter().zip(&reference).enumerate() {
            let deviation = (value - lane).abs();
            assert!(
                deviation <= bound(*lane),
                "{tag}: gradient lane {index} is {value:e}, provider says {lane:e} (bound {:e})",
                bound(*lane)
            );
        }

        // Structural: each active frame's posterior sums to unit mass, so its
        // gradient sums to the seeded, doubly normalized unit.
        for (b, &length) in case.input_lengths.iter().enumerate() {
            let target = case.target_lengths[b].max(1) as f32;
            let expected_sum = (-UPSTREAM / target) / case.batch as f32;
            for t in 0..length {
                let mut sum = 0.0_f32;
                for c in 0..case.classes {
                    sum += got[(t * case.batch + b) * case.classes + c];
                }
                let slack = case.classes as f32 * 64.0 * f32::EPSILON * expected_sum.abs().max(1.0);
                assert!(
                    (sum - expected_sum).abs() <= slack,
                    "{tag}: frame {t} of sample {b} sums to {sum:e}, unit mass says {expected_sum:e}"
                );
            }
        }

        // Structural: padded lanes are untouched, bit for bit.
        let padded = device
            .upload(&vec![PADDING_SENTINEL; grid])
            .expect("sentinel upload");
        ops.ctc_backward_into(device, &kernel, UPSTREAM, &padded, &state)
            .expect("sentinel backward dispatch");
        let mut sentinel = vec![0.0_f32; grid];
        device
            .download(&padded, &mut sentinel)
            .expect("sentinel readback");
        for (b, &length) in case.input_lengths.iter().enumerate() {
            for t in length..case.frames {
                for c in 0..case.classes {
                    let index = (t * case.batch + b) * case.classes + c;
                    assert_eq!(
                        sentinel[index], PADDING_SENTINEL,
                        "{tag}: padded lane ({t}, {b}, {c}) was touched"
                    );
                }
            }
        }
    }

    // Rejection: a short grid fails before any launch.
    let problem = CtcProblem::new(2, 1, 2, 0, &[2], &[1], &[1]).expect("a small problem");
    let state = CtcStateBuffers::allocate(device, problem).expect("state alloc");
    let short = device.alloc_zeroed::<f32>(3).expect("short alloc");
    assert!(
        ops.ctc_forward_into(device, &kernel, &short, &state)
            .is_err(),
        "{name}: a short log-probability grid must fail before launch"
    );
    assert!(
        ops.ctc_backward_into(device, &kernel, UPSTREAM, &short, &state)
            .is_err(),
        "{name}: a short gradient grid must fail before launch"
    );
}
