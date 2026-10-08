//! The host as a native CTC step backend.
//!
//! [`HostCtcOps`] runs the frame recurrences as plain CPU loops over the
//! host buffers rather than delegating to `leto_ops::CtcState`: the
//! provider exposes neither its per-frame recurrences nor its retained
//! messages, and smuggling a stateful provider handle through the
//! stateless kernel object would trade an honest implementation for
//! hidden mutation. The host is therefore a real step backend — the fifth
//! after WGSL, CUDA, HIP, and the conformance oracle's expectations —
//! whose arithmetic is line-for-line the provider's, so the conformance
//! suite holds it to the same contract on every run, no device required.
//!
//! A Weight is an `(high, low)` lane pair; every helper below reproduces
//! the provider's operation order, including the merge's low-lane
//! tie-break and the `ln(1 + exp(d))` correction through the same
//! `f32` entry points, so host results are bit-identical to the oracle.

use hephaestus_core::{
    CtcBackwardParams, CtcOps, CtcProblem, CtcStateBuffers, CtcStepOps, DeviceBuffer,
    HephaestusError, Result,
};

use crate::{HostBuffer, HostDevice};

/// A log weight and its rounding residual.
type Weight = (f32, f32);

const fn unreachable() -> Weight {
    (f32::NEG_INFINITY, 0.0)
}

fn is_unreachable(weight: Weight) -> bool {
    weight.0 == f32::NEG_INFINITY
}

/// Knuth TwoSum: the rounded sum and its residual.
fn w_sum(a: f32, b: f32) -> Weight {
    let high = a + b;
    let virtual_b = high - a;
    let low = (a - (high - virtual_b)) + (b - virtual_b);
    (high, low)
}

/// Compensated addition in the provider's `(self, other)` operand order.
fn w_add(x: Weight, y: Weight) -> Weight {
    if is_unreachable(x) || is_unreachable(y) {
        return unreachable();
    }
    let leading = w_sum(x.0, y.0);
    let tail = leading.1 + (x.1 + y.1);
    w_sum(leading.0, tail)
}

fn w_sub(x: Weight, y: Weight) -> Weight {
    w_add(x, (-y.0, -y.1))
}

fn w_value(weight: Weight) -> f32 {
    weight.0 + weight.1
}

/// Log-sum-exp merge with the provider's exact selection: strictly
/// greater high wins, ties break on the low lane.
fn w_merge(x: Weight, y: Weight) -> Weight {
    if is_unreachable(x) {
        return y;
    }
    if is_unreachable(y) {
        return x;
    }
    let (large, small) = if x.0 > y.0 || (x.0 == y.0 && x.1 >= y.1) {
        (x, y)
    } else {
        (y, x)
    };
    if small.0 - large.0 == f32::NEG_INFINITY {
        return large;
    }
    let difference = w_value(w_sub(small, large));
    let correction = (1.0 + difference.exp()).ln();
    w_add(large, (correction, 0.0))
}

fn read_pair(lanes: &[f32], cell: usize) -> Weight {
    (lanes[cell * 2], lanes[cell * 2 + 1])
}

fn write_pair(lanes: &mut [f32], cell: usize, weight: Weight) {
    lanes[cell * 2] = weight.0;
    lanes[cell * 2 + 1] = weight.1;
}

fn validate_step(
    problem: &CtcProblem,
    log_probs_len: usize,
    state: &CtcStateBuffers<HostDevice>,
) -> Result<()> {
    problem.validate_grid_len("log-probability grid", log_probs_len)?;
    problem.validate_state_lengths(
        state.labels.len(),
        state.meta.len(),
        state.alpha.len(),
        state.beta.len(),
        state.likelihood.len(),
        state.divisors.len(),
    )
}

/// Reject an output buffer sharing its allocation with any same-typed
/// input: taking a second guard on a held lock would deadlock, and
/// cross-typed buffers can never alias.
fn require_output_disjoint(output: &HostBuffer<f32>, inputs: &[&HostBuffer<f32>]) -> Result<()> {
    for input in inputs {
        if input.aliases(output) {
            return Err(HephaestusError::DispatchFailed {
                message: "ctc output buffer must not alias its inputs".to_string(),
            });
        }
    }
    Ok(())
}

/// CTC frame recurrences for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostCtcOps;

impl HostCtcOps {
    fn alpha_step(
        t: u32,
        log_probs: &HostBuffer<f32>,
        state: &CtcStateBuffers<HostDevice>,
    ) -> Result<()> {
        let problem = &state.problem;
        validate_step(problem, log_probs.len(), state)?;
        require_output_disjoint(&state.alpha, &[log_probs])?;
        let t = t as usize;
        let inputs = log_probs.read();
        let labels = state.labels.read();
        let mut alpha = state.alpha.write();
        let batch = problem.batch();
        let classes = problem.classes();
        for (b, sample) in problem.samples().iter().enumerate() {
            if t >= sample.frames {
                continue;
            }
            let labels_offset = sample.labels_offset;
            for s in 0..sample.states {
                let label = labels[labels_offset + s] as usize;
                let out = sample.offset + t * sample.states + s;
                if t == 0 {
                    let weight = if s < 2 {
                        (inputs[b * classes + label], 0.0)
                    } else {
                        unreachable()
                    };
                    write_pair(&mut alpha, out, weight);
                    continue;
                }
                let prev = sample.offset + (t - 1) * sample.states;
                let mut value = read_pair(&alpha, prev + s);
                if s > 0 {
                    value = w_merge(value, read_pair(&alpha, prev + s - 1));
                }
                if s > 1 && s % 2 == 1 && label != labels[labels_offset + s - 2] as usize {
                    value = w_merge(value, read_pair(&alpha, prev + s - 2));
                }
                let emission = inputs[(t * batch + b) * classes + label];
                write_pair(&mut alpha, out, w_add(value, (emission, 0.0)));
            }
        }
        Ok(())
    }

    fn beta_step(
        t: u32,
        log_probs: &HostBuffer<f32>,
        state: &CtcStateBuffers<HostDevice>,
    ) -> Result<()> {
        let problem = &state.problem;
        validate_step(problem, log_probs.len(), state)?;
        require_output_disjoint(&state.beta, &[log_probs])?;
        let t = t as usize;
        let inputs = log_probs.read();
        let labels = state.labels.read();
        let mut beta = state.beta.write();
        let batch = problem.batch();
        let classes = problem.classes();
        for (b, sample) in problem.samples().iter().enumerate() {
            if t >= sample.frames {
                continue;
            }
            let labels_offset = sample.labels_offset;
            let states = sample.states;
            for s in 0..states {
                let label = labels[labels_offset + s] as usize;
                let out = sample.offset + t * states + s;
                if t == sample.frames - 1 {
                    let weight = if s == states - 1 || s + 2 == states {
                        (0.0, 0.0)
                    } else {
                        unreachable()
                    };
                    write_pair(&mut beta, out, weight);
                    continue;
                }
                let next = sample.offset + (t + 1) * states;
                let emission = |state: usize| {
                    inputs[((t + 1) * batch + b) * classes + labels[labels_offset + state] as usize]
                };
                let mut value = w_add((emission(s), 0.0), read_pair(&beta, next + s));
                if s + 1 < states {
                    value = w_merge(
                        value,
                        w_add((emission(s + 1), 0.0), read_pair(&beta, next + s + 1)),
                    );
                }
                if s + 2 < states && s % 2 == 1 && label != labels[labels_offset + s + 2] as usize {
                    value = w_merge(
                        value,
                        w_add((emission(s + 2), 0.0), read_pair(&beta, next + s + 2)),
                    );
                }
                write_pair(&mut beta, out, value);
            }
        }
        Ok(())
    }
}

impl CtcStepOps<HostDevice> for HostCtcOps {
    /// The host compiles nothing; the loops run per dispatch.
    type CtcStep = ();

    fn prepare_ctc_step(&self, _device: &HostDevice) -> Result<Self::CtcStep> {
        Ok(())
    }

    fn ctc_alpha_step_into(
        &self,
        _device: &HostDevice,
        _kernel: &Self::CtcStep,
        t: u32,
        log_probs: &HostBuffer<f32>,
        state: &CtcStateBuffers<HostDevice>,
    ) -> Result<()> {
        Self::alpha_step(t, log_probs, state)
    }

    fn ctc_beta_step_into(
        &self,
        _device: &HostDevice,
        _kernel: &Self::CtcStep,
        t: u32,
        log_probs: &HostBuffer<f32>,
        state: &CtcStateBuffers<HostDevice>,
    ) -> Result<()> {
        Self::beta_step(t, log_probs, state)
    }

    fn ctc_loss_finish_into(
        &self,
        _device: &HostDevice,
        _kernel: &Self::CtcStep,
        state: &CtcStateBuffers<HostDevice>,
    ) -> Result<()> {
        let problem = &state.problem;
        problem.validate_state_lengths(
            state.labels.len(),
            state.meta.len(),
            state.alpha.len(),
            state.beta.len(),
            state.likelihood.len(),
            state.divisors.len(),
        )?;
        require_output_disjoint(&state.likelihood, &[&state.alpha])?;
        let _meta = state.meta.read();
        let alpha = state.alpha.read();
        let mut likelihood = state.likelihood.write();
        for (b, sample) in problem.samples().iter().enumerate() {
            debug_assert_eq!(_meta[b * 4] as usize, sample.frames);
            if sample.frames == 0 {
                let weight = if sample.states == 1 {
                    (0.0, 0.0)
                } else {
                    unreachable()
                };
                write_pair(&mut likelihood, b, weight);
                continue;
            }
            let term = sample.offset + (sample.frames - 1) * sample.states + sample.states - 1;
            let mut ll = read_pair(&alpha, term);
            if sample.states > 1 {
                ll = w_merge(ll, read_pair(&alpha, term - 1));
            }
            write_pair(&mut likelihood, b, ll);
        }
        Ok(())
    }

    fn ctc_posterior_into(
        &self,
        _device: &HostDevice,
        _kernel: &Self::CtcStep,
        upstream: f32,
        grad: &HostBuffer<f32>,
        state: &CtcStateBuffers<HostDevice>,
    ) -> Result<()> {
        let problem = &state.problem;
        problem.validate_grid_len("gradient grid", grad.len())?;
        problem.validate_state_lengths(
            state.labels.len(),
            state.meta.len(),
            state.alpha.len(),
            state.beta.len(),
            state.likelihood.len(),
            state.divisors.len(),
        )?;
        let params = CtcBackwardParams::new(problem, upstream)?;
        require_output_disjoint(
            grad,
            &[
                &state.alpha,
                &state.beta,
                &state.likelihood,
                &state.divisors,
            ],
        )?;
        let labels = state.labels.read();
        let _meta = state.meta.read();
        let alpha = state.alpha.read();
        let beta = state.beta.read();
        let likelihood = state.likelihood.read();
        let divisors = state.divisors.read();
        let mut grad = grad.write();
        let batch = problem.batch();
        let classes = problem.classes();
        let batch_divisor = params.dims[1] as f32;
        debug_assert_eq!(batch_divisor, batch as f32);
        for (b, sample) in problem.samples().iter().enumerate() {
            debug_assert_eq!(_meta[b * 4] as usize, sample.frames);
            let ll = read_pair(&likelihood, b);
            for t in 0..sample.frames {
                let base = sample.offset + t * sample.states;
                for class in 0..classes {
                    let mut posterior = 0.0_f32;
                    for s in 0..sample.states {
                        if labels[sample.labels_offset + s] as usize != class {
                            continue;
                        }
                        let a = read_pair(&alpha, base + s);
                        let bb = read_pair(&beta, base + s);
                        if is_unreachable(a) || is_unreachable(bb) {
                            continue;
                        }
                        posterior += w_value(w_add(a, w_sub(bb, ll))).exp();
                    }
                    let update = ((-posterior * params.scalars[0]) / divisors[b]) / batch_divisor;
                    let flat = (t * batch + b) * classes + class;
                    grad[flat] += update;
                }
            }
        }
        Ok(())
    }
}

impl CtcOps<HostDevice> for HostCtcOps {
    /// The host compiles nothing; the loops run per dispatch.
    type Ctc = ();

    fn prepare_ctc(&self, _device: &HostDevice) -> Result<Self::Ctc> {
        Ok(())
    }

    fn ctc_forward_into(
        &self,
        device: &HostDevice,
        kernel: &Self::Ctc,
        log_probs: &HostBuffer<f32>,
        state: &CtcStateBuffers<HostDevice>,
    ) -> Result<f32> {
        hephaestus_core::drive_ctc_forward(self, device, kernel, log_probs, state)
    }

    fn ctc_backward_into(
        &self,
        device: &HostDevice,
        kernel: &Self::Ctc,
        upstream: f32,
        grad: &HostBuffer<f32>,
        state: &CtcStateBuffers<HostDevice>,
    ) -> Result<()> {
        self.ctc_posterior_into(device, kernel, upstream, grad, state)
    }
}
