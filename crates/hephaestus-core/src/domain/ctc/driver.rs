use super::invalid;
use super::seams::CtcStepOps;
use super::state::CtcStateBuffers;
use crate::{ComputeDevice, Result};

/// Walk the shared forward pass over any step implementation.
///
/// Alpha sweeps frames upward, beta downward, the likelihood finish merges
/// the terminal pairs, and the host sums the per-sample contributions in
/// sample order — the provider's accumulation with its overflow checks.
///
/// # Errors
///
/// Returns the step dispatch failure, the likelihood readback failure, or a
/// non-finite accumulation the provider would equally reject.
pub fn drive_ctc_forward<D, S>(
    ops: &S,
    device: &D,
    kernel: &S::CtcStep,
    log_probs: &D::Buffer<f32>,
    state: &CtcStateBuffers<D>,
) -> Result<f32>
where
    D: ComputeDevice,
    S: CtcStepOps<D>,
{
    let frames = state.problem.frames();
    for t in 0..frames {
        let t = u32::try_from(t).map_err(|_| invalid(format!("ctc frame {t} does not fit u32")))?;
        ops.ctc_alpha_step_into(device, kernel, t, log_probs, state)?;
    }
    for t in (0..frames).rev() {
        let t = u32::try_from(t).map_err(|_| invalid(format!("ctc frame {t} does not fit u32")))?;
        ops.ctc_beta_step_into(device, kernel, t, log_probs, state)?;
    }
    ops.ctc_loss_finish_into(device, kernel, state)?;
    let batch = state.problem.batch();
    let mut likelihood = vec![0.0_f32; batch * 2];
    device.download(&state.likelihood, &mut likelihood)?;
    let batch_divisor = state.problem.batch_divisor();
    let mut loss = 0.0_f32;
    for sample in 0..batch {
        let high = likelihood[sample * 2];
        let value = high + likelihood[sample * 2 + 1];
        let divisor = state.problem.divisors()[sample];
        let contribution = (-value / divisor) / batch_divisor;
        // Reachability is the high lane exactly, as the provider checks it:
        // only a negative-infinity high lane takes the unreachable path.
        let reachable = high != f32::NEG_INFINITY;
        if reachable && !contribution.is_finite() {
            return Err(invalid(format!(
                "ctc sample {sample} contribution is not finite"
            )));
        }
        let previous = loss;
        loss += contribution;
        if loss.is_nan() || (previous.is_finite() && contribution.is_finite() && !loss.is_finite())
        {
            return Err(invalid(format!("ctc sample {sample} loss is not finite")));
        }
    }
    Ok(loss)
}
