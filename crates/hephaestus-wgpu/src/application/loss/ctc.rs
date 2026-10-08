//! Provider-owned connectionist temporal classification.
//!
//! The device counterpart of `leto_ops::CtcState`'s forward-backward loss.
//! One shared WGSL prelude serves all four entries — the alpha and beta
//! frame steps, the likelihood finish, and the posterior sweep — so the
//! compensated log-space arithmetic exists once and each entry pairs it
//! with exactly the bindings it touches. Like the cross-entropy loss beside
//! it the kernel is f32-only: WGSL does not guarantee f64 storage, and a
//! generic scalar would be a falsely generic boundary.
//!
//! # Bit-exactness against the CPU path, up to transcendentals
//!
//! Each lane reproduces the provider's arithmetic in the same operation
//! order — Knuth TwoSum with the exact `virtual_b` chain, the merge's
//! large/small selection including its low-lane tie-break, and the
//! `ln(1 + exp(d))` correction — with infinities spelled as bit patterns
//! so the unreachable checks are exact. WGSL arithmetic is IEEE without
//! implicit contraction, so the same rational operations round the same
//! way; only `exp` and `log` may differ from the host libm by a unit in
//! the last place, and denormal lanes may flush where the host keeps them.
//! The conformance suite compares against the provider oracle with a
//! tolerance sized for exactly that.
//!
//! # Why the steps validate on every frame
//!
//! The drive loop calls a step per frame, and each call re-checks its
//! buffer lengths against the problem. A dozen integer compares per
//! dispatch is nothing next to a kernel launch, and the alternative is a
//! silently mis-sized buffer producing robustness zeros the loss would
//! then launder into a plausible-looking number.

use hephaestus_core::{
    ComputeDevice, CtcBackwardParams, CtcProblem, CtcStateBuffers, CtcStepParams, DispatchGrid,
    HephaestusError, MultiStorageKernel, Result,
};
pub use hephaestus_core::{CtcOps, CtcStepOps};

use crate::application::storage_kernel::{
    WgslMultiStorageKernel, WgslStorageBinding, WgslStorageBindingLayout,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Workgroup shape covering the `[batch, states]` step domain.
const STEP_WORKGROUP: [usize; 3] = [8, 8, 1];
/// Workgroup shape covering the `[batch]` likelihood domain.
const FINISH_WORKGROUP: [usize; 3] = [64, 1, 1];
/// Workgroup shape covering the `[batch, frames, classes]` posterior domain.
const POSTERIOR_WORKGROUP: [usize; 3] = [4, 4, 4];

/// Compiled CTC step kernels.
///
/// Monomorphized at construction and reused across problems on the same
/// device; the frame and the seed ride in each dispatch's parameter block.
#[derive(Debug)]
pub struct CtcKernel {
    alpha: WgslMultiStorageKernel,
    beta: WgslMultiStorageKernel,
    loss_finish: WgslMultiStorageKernel,
    posterior: WgslMultiStorageKernel,
}

fn require_ctc_storage(device: &WgpuDevice) -> Result<()> {
    // The posterior binds seven storage buffers; WGPU 30 counts the uniform
    // buffer against the combined budget too.
    const REQUIRED_STORAGE: u32 = 7;
    const REQUIRED_COMBINED: u32 = 8;
    let limits = device.limits();
    if limits.max_storage_buffers_per_shader_stage >= REQUIRED_STORAGE
        && limits.max_buffers_and_acceleration_structures_per_shader_stage >= REQUIRED_COMBINED
    {
        Ok(())
    } else {
        Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "WGPU CTC requires {REQUIRED_STORAGE} storage buffers \
                 ({REQUIRED_COMBINED} combined with the uniform buffer) per shader stage"
            ),
        })
    }
}

impl CtcKernel {
    /// Compile the four CTC entries for a device.
    ///
    /// The posterior binds seven storage buffers plus the uniform block, so
    /// construction requires a device raised above the downlevel four; the
    /// steps alone would fit, but a loss without its backward pass is not a
    /// loss this backend offers.
    ///
    /// # Errors
    ///
    /// Returns `HephaestusError::InvalidConfiguration` when the device
    /// offers too few storage buffers, or `HephaestusError::DispatchFailed`
    /// when a WGSL source or binding layout is rejected by the device.
    pub fn new(device: &WgpuDevice) -> Result<Self> {
        require_ctc_storage(device)?;
        let step_bindings = [
            WgslStorageBindingLayout::read_only(0),
            WgslStorageBindingLayout::read_only(2),
            WgslStorageBindingLayout::read_only(3),
        ];
        let mut alpha_bindings = [WgslStorageBindingLayout::read_only(0); 4];
        alpha_bindings[..3].copy_from_slice(&step_bindings);
        alpha_bindings[3] = WgslStorageBindingLayout::read_write(4);
        let mut beta_bindings = [WgslStorageBindingLayout::read_only(0); 4];
        beta_bindings[..3].copy_from_slice(&step_bindings);
        beta_bindings[3] = WgslStorageBindingLayout::read_write(5);
        Ok(Self {
            alpha: WgslMultiStorageKernel::new(
                device,
                "hephaestus-ctc-alpha",
                CTC_ALPHA_SHADER,
                "ctc_alpha",
                &alpha_bindings,
                1,
            )?,
            beta: WgslMultiStorageKernel::new(
                device,
                "hephaestus-ctc-beta",
                CTC_BETA_SHADER,
                "ctc_beta",
                &beta_bindings,
                1,
            )?,
            loss_finish: WgslMultiStorageKernel::new(
                device,
                "hephaestus-ctc-loss-finish",
                CTC_LOSS_FINISH_SHADER,
                "ctc_loss_finish",
                &[
                    WgslStorageBindingLayout::read_only(3),
                    WgslStorageBindingLayout::read_only(4),
                    WgslStorageBindingLayout::read_write(6),
                ],
                1,
            )?,
            posterior: WgslMultiStorageKernel::new(
                device,
                "hephaestus-ctc-posterior",
                CTC_POSTERIOR_SHADER,
                "ctc_posterior",
                &[
                    WgslStorageBindingLayout::read_only(2),
                    WgslStorageBindingLayout::read_only(3),
                    WgslStorageBindingLayout::read_only(4),
                    WgslStorageBindingLayout::read_only(5),
                    WgslStorageBindingLayout::read_only(6),
                    WgslStorageBindingLayout::read_only(7),
                    WgslStorageBindingLayout::read_write(8),
                ],
                1,
            )?,
        })
    }

    fn validate_step(
        problem: &CtcProblem,
        log_probs_len: usize,
        state: &CtcStateBuffers<WgpuDevice>,
    ) -> Result<()> {
        problem.validate_grid_len("log-probability grid", log_probs_len)?;
        problem.validate_state_lengths(
            state.labels.len,
            state.meta.len,
            state.alpha.len,
            state.beta.len,
            state.likelihood.len,
            state.divisors.len,
        )
    }

    fn step_grid(problem: &CtcProblem) -> Result<DispatchGrid> {
        DispatchGrid::covering_domain([problem.batch(), problem.max_states(), 1], STEP_WORKGROUP)
    }

    /// Run the alpha recurrence at frame `t`, `log_probs` into `state`.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against the problem, or the
    /// backend dispatch failure.
    pub fn alpha_step(
        &self,
        device: &WgpuDevice,
        t: u32,
        log_probs: &WgpuBuffer<f32>,
        state: &CtcStateBuffers<WgpuDevice>,
    ) -> Result<()> {
        Self::validate_step(&state.problem, log_probs.len, state)?;
        self.alpha.dispatch(
            device,
            [
                WgslStorageBinding::new(0, log_probs),
                WgslStorageBinding::new(2, &state.labels),
                WgslStorageBinding::new(3, &state.meta),
                WgslStorageBinding::new(4, &state.alpha),
            ],
            &CtcStepParams::for_frame(&state.problem, t),
            Self::step_grid(&state.problem)?,
        )
    }

    /// Run the beta recurrence at frame `t`, `log_probs` into `state`.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against the problem, or the
    /// backend dispatch failure.
    pub fn beta_step(
        &self,
        device: &WgpuDevice,
        t: u32,
        log_probs: &WgpuBuffer<f32>,
        state: &CtcStateBuffers<WgpuDevice>,
    ) -> Result<()> {
        Self::validate_step(&state.problem, log_probs.len, state)?;
        self.beta.dispatch(
            device,
            [
                WgslStorageBinding::new(0, log_probs),
                WgslStorageBinding::new(2, &state.labels),
                WgslStorageBinding::new(3, &state.meta),
                WgslStorageBinding::new(5, &state.beta),
            ],
            &CtcStepParams::for_frame(&state.problem, t),
            Self::step_grid(&state.problem)?,
        )
    }

    /// Merge each sample's terminal alpha pair into its likelihood.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against the problem, or the
    /// backend dispatch failure.
    pub fn loss_finish(
        &self,
        device: &WgpuDevice,
        state: &CtcStateBuffers<WgpuDevice>,
    ) -> Result<()> {
        state.problem.validate_state_lengths(
            state.labels.len,
            state.meta.len,
            state.alpha.len,
            state.beta.len,
            state.likelihood.len,
            state.divisors.len,
        )?;
        // The frame slot is unused by the finish entry; zero keeps the
        // uniform block's meaning uniform across the step dispatches.
        let params = CtcStepParams::for_frame(&state.problem, 0);
        let grid = DispatchGrid::covering_domain([state.problem.batch(), 1, 1], FINISH_WORKGROUP)?;
        // A degenerate problem leaves alpha empty, which the API refuses to
        // bind; every lane takes the empty-frame branch without reading
        // alpha, so a one-lane stand-in is sound.
        let stand_in;
        let alpha = if state.alpha.len == 0 {
            stand_in = device.alloc_zeroed::<f32>(1)?;
            &stand_in
        } else {
            &state.alpha
        };
        self.loss_finish.dispatch(
            device,
            [
                WgslStorageBinding::new(3, &state.meta),
                WgslStorageBinding::new(4, alpha),
                WgslStorageBinding::new(6, &state.likelihood),
            ],
            &params,
            grid,
        )
    }

    /// Accumulate the seeded posterior into `grad` from retained `state`.
    ///
    /// Must not run when forward reported a non-finite loss.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against the problem, a non-finite
    /// seed, or the backend dispatch failure.
    pub fn posterior(
        &self,
        device: &WgpuDevice,
        upstream: f32,
        grad: &WgpuBuffer<f32>,
        state: &CtcStateBuffers<WgpuDevice>,
    ) -> Result<()> {
        state.problem.validate_grid_len("gradient grid", grad.len)?;
        state.problem.validate_state_lengths(
            state.labels.len,
            state.meta.len,
            state.alpha.len,
            state.beta.len,
            state.likelihood.len,
            state.divisors.len,
        )?;
        let params = CtcBackwardParams::new(&state.problem, upstream)?;
        let grid = DispatchGrid::covering_domain(
            [
                state.problem.batch(),
                state.problem.max_frames(),
                state.problem.classes(),
            ],
            POSTERIOR_WORKGROUP,
        )?;
        self.posterior.dispatch(
            device,
            [
                WgslStorageBinding::new(2, &state.labels),
                WgslStorageBinding::new(3, &state.meta),
                WgslStorageBinding::new(4, &state.alpha),
                WgslStorageBinding::new(5, &state.beta),
                WgslStorageBinding::new(6, &state.likelihood),
                WgslStorageBinding::new(7, &state.divisors),
                WgslStorageBinding::new(8, grad),
            ],
            &params,
            grid,
        )
    }
}

// Each entry pairs the shared compensated-log-space prelude with exactly
// the bindings it touches, assembled at compile time so the Weight
// operations exist in one file.
const CTC_ALPHA_SHADER: &str = concat!(
    include_str!("shader/ctc_common.wgsl"),
    include_str!("shader/ctc_alpha.wgsl"),
);

const CTC_BETA_SHADER: &str = concat!(
    include_str!("shader/ctc_common.wgsl"),
    include_str!("shader/ctc_beta.wgsl"),
);

const CTC_LOSS_FINISH_SHADER: &str = concat!(
    include_str!("shader/ctc_common.wgsl"),
    include_str!("shader/ctc_loss_finish.wgsl"),
);

const CTC_POSTERIOR_SHADER: &str = concat!(
    include_str!("shader/ctc_common.wgsl"),
    include_str!("shader/ctc_posterior.wgsl"),
);

// Each entry binds exactly the uniform block its Rust params upload:
// `StepUniforms` for the steps, the full `Uniforms` for the posterior.

/// Provider-owned implementation of [`CtcStepOps`] and [`CtcOps`] for WGPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuCtcOps;

impl CtcStepOps<WgpuDevice> for WgpuCtcOps {
    type CtcStep = CtcKernel;

    fn prepare_ctc_step(&self, device: &WgpuDevice) -> Result<Self::CtcStep> {
        CtcKernel::new(device)
    }

    fn ctc_alpha_step_into(
        &self,
        device: &WgpuDevice,
        kernel: &Self::CtcStep,
        t: u32,
        log_probs: &WgpuBuffer<f32>,
        state: &CtcStateBuffers<WgpuDevice>,
    ) -> Result<()> {
        kernel.alpha_step(device, t, log_probs, state)
    }

    fn ctc_beta_step_into(
        &self,
        device: &WgpuDevice,
        kernel: &Self::CtcStep,
        t: u32,
        log_probs: &WgpuBuffer<f32>,
        state: &CtcStateBuffers<WgpuDevice>,
    ) -> Result<()> {
        kernel.beta_step(device, t, log_probs, state)
    }

    fn ctc_loss_finish_into(
        &self,
        device: &WgpuDevice,
        kernel: &Self::CtcStep,
        state: &CtcStateBuffers<WgpuDevice>,
    ) -> Result<()> {
        kernel.loss_finish(device, state)
    }

    fn ctc_posterior_into(
        &self,
        device: &WgpuDevice,
        kernel: &Self::CtcStep,
        upstream: f32,
        grad: &WgpuBuffer<f32>,
        state: &CtcStateBuffers<WgpuDevice>,
    ) -> Result<()> {
        kernel.posterior(device, upstream, grad, state)
    }
}

impl CtcOps<WgpuDevice> for WgpuCtcOps {
    type Ctc = CtcKernel;

    fn prepare_ctc(&self, device: &WgpuDevice) -> Result<Self::Ctc> {
        CtcKernel::new(device)
    }

    fn ctc_forward_into(
        &self,
        device: &WgpuDevice,
        kernel: &Self::Ctc,
        log_probs: &WgpuBuffer<f32>,
        state: &CtcStateBuffers<WgpuDevice>,
    ) -> Result<f32> {
        hephaestus_core::drive_ctc_forward(self, device, kernel, log_probs, state)
    }

    fn ctc_backward_into(
        &self,
        device: &WgpuDevice,
        kernel: &Self::Ctc,
        upstream: f32,
        grad: &WgpuBuffer<f32>,
        state: &CtcStateBuffers<WgpuDevice>,
    ) -> Result<()> {
        kernel.posterior(device, upstream, grad, state)
    }
}
