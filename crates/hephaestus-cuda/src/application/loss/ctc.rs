//! CUDA connectionist temporal classification.
//!
//! The CUDA counterpart of the WGSL forward-backward loss, computing the
//! same compensated log-space recurrences from the same problem buffers.
//! One CUDA source serves all four entries — the alpha and beta frame
//! steps, the likelihood finish, and the posterior sweep — and the
//! per-lane arithmetic keeps the provider's exact operation order. `f32`
//! only, matching the uniform block layout the steps share with WGSL.
//!
//! # Bit-exactness against the CPU path, up to transcendentals
//!
//! NVRTC compiles without fast-math, and the kernels hold no
//! multiply-add pair for contraction to fuse — the TwoSum chains add and
//! subtract only, and the posterior's lone product feeds divisions — so
//! the same rational operations round the same way as on the host. Only
//! `expf` and `logf` may differ from the host libm by a unit in the last
//! place. Infinities are spelled as bit patterns through
//! `__int_as_float`, so the unreachable checks are exact on every
//! architecture.

use hephaestus_core::{
    CtcBackwardParams, CtcProblem, CtcStateBuffers, CtcStepParams, DeviceBuffer, DispatchGrid,
    MultiStorageKernel, Result,
};
pub use hephaestus_core::{CtcOps, CtcStepOps};

use crate::CudaDevice;
use crate::application::storage_kernel::{CudaMultiStorageKernel, CudaStorageBinding};
use crate::infrastructure::buffer::CudaBuffer;

/// Block shape covering the `[batch, states]` step domain.
const STEP_BLOCK: [u32; 3] = [8, 8, 1];
/// Block shape covering the `[batch]` likelihood domain.
const FINISH_BLOCK: [u32; 3] = [64, 1, 1];
/// Block shape covering the `[batch, frames, classes]` posterior domain.
const POSTERIOR_BLOCK: [u32; 3] = [4, 4, 4];

const CTC_KERNEL: &str = r#"
struct CtcStepParams {
    unsigned int dims[4];
};

struct CtcBackwardParams {
    unsigned int dims[4];
    float scalars[4];
};

__device__ __forceinline__ float ctc_neg_inf() {
    return __int_as_float(0xff800000u);
}

__device__ __forceinline__ bool ctc_is_neg_inf(float x) {
    return __float_as_int(x) == (int)0xff800000u;
}

// A Weight is a float2 of (high, low); every helper reproduces the
// provider's operation order, including the merge's low-lane tie-break.
__device__ __forceinline__ float2 ctc_sum(float a, float b) {
    float high = a + b;
    float virtual_b = high - a;
    float low = (a - (high - virtual_b)) + (b - virtual_b);
    return make_float2(high, low);
}

__device__ __forceinline__ float2 ctc_add(float2 x, float2 y) {
    if (ctc_is_neg_inf(x.x) || ctc_is_neg_inf(y.x)) {
        return make_float2(ctc_neg_inf(), 0.0f);
    }
    float2 leading = ctc_sum(x.x, y.x);
    float tail = leading.y + (x.y + y.y);
    return ctc_sum(leading.x, tail);
}

__device__ __forceinline__ float2 ctc_sub(float2 x, float2 y) {
    return ctc_add(x, make_float2(-y.x, -y.y));
}

__device__ __forceinline__ float ctc_value(float2 x) {
    return x.x + x.y;
}

__device__ __forceinline__ float2 ctc_merge(float2 x, float2 y) {
    if (ctc_is_neg_inf(x.x)) {
        return y;
    }
    if (ctc_is_neg_inf(y.x)) {
        return x;
    }
    float2 large = x;
    float2 small = y;
    if (!(x.x > y.x || (x.x == y.x && x.y >= y.y))) {
        large = y;
        small = x;
    }
    if (ctc_is_neg_inf(small.x - large.x)) {
        return large;
    }
    float difference = ctc_value(ctc_sub(small, large));
    float correction = logf(1.0f + expf(difference));
    return ctc_add(large, make_float2(correction, 0.0f));
}

__device__ __forceinline__ float ctc_emission(
    const float* log_probs, unsigned int t, unsigned int b, unsigned int label,
    unsigned int batch, unsigned int classes
) {
    return log_probs[(t * batch + b) * classes + label];
}

// One alpha frame: frame 0 initializes the reachable prefixes, later
// frames merge the stay, step, and skip predecessors in provider order
// and add the emission last.
extern "C" __global__ void ctc_alpha(
    const float* log_probs,
    const unsigned int* labels,
    const unsigned int* meta,
    float* alpha,
    CtcStepParams params
) {
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int s = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int t = params.dims[0];
    if (b >= params.dims[1]) {
        return;
    }
    unsigned int frames = meta[b * 4u];
    unsigned int states = meta[b * 4u + 1u];
    unsigned int offset = meta[b * 4u + 2u];
    unsigned int labels_offset = meta[b * 4u + 3u];
    if (s >= states || t >= frames) {
        return;
    }
    unsigned int label = labels[labels_offset + s];
    unsigned int out = (offset + t * states + s) * 2u;
    unsigned int batch = params.dims[1];
    unsigned int classes = params.dims[2];
    if (t == 0u) {
        if (s < 2u) {
            alpha[out] = ctc_emission(log_probs, 0u, b, label, batch, classes);
            alpha[out + 1u] = 0.0f;
        } else {
            alpha[out] = ctc_neg_inf();
            alpha[out + 1u] = 0.0f;
        }
        return;
    }
    unsigned int prev = (offset + (t - 1u) * states) * 2u;
    float2 value = make_float2(alpha[prev + s * 2u], alpha[prev + s * 2u + 1u]);
    if (s > 0u) {
        float2 below = make_float2(alpha[prev + (s - 1u) * 2u], alpha[prev + (s - 1u) * 2u + 1u]);
        value = ctc_merge(value, below);
    }
    if (s > 1u && s % 2u == 1u && label != labels[labels_offset + s - 2u]) {
        float2 skip = make_float2(alpha[prev + (s - 2u) * 2u], alpha[prev + (s - 2u) * 2u + 1u]);
        value = ctc_merge(value, skip);
    }
    value = ctc_add(value, make_float2(ctc_emission(log_probs, t, b, label, batch, classes), 0.0f));
    alpha[out] = value.x;
    alpha[out + 1u] = value.y;
}

// One beta frame, excluding the current emission: the terminal frame
// accepts the final two states at log-zero, earlier frames merge the
// stay, step, and skip successors with the *next* emission added first.
extern "C" __global__ void ctc_beta(
    const float* log_probs,
    const unsigned int* labels,
    const unsigned int* meta,
    float* beta,
    CtcStepParams params
) {
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int s = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int t = params.dims[0];
    if (b >= params.dims[1]) {
        return;
    }
    unsigned int frames = meta[b * 4u];
    unsigned int states = meta[b * 4u + 1u];
    unsigned int offset = meta[b * 4u + 2u];
    unsigned int labels_offset = meta[b * 4u + 3u];
    if (s >= states || t >= frames) {
        return;
    }
    unsigned int label = labels[labels_offset + s];
    unsigned int out = (offset + t * states + s) * 2u;
    unsigned int batch = params.dims[1];
    unsigned int classes = params.dims[2];
    if (t == frames - 1u) {
        if (s == states - 1u || s + 2u == states) {
            beta[out] = 0.0f;
            beta[out + 1u] = 0.0f;
        } else {
            beta[out] = ctc_neg_inf();
            beta[out + 1u] = 0.0f;
        }
        return;
    }
    unsigned int next = (offset + (t + 1u) * states) * 2u;
    float2 value = ctc_add(
        make_float2(ctc_emission(log_probs, t + 1u, b, label, batch, classes), 0.0f),
        make_float2(beta[next + s * 2u], beta[next + s * 2u + 1u])
    );
    if (s + 1u < states) {
        float2 step = ctc_add(
            make_float2(ctc_emission(log_probs, t + 1u, b, labels[labels_offset + s + 1u], batch, classes), 0.0f),
            make_float2(beta[next + (s + 1u) * 2u], beta[next + (s + 1u) * 2u + 1u])
        );
        value = ctc_merge(value, step);
    }
    if (s + 2u < states && s % 2u == 1u && label != labels[labels_offset + s + 2u]) {
        float2 skip = ctc_add(
            make_float2(ctc_emission(log_probs, t + 1u, b, labels[labels_offset + s + 2u], batch, classes), 0.0f),
            make_float2(beta[next + (s + 2u) * 2u], beta[next + (s + 2u) * 2u + 1u])
        );
        value = ctc_merge(value, skip);
    }
    beta[out] = value.x;
    beta[out + 1u] = value.y;
}

// One lane per sample: merge the terminal alpha pair, or take the
// empty-frame edge — log-zero for an empty target, unreachable otherwise.
extern "C" __global__ void ctc_loss_finish(
    const unsigned int* meta,
    const float* alpha,
    float* likelihood,
    CtcStepParams params
) {
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= params.dims[1]) {
        return;
    }
    unsigned int frames = meta[b * 4u];
    unsigned int states = meta[b * 4u + 1u];
    unsigned int offset = meta[b * 4u + 2u];
    unsigned int out = b * 2u;
    if (frames == 0u) {
        if (states == 1u) {
            likelihood[out] = 0.0f;
            likelihood[out + 1u] = 0.0f;
        } else {
            likelihood[out] = ctc_neg_inf();
            likelihood[out + 1u] = 0.0f;
        }
        return;
    }
    unsigned int term = (offset + (frames - 1u) * states + states - 1u) * 2u;
    float2 ll = make_float2(alpha[term], alpha[term + 1u]);
    if (states > 1u) {
        ll = ctc_merge(ll, make_float2(alpha[term - 2u], alpha[term - 1u]));
    }
    likelihood[out] = ll.x;
    likelihood[out + 1u] = ll.y;
}

// One lane per (sample, frame, class): sum the class's state occupancies
// in state order — subtracting the likelihood before adding, so a
// representable posterior never overflows — and accumulate the seeded,
// doubly normalized update.
extern "C" __global__ void ctc_posterior(
    const unsigned int* labels,
    const unsigned int* meta,
    const float* alpha,
    const float* beta,
    const float* likelihood,
    const float* divisors,
    float* grad,
    CtcBackwardParams params
) {
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int t = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int cls = blockIdx.z * blockDim.z + threadIdx.z;
    if (b >= params.dims[1] || cls >= params.dims[2]) {
        return;
    }
    unsigned int frames = meta[b * 4u];
    unsigned int states = meta[b * 4u + 1u];
    unsigned int offset = meta[b * 4u + 2u];
    unsigned int labels_offset = meta[b * 4u + 3u];
    if (t >= frames) {
        return;
    }
    float2 ll = make_float2(likelihood[b * 2u], likelihood[b * 2u + 1u]);
    unsigned int base = (offset + t * states) * 2u;
    float posterior = 0.0f;
    for (unsigned int s = 0u; s < states; ++s) {
        if (labels[labels_offset + s] != cls) {
            continue;
        }
        float2 a = make_float2(alpha[base + s * 2u], alpha[base + s * 2u + 1u]);
        float2 bb = make_float2(beta[base + s * 2u], beta[base + s * 2u + 1u]);
        if (ctc_is_neg_inf(a.x) || ctc_is_neg_inf(bb.x)) {
            continue;
        }
        posterior = posterior + expf(ctc_value(ctc_add(a, ctc_sub(bb, ll))));
    }
    float update = ((-posterior * params.scalars[0]) / divisors[b]) / (float)params.dims[1];
    unsigned int flat = (t * params.dims[1] + b) * params.dims[2] + cls;
    grad[flat] = grad[flat] + update;
}
"#;

/// Compiled CUDA CTC step kernels.
///
/// Built once per device and reused across problems; the frame and the
/// seed ride in each launch's parameter block.
#[derive(Debug)]
pub struct CtcKernel {
    alpha: CudaMultiStorageKernel,
    beta: CudaMultiStorageKernel,
    loss_finish: CudaMultiStorageKernel,
    posterior: CudaMultiStorageKernel,
}

impl CtcKernel {
    /// Compile the four CTC entries for a CUDA device.
    ///
    /// # Errors
    ///
    /// Returns the runtime compiler or module-load failure.
    pub fn new(_device: &CudaDevice) -> Result<Self> {
        Ok(Self {
            alpha: CudaMultiStorageKernel::new(
                "hephaestus-ctc-alpha",
                CTC_KERNEL,
                "ctc_alpha",
                &[0, 1, 2, 3],
                STEP_BLOCK,
                0,
            )?,
            beta: CudaMultiStorageKernel::new(
                "hephaestus-ctc-beta",
                CTC_KERNEL,
                "ctc_beta",
                &[0, 1, 2, 3],
                STEP_BLOCK,
                0,
            )?,
            loss_finish: CudaMultiStorageKernel::new(
                "hephaestus-ctc-loss-finish",
                CTC_KERNEL,
                "ctc_loss_finish",
                &[0, 1, 2],
                FINISH_BLOCK,
                0,
            )?,
            posterior: CudaMultiStorageKernel::new(
                "hephaestus-ctc-posterior",
                CTC_KERNEL,
                "ctc_posterior",
                &[0, 1, 2, 3, 4, 5, 6],
                POSTERIOR_BLOCK,
                0,
            )?,
        })
    }

    fn validate_step(
        problem: &CtcProblem,
        log_probs_len: usize,
        state: &CtcStateBuffers<CudaDevice>,
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

    fn step_grid(problem: &CtcProblem) -> Result<DispatchGrid> {
        DispatchGrid::covering_domain(
            [problem.batch(), problem.max_states(), 1],
            [
                STEP_BLOCK[0] as usize,
                STEP_BLOCK[1] as usize,
                STEP_BLOCK[2] as usize,
            ],
        )
    }

    /// Run the alpha recurrence at frame `t`, `log_probs` into `state`.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against the problem, or the
    /// launch failure.
    pub fn alpha_step(
        &self,
        device: &CudaDevice,
        t: u32,
        log_probs: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        Self::validate_step(&state.problem, log_probs.len(), state)?;
        MultiStorageKernel::<CudaDevice, CtcStepParams, [CudaStorageBinding<'_>; 4]>::dispatch(
            &self.alpha,
            device,
            [
                CudaStorageBinding::new(0, log_probs),
                CudaStorageBinding::new(1, &state.labels),
                CudaStorageBinding::new(2, &state.meta),
                CudaStorageBinding::new(3, &state.alpha),
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
    /// launch failure.
    pub fn beta_step(
        &self,
        device: &CudaDevice,
        t: u32,
        log_probs: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        Self::validate_step(&state.problem, log_probs.len(), state)?;
        MultiStorageKernel::<CudaDevice, CtcStepParams, [CudaStorageBinding<'_>; 4]>::dispatch(
            &self.beta,
            device,
            [
                CudaStorageBinding::new(0, log_probs),
                CudaStorageBinding::new(1, &state.labels),
                CudaStorageBinding::new(2, &state.meta),
                CudaStorageBinding::new(3, &state.beta),
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
    /// launch failure.
    pub fn loss_finish(
        &self,
        device: &CudaDevice,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        state.problem.validate_state_lengths(
            state.labels.len(),
            state.meta.len(),
            state.alpha.len(),
            state.beta.len(),
            state.likelihood.len(),
            state.divisors.len(),
        )?;
        let params = CtcStepParams::for_frame(&state.problem, 0);
        let grid = DispatchGrid::covering_domain(
            [state.problem.batch(), 1, 1],
            [
                FINISH_BLOCK[0] as usize,
                FINISH_BLOCK[1] as usize,
                FINISH_BLOCK[2] as usize,
            ],
        )?;
        MultiStorageKernel::<CudaDevice, CtcStepParams, [CudaStorageBinding<'_>; 3]>::dispatch(
            &self.loss_finish,
            device,
            [
                CudaStorageBinding::new(0, &state.meta),
                CudaStorageBinding::new(1, &state.alpha),
                CudaStorageBinding::new(2, &state.likelihood),
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
    /// seed, or the launch failure.
    pub fn posterior(
        &self,
        device: &CudaDevice,
        upstream: f32,
        grad: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        state
            .problem
            .validate_grid_len("gradient grid", grad.len())?;
        state.problem.validate_state_lengths(
            state.labels.len(),
            state.meta.len(),
            state.alpha.len(),
            state.beta.len(),
            state.likelihood.len(),
            state.divisors.len(),
        )?;
        let params = CtcBackwardParams::new(&state.problem, upstream)?;
        let grid = DispatchGrid::covering_domain(
            [
                state.problem.batch(),
                state.problem.max_frames(),
                state.problem.classes(),
            ],
            [
                POSTERIOR_BLOCK[0] as usize,
                POSTERIOR_BLOCK[1] as usize,
                POSTERIOR_BLOCK[2] as usize,
            ],
        )?;
        MultiStorageKernel::<CudaDevice, CtcBackwardParams, [CudaStorageBinding<'_>; 7]>::dispatch(
            &self.posterior,
            device,
            [
                CudaStorageBinding::new(0, &state.labels),
                CudaStorageBinding::new(1, &state.meta),
                CudaStorageBinding::new(2, &state.alpha),
                CudaStorageBinding::new(3, &state.beta),
                CudaStorageBinding::new(4, &state.likelihood),
                CudaStorageBinding::new(5, &state.divisors),
                CudaStorageBinding::new(6, grad),
            ],
            &params,
            grid,
        )
    }
}

/// Provider-owned implementation of [`CtcStepOps`] and [`CtcOps`] for CUDA.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaCtcOps;

impl CtcStepOps<CudaDevice> for CudaCtcOps {
    type CtcStep = CtcKernel;

    fn prepare_ctc_step(&self, device: &CudaDevice) -> Result<Self::CtcStep> {
        CtcKernel::new(device)
    }

    fn ctc_alpha_step_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::CtcStep,
        t: u32,
        log_probs: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        kernel.alpha_step(device, t, log_probs, state)
    }

    fn ctc_beta_step_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::CtcStep,
        t: u32,
        log_probs: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        kernel.beta_step(device, t, log_probs, state)
    }

    fn ctc_loss_finish_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::CtcStep,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        kernel.loss_finish(device, state)
    }

    fn ctc_posterior_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::CtcStep,
        upstream: f32,
        grad: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        kernel.posterior(device, upstream, grad, state)
    }
}

impl CtcOps<CudaDevice> for CudaCtcOps {
    type Ctc = CtcKernel;

    fn prepare_ctc(&self, device: &CudaDevice) -> Result<Self::Ctc> {
        CtcKernel::new(device)
    }

    fn ctc_forward_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::Ctc,
        log_probs: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<f32> {
        hephaestus_core::drive_ctc_forward(self, device, kernel, log_probs, state)
    }

    fn ctc_backward_into(
        &self,
        device: &CudaDevice,
        kernel: &Self::Ctc,
        upstream: f32,
        grad: &CudaBuffer<f32>,
        state: &CtcStateBuffers<CudaDevice>,
    ) -> Result<()> {
        kernel.posterior(device, upstream, grad, state)
    }
}
