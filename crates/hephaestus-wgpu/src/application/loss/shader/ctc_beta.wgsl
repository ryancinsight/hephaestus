@group(0) @binding(0) var<storage, read> log_probs: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: StepUniforms;
@group(0) @binding(2) var<storage, read> labels: array<u32>;
@group(0) @binding(3) var<storage, read> sample_meta: array<u32>;
@group(0) @binding(5) var<storage, read_write> beta: array<f32>;

fn emission(t: u32, b: u32, label: u32) -> f32 {
    return log_probs[(t * uniforms.dims.y + b) * uniforms.dims.z + label];
}

fn beta_lane(frame_base: u32, s: u32) -> vec2<f32> {
    return vec2<f32>(beta[frame_base + s * 2u], beta[frame_base + s * 2u + 1u]);
}

// One beta frame, excluding the current emission: the terminal frame
// accepts the final two states at log-zero, earlier frames merge the
// stay, step, and skip successors with the *next* emission added first.
@compute @workgroup_size(8, 8, 1)
fn ctc_beta(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let s = global_id.y;
    let t = uniforms.dims.x;
    if (b >= uniforms.dims.y) {
        return;
    }
    let frames = sample_meta[b * 4u];
    let states = sample_meta[b * 4u + 1u];
    let offset = sample_meta[b * 4u + 2u];
    let labels_offset = sample_meta[b * 4u + 3u];
    if (s >= states || t >= frames) {
        return;
    }
    let label = labels[labels_offset + s];
    let out = (offset + t * states + s) * 2u;
    if (t == frames - 1u) {
        if (s == states - 1u || s + 2u == states) {
            beta[out] = 0.0;
            beta[out + 1u] = 0.0;
        } else {
            beta[out] = neg_inf();
            beta[out + 1u] = 0.0;
        }
        return;
    }
    let next = (offset + (t + 1u) * states) * 2u;
    var value = weight_add(vec2<f32>(emission(t + 1u, b, label), 0.0), beta_lane(next, s));
    if (s + 1u < states) {
        let step = weight_add(
            vec2<f32>(emission(t + 1u, b, labels[labels_offset + s + 1u]), 0.0),
            beta_lane(next, s + 1u)
        );
        value = weight_merge(value, step);
    }
    if (s + 2u < states && s % 2u == 1u && label != labels[labels_offset + s + 2u]) {
        let skip = weight_add(
            vec2<f32>(emission(t + 1u, b, labels[labels_offset + s + 2u]), 0.0),
            beta_lane(next, s + 2u)
        );
        value = weight_merge(value, skip);
    }
    beta[out] = value.x;
    beta[out + 1u] = value.y;
}
