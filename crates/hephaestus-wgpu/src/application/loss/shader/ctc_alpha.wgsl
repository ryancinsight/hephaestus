@group(0) @binding(0) var<storage, read> log_probs: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: StepUniforms;
@group(0) @binding(2) var<storage, read> labels: array<u32>;
@group(0) @binding(3) var<storage, read> sample_meta: array<u32>;
@group(0) @binding(4) var<storage, read_write> alpha: array<f32>;

fn emission(t: u32, b: u32, label: u32) -> f32 {
    return log_probs[(t * uniforms.dims.y + b) * uniforms.dims.z + label];
}

// One alpha frame: frame 0 initializes the reachable prefixes, later
// frames merge the stay, step, and skip predecessors in provider order
// and add the emission last.
@compute @workgroup_size(8, 8, 1)
fn ctc_alpha(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    if (t == 0u) {
        if (s < 2u) {
            alpha[out] = emission(0u, b, label);
            alpha[out + 1u] = 0.0;
        } else {
            alpha[out] = neg_inf();
            alpha[out + 1u] = 0.0;
        }
        return;
    }
    let prev = (offset + (t - 1u) * states) * 2u;
    var value = vec2<f32>(alpha[prev + s * 2u], alpha[prev + s * 2u + 1u]);
    if (s > 0u) {
        let below = vec2<f32>(alpha[prev + (s - 1u) * 2u], alpha[prev + (s - 1u) * 2u + 1u]);
        value = weight_merge(value, below);
    }
    if (s > 1u && s % 2u == 1u && label != labels[labels_offset + s - 2u]) {
        let skip = vec2<f32>(alpha[prev + (s - 2u) * 2u], alpha[prev + (s - 2u) * 2u + 1u]);
        value = weight_merge(value, skip);
    }
    value = weight_add(value, vec2<f32>(emission(t, b, label), 0.0));
    alpha[out] = value.x;
    alpha[out + 1u] = value.y;
}
