@group(0) @binding(1) var<uniform> uniforms: StepUniforms;
@group(0) @binding(3) var<storage, read> sample_meta: array<u32>;
@group(0) @binding(4) var<storage, read> alpha: array<f32>;
@group(0) @binding(6) var<storage, read_write> likelihood: array<f32>;

// One lane per sample: merge the terminal alpha pair, or take the
// empty-frame edge — log-zero for an empty target, unreachable otherwise.
@compute @workgroup_size(64, 1, 1)
fn ctc_loss_finish(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    if (b >= uniforms.dims.y) {
        return;
    }
    let frames = sample_meta[b * 4u];
    let states = sample_meta[b * 4u + 1u];
    let offset = sample_meta[b * 4u + 2u];
    let out = b * 2u;
    if (frames == 0u) {
        if (states == 1u) {
            likelihood[out] = 0.0;
            likelihood[out + 1u] = 0.0;
        } else {
            likelihood[out] = neg_inf();
            likelihood[out + 1u] = 0.0;
        }
        return;
    }
    let term = (offset + (frames - 1u) * states + states - 1u) * 2u;
    var ll = vec2<f32>(alpha[term], alpha[term + 1u]);
    if (states > 1u) {
        ll = weight_merge(ll, vec2<f32>(alpha[term - 2u], alpha[term - 1u]));
    }
    likelihood[out] = ll.x;
    likelihood[out + 1u] = ll.y;
}
