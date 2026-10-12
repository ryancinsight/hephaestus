@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read> labels: array<u32>;
@group(0) @binding(3) var<storage, read> sample_meta: array<u32>;
@group(0) @binding(4) var<storage, read> alpha: array<f32>;
@group(0) @binding(5) var<storage, read> beta: array<f32>;
@group(0) @binding(6) var<storage, read> likelihood: array<f32>;
@group(0) @binding(7) var<storage, read> divisors: array<f32>;
@group(0) @binding(8) var<storage, read_write> grad: array<f32>;

// One lane per (sample, frame, class): sum the class's state occupancies
// in state order — subtracting the likelihood before adding, so a
// representable posterior never overflows — and accumulate the seeded,
// doubly normalized update.
@compute @workgroup_size(4, 4, 4)
fn ctc_posterior(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let t = global_id.y;
    let cls = global_id.z;
    if (b >= uniforms.dims.y || cls >= uniforms.dims.z) {
        return;
    }
    let frames = sample_meta[b * 4u];
    let states = sample_meta[b * 4u + 1u];
    let offset = sample_meta[b * 4u + 2u];
    let labels_offset = sample_meta[b * 4u + 3u];
    if (t >= frames) {
        return;
    }
    let ll = vec2<f32>(likelihood[b * 2u], likelihood[b * 2u + 1u]);
    let base = (offset + t * states) * 2u;
    var posterior = 0.0;
    for (var s = 0u; s < states; s++) {
        if (labels[labels_offset + s] != cls) {
            continue;
        }
        let a = vec2<f32>(alpha[base + s * 2u], alpha[base + s * 2u + 1u]);
        let bb = vec2<f32>(beta[base + s * 2u], beta[base + s * 2u + 1u]);
        if (is_neg_inf(a.x) || is_neg_inf(bb.x)) {
            continue;
        }
        posterior = posterior + exp(weight_value(weight_add(a, weight_sub(bb, ll))));
    }
    let update = ((-posterior * uniforms.scalars.x) / divisors[b]) / f32(uniforms.dims.y);
    let flat = (t * uniforms.dims.y + b) * uniforms.dims.z + cls;
    grad[flat] = grad[flat] + update;
}
