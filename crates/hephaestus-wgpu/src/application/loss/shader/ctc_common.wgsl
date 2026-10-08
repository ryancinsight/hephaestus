struct StepUniforms {
    dims: vec4<u32>,
};

struct Uniforms {
    dims: vec4<u32>,
    scalars: vec4<f32>,
};

const NEG_INF_BITS: u32 = 0xff800000u;

fn neg_inf() -> f32 {
    return bitcast<f32>(NEG_INF_BITS);
}

fn is_neg_inf(x: f32) -> bool {
    return bitcast<u32>(x) == NEG_INF_BITS;
}

// Knuth TwoSum: high is the rounded sum, low its residual.
fn weight_sum(a: f32, b: f32) -> vec2<f32> {
    let high = a + b;
    let virtual_b = high - a;
    let low = (a - (high - virtual_b)) + (b - virtual_b);
    return vec2<f32>(high, low);
}

// Compensated addition. Operand order matters — the residual chain is not
// symmetric — so callers keep the provider's (self, other) order. The
// provider errors on non-finite intermediates; the device requires the
// representable class and propagates the lanes.
fn weight_add(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    if (is_neg_inf(x.x) || is_neg_inf(y.x)) {
        return vec2<f32>(neg_inf(), 0.0);
    }
    let leading = weight_sum(x.x, y.x);
    let tail = leading.y + (x.y + y.y);
    return weight_sum(leading.x, tail);
}

fn weight_sub(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    return weight_add(x, vec2<f32>(-y.x, -y.y));
}

fn weight_value(x: vec2<f32>) -> f32 {
    return x.x + x.y;
}

// Log-sum-exp merge through ln(1 + exp(d)), with the provider's exact
// selection: strictly greater high wins, ties break on the low lane.
fn weight_merge(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    if (is_neg_inf(x.x)) {
        return y;
    }
    if (is_neg_inf(y.x)) {
        return x;
    }
    var large = x;
    var small = y;
    if (!(x.x > y.x || (x.x == y.x && x.y >= y.y))) {
        large = y;
        small = x;
    }
    if (is_neg_inf(small.x - large.x)) {
        return large;
    }
    let difference = weight_value(weight_sub(small, large));
    let correction = log(1.0 + exp(difference));
    return weight_add(large, vec2<f32>(correction, 0.0));
}
