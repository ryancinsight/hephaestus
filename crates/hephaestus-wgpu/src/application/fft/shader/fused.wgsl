struct FusedParams {
    n: u32,
    log2n: u32,
    inverse: u32,
    batch_count: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    axis: u32,
    batch_grid_x: u32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
}

@group(0) @binding(0)
var<storage, read_write> volume_re: array<f32>;
@group(0) @binding(1)
var<storage, read_write> volume_im: array<f32>;
@group(0) @binding(2)
var<storage, read> roots: array<f32>;
@group(0) @binding(3)
var<uniform> params: FusedParams;

const MAX_LENGTH: u32 = 1024u;
const MAX_HALF_LENGTH: u32 = MAX_LENGTH >> 1u;
const WORKGROUP_SIZE: u32 = 64u;

var<workgroup> values_re: array<f32, 1024>;
var<workgroup> values_im: array<f32, 1024>;
var<workgroup> roots_re: array<f32, 512>;
var<workgroup> roots_im: array<f32, 512>;

@compute @workgroup_size(64, 1, 1)
fn fft_fused_axis(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let batch = workgroup_id.x + workgroup_id.y * params.batch_grid_x;
    if batch >= params.batch_count {
        return;
    }

    let n = params.n;
    let half_n = n >> 1u;
    let root_stride = MAX_LENGTH / n;
    var index = local_id.x;
    loop {
        if index >= half_n {
            break;
        }
        let root_index = index * root_stride;
        roots_re[index] = roots[root_index];
        roots_im[index] = roots[MAX_HALF_LENGTH + root_index];
        index += WORKGROUP_SIZE;
    }

    var stride: u32;
    var base: u32;
    if params.axis == 0u {
        let y = batch / params.nz;
        let z = batch % params.nz;
        stride = params.ny * params.nz;
        base = y * params.nz + z;
    } else if params.axis == 1u {
        let x = batch / params.nz;
        let z = batch % params.nz;
        stride = params.nz;
        base = x * params.ny * params.nz + z;
    } else {
        let x = batch / params.ny;
        let y = batch % params.ny;
        stride = 1u;
        base = x * params.ny * params.nz + y * params.nz;
    }

    index = local_id.x;
    loop {
        if index >= n {
            break;
        }
        let reversed = reverseBits(index) >> (32u - params.log2n);
        let source = base + reversed * stride;
        values_re[index] = volume_re[source];
        values_im[index] = volume_im[source];
        index += WORKGROUP_SIZE;
    }
    workgroupBarrier();

    var stage = 0u;
    loop {
        if stage >= params.log2n {
            break;
        }
        let half_group = 1u << stage;
        let group_size = half_group << 1u;
        let twiddle_stride = n >> (stage + 1u);
        index = local_id.x;
        loop {
            if index >= half_n {
                break;
            }
            let group = index / half_group;
            let position = index % half_group;
            let even = group * group_size + position;
            let odd = even + half_group;
            let twiddle = position * twiddle_stride;
            let w_re = roots_re[twiddle];
            var w_im = roots_im[twiddle];
            if params.inverse != 0u {
                w_im = -w_im;
            }

            let e_re = values_re[even];
            let e_im = values_im[even];
            let o_re = values_re[odd];
            let o_im = values_im[odd];
            let product_re = w_re * o_re - w_im * o_im;
            let product_im = w_re * o_im + w_im * o_re;
            values_re[even] = e_re + product_re;
            values_im[even] = e_im + product_im;
            values_re[odd] = e_re - product_re;
            values_im[odd] = e_im - product_im;
            index += WORKGROUP_SIZE;
        }
        stage += 1u;
        workgroupBarrier();
    }

    index = local_id.x;
    let scale = select(1.0, 1.0 / f32(n), params.inverse != 0u);
    loop {
        if index >= n {
            break;
        }
        let destination = base + index * stride;
        volume_re[destination] = values_re[index] * scale;
        volume_im[destination] = values_im[index] * scale;
        index += WORKGROUP_SIZE;
    }
}
