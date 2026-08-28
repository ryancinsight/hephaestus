struct ChirpParams {
    n: u32,
    m: u32,
    batch_count: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read_write> data_re: array<{{scalar}}>;
@group(0) @binding(1)
var<storage, read_write> data_im: array<{{scalar}}>;
@group(0) @binding(2)
var<storage, read> chirp_re: array<{{scalar}}>;
@group(0) @binding(3)
var<storage, read> chirp_im: array<{{scalar}}>;

@group(0) @binding(4)
var<uniform> params: ChirpParams;

@compute @workgroup_size(256, 1, 1)
fn chirp_premul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.m * params.batch_count;
    if idx >= total {
        return;
    }
    let local_idx = idx % params.m;

    if local_idx >= params.n {
        data_re[idx] = 0.0;
        data_im[idx] = 0.0;
        return;
    }

    let re = data_re[idx];
    let im = data_im[idx];
    let factor_re = chirp_re[local_idx];
    let factor_im = chirp_im[local_idx];
    data_re[idx] = re * factor_re - im * factor_im;
    data_im[idx] = re * factor_im + im * factor_re;
}

@compute @workgroup_size(256, 1, 1)
fn chirp_pointmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.m * params.batch_count;
    if idx >= total {
        return;
    }

    let a_re = data_re[idx];
    let a_im = data_im[idx];
    let local_idx = idx % params.m;
    let h_re = chirp_re[local_idx];
    let h_im = chirp_im[local_idx];

    data_re[idx] = a_re * h_re - a_im * h_im;
    data_im[idx] = a_re * h_im + a_im * h_re;
}

@compute @workgroup_size(256, 1, 1)
fn chirp_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_idx = gid.x;
    let total = params.n * params.batch_count;
    if linear_idx >= total {
        return;
    }
    let row = linear_idx / params.n;
    let local_idx = linear_idx % params.n;
    let idx = row * params.m + local_idx;
    let inv_n = chirp_re[params.n];
    data_re[idx] = data_re[idx] * inv_n;
    data_im[idx] = data_im[idx] * inv_n;
}

@compute @workgroup_size(256, 1, 1)
fn chirp_postmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_idx = gid.x;
    let total = params.n * params.batch_count;
    if linear_idx >= total {
        return;
    }
    let row = linear_idx / params.n;
    let local_idx = linear_idx % params.n;
    let idx = row * params.m + local_idx;

    let re = data_re[idx];
    let im = data_im[idx];
    let factor_re = chirp_re[local_idx];
    let factor_im = chirp_im[local_idx];
    data_re[idx] = re * factor_re - im * factor_im;
    data_im[idx] = re * factor_im + im * factor_re;
}

@compute @workgroup_size(256, 1, 1)
fn chirp_negate_im(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_idx = gid.x;
    let total = params.n * params.batch_count;
    if linear_idx >= total {
        return;
    }
    let row = linear_idx / params.n;
    let local_idx = linear_idx % params.n;
    let idx = row * params.m + local_idx;
    data_im[idx] = -data_im[idx];
}
