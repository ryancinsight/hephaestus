// ---------------------------------------------------------------------------
// Q accumulation uniform
// ---------------------------------------------------------------------------

/// Packed metadata for the **Q** accumulation kernel.
#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
pub(super) struct QMeta {
    /// Row count *m* of the factored matrix, and the order of **Q**.
    pub(super) m: u32,
    /// Column count *n*, the row stride of the packed factor.
    pub(super) n: u32,
    /// Number of stored reflectors, `min(m, n)`.
    pub(super) reflector_count: u32,
}

/// Per-reflector scalars the packed factor does not carry.
///
/// The head `v_k[k]` is displaced from the packed diagonal (which holds
/// `R[k][k]`), so it travels alongside β rather than being read from `packed`.
#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
pub(super) struct QReflector {
    /// Householder vector head component `v_k[k]`.
    pub(super) head: f32,
    /// Householder scale factor `β_k = 2 / (v_kᵀ v_k)`.
    pub(super) beta: f32,
}

// ---------------------------------------------------------------------------
// Q accumulation kernel:  Q ← H₁ (H₂ (⋯ (H_k · I)))
// ---------------------------------------------------------------------------

/// WGSL source accumulating **Q** by applying the stored reflectors to an
/// identity, one workgroup per column of **Q**.
pub(super) fn q_accumulate_shader_source() -> String {
    r#"struct QMeta {
    m: u32,
    n: u32,
    reflector_count: u32,
}
struct QReflector {
    head: f32,
    beta: f32,
}

@group(0) @binding(0) var<storage, read>       packed: array<f32>;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read>       reflectors: array<QReflector>;
@group(0) @binding(3) var<uniform>             params: QMeta;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let col = wid.x;
    if (col >= params.m) {
        return;
    }
    let tid = lid.x;
    let m = params.m;
    let n = params.n;

    // Reverse order: Q = H_1 (H_2 (... (H_k I))), so the last reflector
    // reaches the identity first.
    for (var idx = 0u; idx < params.reflector_count; idx = idx + 1u) {
        let k = params.reflector_count - 1u - idx;
        let head = reflectors[k].head;
        let beta = reflectors[k].beta;
        let span = m - k;

        // Phase 1: dot = v_kᵀ · Q[k:m, col], reduced as a 256-way tree.
        var partial = f32(0.0);
        var row = tid;
        while (row < span) {
            let r = k + row;
            // The packed diagonal slot holds R[k][k]; the reflector head
            // arrives separately.
            var v = packed[r * n + k];
            if (row == 0u) { v = head; }
            partial = partial + v * q[r * m + col];
            row = row + 256u;
        }
        sdata[tid] = partial;
        workgroupBarrier();
        for (var s = 128u; s > 0u; s = s >> 1u) {
            if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
            workgroupBarrier();
        }
        let scaled = beta * sdata[0];
        workgroupBarrier();

        // Phase 2: Q[k:m, col] -= β_k · v_k · dot.
        //
        // A zero β is deliberately *not* skipped with `continue`: β is a
        // storage read, so branching on it around the `workgroupBarrier()`
        // calls above would be non-uniform control flow, which WGSL forbids.
        // With β = 0 the update below is already the identity, so the
        // reflector costs a pass and changes nothing.
        row = tid;
        while (row < span) {
            let r = k + row;
            var v = packed[r * n + k];
            if (row == 0u) { v = head; }
            q[r * m + col] = q[r * m + col] - scaled * v;
            row = row + 256u;
        }
        storageBarrier();
        workgroupBarrier();
    }
}
"#
    .to_string()
}

pub(super) struct QAccumulateKernel;
