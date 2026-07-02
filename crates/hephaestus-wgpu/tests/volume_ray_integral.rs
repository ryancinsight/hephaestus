//! Differential tests for the volume ray-integral kernel.
//!
//! Oracles: a uniform field integrates to `value × chord length` (exact for any
//! step); an affine field is integrated exactly by midpoint sampling; misses
//! integrate to zero — the same analytical contracts as a CPU midpoint
//! ray-marcher, evaluated on the live GPU.

use hephaestus_core::{BlockWidth, ComputeDevice};
use hephaestus_wgpu::{ray_line_integrals, FieldGeometry, WgpuDevice, RAY_STRIDE};

fn device_or_skip() -> Option<WgpuDevice> {
    match WgpuDevice::try_default("hephaestus-volume-test") {
        Ok(device) => Some(device),
        Err(e) => {
            eprintln!("skipping volume ray-integral test: {e}");
            None
        }
    }
}

/// 9×5×5 field with 2.0 world spacing → node box [0,16]×[0,8]×[0,8].
fn geometry() -> FieldGeometry {
    FieldGeometry {
        dims: [9, 5, 5],
        origin: [0.0, 0.0, 0.0],
        spacing: [2.0, 2.0, 2.0],
    }
}

fn upload_field(device: &WgpuDevice, f: impl Fn(u32, u32, u32) -> f32) -> Vec<f32> {
    let g = geometry();
    let mut host = Vec::new();
    for ix in 0..g.dims[0] {
        for iy in 0..g.dims[1] {
            for iz in 0..g.dims[2] {
                host.push(f(ix, iy, iz));
            }
        }
    }
    host
}

fn run(device: &WgpuDevice, host_field: &[f32], rays: &[f32], step: f32) -> Vec<f32> {
    let field = device.upload(host_field).unwrap();
    let ray_buf = device.upload(rays).unwrap();
    let n = rays.len() / RAY_STRIDE;
    let out = ray_line_integrals(
        device,
        &field,
        geometry(),
        &ray_buf,
        n,
        step,
        BlockWidth::DEFAULT,
    )
    .unwrap();
    let mut got = vec![0.0f32; n];
    device.download(&out, &mut got).unwrap();
    got
}

#[test]
fn uniform_field_integrates_to_value_times_chord() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let host = upload_field(&device, |_, _, _| 0.25);
    // +x ray through the middle: chord = 16; a miss far outside in y.
    let rays = [
        -10.0, 4.0, 4.0, 1.0, 0.0, 0.0, // hit: expect 0.25 * 16 = 4.0
        -10.0, 100.0, 4.0, 1.0, 0.0, 0.0, // miss: expect 0
    ];
    let got = run(&device, &host, &rays, 0.5);
    assert!(
        (got[0] - 4.0).abs() < 1e-4,
        "uniform chord integral {} != 4.0",
        got[0]
    );
    assert_eq!(got[1], 0.0, "missing ray must integrate to 0");
}

#[test]
fn affine_field_is_integrated_exactly_by_midpoint() {
    let Some(device) = device_or_skip() else {
        return;
    };
    // field(ix) = 0.01*ix + 0.02 → along +x world x: f(x) = 0.005*x + 0.02.
    // ∫₀¹⁶ f dx = 0.005*128 + 0.02*16 = 0.96. Midpoint is exact for affine.
    let host = upload_field(&device, |ix, _, _| 0.01 * ix as f32 + 0.02);
    let rays = [-10.0, 4.0, 4.0, 1.0, 0.0, 0.0];
    let got = run(&device, &host, &rays, 1.0);
    assert!(
        (got[0] - 0.96).abs() < 1e-4,
        "affine integral {} != 0.96",
        got[0]
    );
}

#[test]
fn step_size_does_not_change_a_uniform_integral() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let host = upload_field(&device, |_, _, _| 0.25);
    let rays = [-10.0, 4.0, 4.0, 1.0, 0.0, 0.0];
    let coarse = run(&device, &host, &rays, 8.0)[0];
    let fine = run(&device, &host, &rays, 0.125)[0];
    assert!(
        (coarse - fine).abs() < 1e-4,
        "step dependence: {coarse} vs {fine}"
    );
}

#[test]
fn oblique_ray_matches_cpu_reference() {
    let Some(device) = device_or_skip() else {
        return;
    };
    // Diagonal in-plane ray against a CPU replica of the same march.
    let host = upload_field(&device, |ix, iy, iz| {
        0.01 * ix as f32 + 0.007 * iy as f32 + 0.003 * iz as f32 + 0.05
    });
    let inv = 1.0 / (2.0f32).sqrt();
    let rays = [-10.0, -6.0, 4.0, inv, inv, 0.0];
    let step = 0.5f32;
    let got = run(&device, &host, &rays, step)[0];

    // CPU reference: identical slab clip + midpoint march + trilinear.
    let g = geometry();
    let (o, d) = ([-10.0f32, -6.0, 4.0], [inv, inv, 0.0f32]);
    let bmin = g.origin;
    let bmax = [16.0f32, 8.0, 8.0];
    let (mut t_enter, mut t_exit) = (f32::NEG_INFINITY, f32::INFINITY);
    for a in 0..3 {
        let (t1, t2) = ((bmin[a] - o[a]) / d[a], (bmax[a] - o[a]) / d[a]);
        let (lo, hi) = (t1.min(t2), t1.max(t2));
        if lo.is_finite() || !d[a].abs().eq(&0.0) {
            t_enter = t_enter.max(lo);
            t_exit = t_exit.min(hi);
        }
    }
    let len = t_exit - t_enter;
    let n = ((len / step).ceil() as usize).max(1);
    let actual = len / n as f32;
    let sample = |p: [f32; 3]| -> f32 {
        let idx: Vec<f32> = (0..3)
            .map(|a| (p[a] - g.origin[a]) / g.spacing[a])
            .collect();
        let dims = [9usize, 5, 5];
        for a in 0..3 {
            if idx[a] < 0.0 || idx[a] > (dims[a] - 1) as f32 {
                return 0.0;
            }
        }
        let lo: Vec<usize> = idx.iter().map(|&c| c.floor() as usize).collect();
        let hi: Vec<usize> = (0..3).map(|a| (lo[a] + 1).min(dims[a] - 1)).collect();
        let t: Vec<f32> = (0..3).map(|a| idx[a] - idx[a].floor()).collect();
        let v = |x: usize, y: usize, z: usize| host[(x * 5 + y) * 5 + z];
        let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = mix(v(lo[0], lo[1], lo[2]), v(hi[0], lo[1], lo[2]), t[0]);
        let c10 = mix(v(lo[0], hi[1], lo[2]), v(hi[0], hi[1], lo[2]), t[0]);
        let c01 = mix(v(lo[0], lo[1], hi[2]), v(hi[0], lo[1], hi[2]), t[0]);
        let c11 = mix(v(lo[0], hi[1], hi[2]), v(hi[0], hi[1], hi[2]), t[0]);
        mix(mix(c00, c10, t[1]), mix(c01, c11, t[1]), t[2])
    };
    let mut acc = 0.0f32;
    for s in 0..n {
        let tmid = t_enter + (s as f32 + 0.5) * actual;
        acc += sample([o[0] + d[0] * tmid, o[1] + d[1] * tmid, o[2] + d[2] * tmid]);
    }
    let expected = acc * actual;
    assert!(
        (got - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
        "gpu {got} vs cpu {expected}"
    );
}
