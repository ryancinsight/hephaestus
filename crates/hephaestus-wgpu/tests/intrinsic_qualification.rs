//! Exact inputs from the six allowed-intrinsic CPU countermodels.
//!
//! This executes the same unary seam as the named sweeps. Retained successful
//! diagnostics are observations of the attached adapter, not portable proofs.
//! The workflow records the source tree, lock and host/compiler identities; the
//! test records the actual expression and driver fields, including empty fields.

use hephaestus_conformance::accuracy::ulps_from_f64;
use hephaestus_core::{
    ComputeDevice, ElementwiseOps, ErfcOp, GeluOp, MishGradOp, MishOp, SiluOp, SoftplusOp,
    StridedView, UnaryExpr, UnaryValue, Wgsl,
};
use hephaestus_wgpu::{WgpuDevice, WgpuElementwiseOps};
use leto::Layout;

fn witness_within<Op>(device: &WgpuDevice, band: &'static str, x: f32, max_ulp: f64) -> bool
where
    Op: UnaryExpr<Wgsl> + UnaryValue + 'static,
{
    let _witness = tracing::info_span!(
        "witness",
        op = core::any::type_name::<Op>(),
        band,
        input_bits = x.to_bits(),
        target_ulp = max_ulp
    )
    .entered();
    let (input, output, layout) = {
        let _arrange = tracing::info_span!("arrange").entered();
        tracing::info!(
            expression = <Op as UnaryExpr<Wgsl>>::EXPR,
            "actual WGSL unary expression template"
        );
        (
            device.upload(&[x]).expect("witness input upload"),
            device.alloc_zeroed::<f32>(1).expect("witness output alloc"),
            Layout::c_contiguous([1]).expect("one-element witness layout"),
        )
    };
    let got = {
        let _act = tracing::info_span!("act").entered();
        WgpuElementwiseOps
            .unary_into::<Op, 1>(
                device,
                StridedView::new(&input, &layout),
                StridedView::new(&output, &layout),
            )
            .expect("exact witness unary seam dispatch");
        let mut got = [0.0_f32];
        device
            .download(&output, &mut got)
            .expect("witness readback");
        got[0]
    };
    let _assert = tracing::info_span!("assert").entered();
    let reference = Op::apply(f64::from(x));
    let error_ulp = ulps_from_f64(got, reference);
    tracing::info!(
        input_bits = x.to_bits(),
        output_bits = got.to_bits(),
        reference_bits = reference.to_bits(),
        input = f64::from(x),
        output = f64::from(got),
        reference,
        error_ulp,
        target_ulp = max_ulp,
        "exact intrinsic witness result"
    );
    assert!(
        reference.is_finite() && reference.abs() >= f64::from(f32::MIN_POSITIVE),
        "witness reference must be finite and binary32-normal"
    );
    error_ulp <= max_ulp
}

#[test]
fn exact_intrinsic_witness_qualification() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_ansi(false)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let _qualification = tracing::info_span!("intrinsic_witness_qualification").entered();
    let device = {
        let _arrange = tracing::info_span!("arrange").entered();
        let Some(device) = super::device_or_skip() else {
            tracing::warn!(
                witnesses_executed = 0,
                "adapter unavailable: no qualification evidence"
            );
            return;
        };
        tracing::info!(
            adapter = ?device.adapter_info().expect("actual adapter identity"),
            enabled_features = ?device.features(),
            device_limits = ?device.device_limits(),
            rust_target_os = std::env::consts::OS,
            rust_target_arch = std::env::consts::ARCH,
            require_wgpu_device = std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_some(),
            require_metal_device = std::env::var_os("HEPHAESTUS_METAL_REQUIRE_DEVICE").is_some(),
            "observed adapter; absent native compiler/driver versions remain unknown"
        );
        device
    };
    // Existing bands and targets: these do not replace the complete sweeps.
    let within = [
        witness_within::<ErfcOp>(&device, "erfc-mid", f32::from_bits(0x40b7_df3c), 128.0),
        witness_within::<GeluOp>(&device, "gelu", f32::from_bits(0xc036_f400), 32.0),
        witness_within::<SoftplusOp>(&device, "softplus", f32::from_bits(0xc100_0000), 16.0),
        witness_within::<SiluOp>(&device, "silu", f32::from_bits(0xc0c0_0000), 16.0),
        witness_within::<MishOp>(&device, "mish", f32::from_bits(0xc020_0000), 16.0),
        witness_within::<MishGradOp>(&device, "mish-grad", f32::from_bits(0xc020_0000), 32.0),
    ];
    assert!(
        within.into_iter().all(|within| within),
        "exact intrinsic witness target failed; all six results retained above"
    );
    tracing::info!(witnesses_executed = 6, "exact witness executions complete");
}
