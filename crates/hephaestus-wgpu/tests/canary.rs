//! Driver canary integration: recorded verdicts, live precision grades, and
//! crash-class gates, all measured against the attached adapter.
//!
//! Every assertion branches on the adapter identity: the recorded NVIDIA
//! Vulkan defects pin their verdicts and gate refusals, while unknown adapters
//! default-allow and report their own live grades. Nothing here may abort the
//! process on any driver: crash-class pairs are refused before any shader is
//! built.

use hephaestus_core::{
    BlockWidth, ComputeDevice, CrossEntropyForwardOperands, CrossEntropyOps, DeviceFeature,
    DevicePreference, StridedView,
};
use hephaestus_wgpu::{
    BuiltinVerdict, CanaryWidth, MathBuiltin, PowOp, PrecisionGrade, SoftplusOp, TanhOp,
    WgpuCrossEntropyOps, WgpuDevice, binary_elementwise_into, builtin_verdict, probe_precision,
    require_builtin, scan_builtins, unary_elementwise_into,
};
use leto::Layout;

const RECORDED_VENDOR: u32 = 0x10DE;
const RECORDED_DEVICE: u32 = 0x2C02;

fn device() -> Option<WgpuDevice> {
    let mut limits = WgpuDevice::downlevel_device_limits();
    limits.max_storage_buffers_per_shader_stage = Some(5);
    limits.max_buffers_and_acceleration_structures_per_shader_stage = Some(6);
    match WgpuDevice::try_with_device_preference_and_optional_device_features_and_limits(
        "hephaestus-canary",
        DevicePreference::HighPerformance,
        &[DeviceFeature::ShaderF64],
        limits,
    ) {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("canary skip: {error}");
            None
        }
    }
}

fn is_recorded_adapter(device: &WgpuDevice) -> bool {
    device.adapter_info().is_some_and(|info| {
        info.backend == wgpu::Backend::Vulkan
            && info.vendor == RECORDED_VENDOR
            && info.device == RECORDED_DEVICE
    })
}

fn supports_f64(device: &WgpuDevice) -> bool {
    device.supports_device_feature(DeviceFeature::ShaderF64)
}

#[test]
fn recorded_table_blocks_only_its_adapter() {
    let Some(device) = device() else { return };
    for (builtin, width) in [
        (MathBuiltin::Log, CanaryWidth::F64),
        (MathBuiltin::Tanh, CanaryWidth::F64),
        (MathBuiltin::Pow, CanaryWidth::F64),
    ] {
        let verdict = builtin_verdict(&device, builtin, width);
        if is_recorded_adapter(&device) {
            assert!(
                matches!(verdict, BuiltinVerdict::Blocked { .. }),
                "{builtin:?} at {width:?} must be blocked on the recorded adapter, got {verdict:?}"
            );
        } else {
            assert_eq!(verdict, BuiltinVerdict::Supported);
        }
    }
    // f32 was never defective anywhere recorded; the degraded f64 exp entry
    // is asserted against the live probe below.
    assert_eq!(
        builtin_verdict(&device, MathBuiltin::Exp, CanaryWidth::F32),
        BuiltinVerdict::Supported
    );
    assert_eq!(
        builtin_verdict(&device, MathBuiltin::Log, CanaryWidth::F32),
        BuiltinVerdict::Supported
    );
}

#[test]
fn live_grades_measure_truth_without_crashing() {
    let Some(device) = device() else { return };
    // f32 transcendentals are exact on every adapter measured so far; a Poor
    // grade anywhere is a new driver finding, not a passing result.
    for builtin in [
        MathBuiltin::Exp,
        MathBuiltin::Log,
        MathBuiltin::Pow,
        MathBuiltin::Tanh,
    ] {
        let report = probe_precision(&device, builtin, CanaryWidth::F32)
            .expect("f32 probes dispatch on any adapter");
        assert_eq!(report.samples, 8);
        assert_eq!(
            report.grade(),
            PrecisionGrade::Exact,
            "{builtin:?} f32 worst relative error is {:e}",
            report.max_relative_error
        );
    }
    if !supports_f64(&device) {
        eprintln!("canary skip: adapter lacks ShaderF64");
        return;
    }
    let report = probe_precision(&device, MathBuiltin::Exp, CanaryWidth::F64)
        .expect("f64 exp dispatches (degraded, never blocked)");
    assert!(
        report.grade() != PrecisionGrade::Poor,
        "f64 exp worst relative error is {:e}",
        report.max_relative_error
    );
    if is_recorded_adapter(&device) {
        // The recorded entry says 6e-8; the live probe must agree within an
        // order of magnitude, or the table entry has rotted.
        assert_eq!(report.grade(), PrecisionGrade::SingleGrade);
        assert!(
            (1e-9..1e-5).contains(&report.max_relative_error),
            "recorded f64 exp grade drifted: {:e}",
            report.max_relative_error
        );
        // Crash-class pairs refuse before compiling: no shader is built, so
        // these assertions cannot abort even on the defective driver.
        for builtin in [MathBuiltin::Log, MathBuiltin::Pow, MathBuiltin::Tanh] {
            let error = probe_precision(&device, builtin, CanaryWidth::F64)
                .expect_err("canary must refuse crash-class compiles");
            assert!(
                error.to_string().contains("refuses to compile"),
                "unexpected refusal: {error}"
            );
        }
    }
}

#[test]
fn gates_refuse_crash_class_dispatch_with_cause() {
    let Some(device) = device() else { return };
    if !supports_f64(&device) {
        eprintln!("canary skip: adapter lacks ShaderF64");
        return;
    }
    if !is_recorded_adapter(&device) {
        // Unknown adapters default-allow; nothing to refuse.
        for builtin in [MathBuiltin::Log, MathBuiltin::Pow, MathBuiltin::Tanh] {
            require_builtin(&device, builtin, CanaryWidth::F64, "canary probe")
                .expect("unknown adapters default-allow");
        }
        assert!(scan_builtins("tanh(x)").contains(&MathBuiltin::Tanh));
        return;
    }

    let input = device.upload(&[0.5_f64; 4]).expect("input");
    let output = device.alloc_uninitialized::<f64>(4).expect("output");
    let error =
        unary_elementwise_into::<TanhOp, f64>(&device, &input, &output, BlockWidth::DEFAULT)
            .expect_err("f64 tanh must refuse on the recorded adapter");
    assert!(error.to_string().contains("driver-blocked"), "got: {error}");

    let error =
        unary_elementwise_into::<SoftplusOp, f64>(&device, &input, &output, BlockWidth::DEFAULT)
            .expect_err("f64 softplus (log-bearing) must refuse on the recorded adapter");
    assert!(error.to_string().contains("driver-blocked"), "got: {error}");

    let error = binary_elementwise_into::<PowOp, f64>(
        &device,
        &input,
        &input,
        &output,
        BlockWidth::DEFAULT,
    )
    .expect_err("f64 pow must refuse on the recorded adapter");
    assert!(error.to_string().contains("driver-blocked"), "got: {error}");

    let matrix = Layout::c_contiguous([1, 2]).expect("matrix");
    let vector = Layout::c_contiguous([1]).expect("vector");
    let logits = device.upload(&[0.0_f64, 0.0]).expect("logits");
    let targets = device.upload(&[0_u32]).expect("targets");
    let loss = device.upload(&[0.0_f64]).expect("loss");
    let probabilities = device.alloc_zeroed::<f64>(2).expect("probabilities");
    let error = WgpuCrossEntropyOps
        .cross_entropy_forward_into(
            &device,
            CrossEntropyForwardOperands {
                logits: StridedView::new(&logits, &matrix),
                targets: StridedView::new(&targets, &vector),
                loss: StridedView::new(&loss, &vector),
                probabilities: StridedView::new(&probabilities, &matrix),
            },
        )
        .expect_err("f64 CE forward (log-bearing) must refuse on the recorded adapter");
    assert!(error.to_string().contains("driver-blocked"), "got: {error}");
}
