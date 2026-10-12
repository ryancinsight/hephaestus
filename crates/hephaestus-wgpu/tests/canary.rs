//! Canary integration: backend verdicts, live precision grades, and seam
//! gates, all measured against the attached adapter.
//!
//! Every assertion branches on the adapter's backend: on Vulkan, f64
//! transcendentals lower to GLSL.std.450 instructions outside their operand
//! contract, so verdicts and gates must refuse them; other backends allow
//! them and report their own live grades. Nothing here may abort the process
//! on any driver: blocked pairs are refused before any shader is built.

use hephaestus_core::{
    BlockWidth, ComputeDevice, CrossEntropyForwardOperands, CrossEntropyOps, DeviceFeature,
    DevicePreference, ElementwiseOps, HephaestusError, StridedView,
};
use hephaestus_wgpu::{
    BuiltinVerdict, CanaryWidth, ExpOp, MathBuiltin, PowOp, PrecisionGrade, SinOp, SoftplusOp,
    TanhOp, WgpuCrossEntropyOps, WgpuDevice, WgpuElementwiseOps, binary_elementwise_into,
    builtin_verdict, probe_precision, require_builtin, scan_builtins, unary_elementwise_into,
    unary_elementwise_strided_into,
};
use leto::Layout;

/// Builtins with a live precision probe.
const PROBED: [MathBuiltin; 4] = [
    MathBuiltin::Exp,
    MathBuiltin::Log,
    MathBuiltin::Pow,
    MathBuiltin::Tanh,
];

/// Text every gate refusal carries.
const BLOCKED: &str = "blocked on this backend";

/// Assert a typed backend refusal whose message names the blocked rule.
#[track_caller]
fn assert_blocked(error: &HephaestusError) {
    assert!(
        matches!(error, HephaestusError::Unsupported { message } if message.contains(BLOCKED)),
        "expected a typed blocked refusal, got: {error:?}"
    );
}

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

fn is_vulkan(device: &WgpuDevice) -> bool {
    device
        .adapter_info()
        .is_some_and(|info| info.backend == wgpu::Backend::Vulkan)
}

fn supports_f64(device: &WgpuDevice) -> bool {
    device.supports_device_feature(DeviceFeature::ShaderF64)
}

#[test]
fn vulkan_blocks_every_probed_transcendental_at_double_width() {
    let Some(device) = device() else { return };
    for builtin in PROBED {
        let verdict = builtin_verdict(&device, builtin, CanaryWidth::F64);
        if is_vulkan(&device) {
            assert!(
                matches!(verdict, BuiltinVerdict::Blocked { cause } if cause.contains("GLSL.std.450")),
                "{builtin:?} at F64 must be blocked on Vulkan, got {verdict:?}"
            );
        } else {
            assert_eq!(verdict, BuiltinVerdict::Supported);
        }
        // GLSL.std.450 admits 32-bit operands, so f32 is supported everywhere.
        assert_eq!(
            builtin_verdict(&device, builtin, CanaryWidth::F32),
            BuiltinVerdict::Supported
        );
    }
}

#[test]
fn live_grades_measure_truth_without_crashing() {
    let Some(device) = device() else { return };
    // f32 transcendentals are exact on every adapter measured so far; a Poor
    // grade anywhere is a new driver finding, not a passing result.
    for builtin in PROBED {
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
    if is_vulkan(&device) {
        // Blocked pairs refuse before compiling: no shader is built, so these
        // assertions cannot abort or read undefined results on any driver.
        for builtin in PROBED {
            let error = probe_precision(&device, builtin, CanaryWidth::F64)
                .expect_err("canary must refuse undefined f64 lowerings on Vulkan");
            assert!(
                matches!(&error, HephaestusError::Unsupported { message } if message.contains("refuses to compile")),
                "unexpected refusal: {error:?}"
            );
        }
        return;
    }
    if !supports_f64(&device) {
        eprintln!("canary skip: adapter lacks ShaderF64");
        return;
    }
    let report = probe_precision(&device, MathBuiltin::Exp, CanaryWidth::F64)
        .expect("f64 exp dispatches off Vulkan");
    assert!(
        report.grade() != PrecisionGrade::Poor,
        "f64 exp worst relative error is {:e}",
        report.max_relative_error
    );
}

#[test]
fn gates_refuse_undefined_lowerings_with_cause() {
    let Some(device) = device() else { return };
    if !supports_f64(&device) {
        eprintln!("canary skip: adapter lacks ShaderF64");
        return;
    }
    if !is_vulkan(&device) {
        // Other backends allow the builtins; nothing to refuse.
        for builtin in PROBED {
            require_builtin(&device, builtin, CanaryWidth::F64, "canary probe")
                .expect("non-Vulkan backends allow f64 builtins");
        }
        assert!(scan_builtins("tanh(x)").contains(&MathBuiltin::Tanh));
        return;
    }

    let input = device.upload(&[0.5_f64; 4]).expect("input");
    let output = device.alloc_uninitialized::<f64>(4).expect("output");
    let error = unary_elementwise_into::<ExpOp, f64>(&device, &input, &output, BlockWidth::DEFAULT)
        .expect_err("f64 exp must refuse on Vulkan");
    assert_blocked(&error);

    let error =
        unary_elementwise_into::<TanhOp, f64>(&device, &input, &output, BlockWidth::DEFAULT)
            .expect_err("f64 tanh must refuse on Vulkan");
    assert_blocked(&error);

    let error =
        unary_elementwise_into::<SoftplusOp, f64>(&device, &input, &output, BlockWidth::DEFAULT)
            .expect_err("f64 softplus (exp/log-bearing) must refuse on Vulkan");
    assert_blocked(&error);

    // Restricted builtins without a probe refuse under the same rule.
    let error = unary_elementwise_into::<SinOp, f64>(&device, &input, &output, BlockWidth::DEFAULT)
        .expect_err("f64 sin must refuse on Vulkan");
    assert_blocked(&error);
    assert!(error.to_string().contains("`sin`"), "got: {error}");

    // Strided and prepared seams compile the same templates and refuse alike.
    let layout = Layout::c_contiguous([4]).expect("vector");
    let error = unary_elementwise_strided_into::<ExpOp, f64, 1>(
        &device,
        StridedView::new(&input, &layout),
        StridedView::new(&output, &layout),
        BlockWidth::DEFAULT,
    )
    .expect_err("strided f64 exp must refuse on Vulkan");
    assert_blocked(&error);
    let Err(error) = WgpuElementwiseOps.prepare_unary_into::<TanhOp, 1>(
        &device,
        StridedView::new(&input, &layout),
        StridedView::new(&output, &layout),
    ) else {
        panic!("prepared f64 tanh must refuse on Vulkan");
    };
    assert_blocked(&error);

    let error = binary_elementwise_into::<PowOp, f64>(
        &device,
        &input,
        &input,
        &output,
        BlockWidth::DEFAULT,
    )
    .expect_err("f64 pow must refuse on Vulkan");
    assert_blocked(&error);

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
        .expect_err("f64 CE forward (exp/log-bearing) must refuse on Vulkan");
    assert_blocked(&error);
}
