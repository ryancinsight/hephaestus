use super::binary::shader_source as binary_source;
use super::scalar::shader_source as scalar_source;
use super::unary::shader_source as unary_source;
use crate::CudaDevice;
use crate::application::strided::{binary_shader, scalar_shader, unary_shader};
use crate::infrastructure::compiler::compile_cuda_to_ptx;
use hephaestus_core::{AddOp, BinaryExpr, CudaC, DialectScalar, HephaestusError, MulOp, NegOp};

fn elementwise_instructions<T: DialectScalar<CudaC>>(device: &CudaDevice, suffix: &str) {
    for (source, label, alternatives) in [
        (
            binary_source::<T>(<AddOp as BinaryExpr<CudaC>>::EXPR),
            "add",
            ["add."].as_slice(),
        ),
        (
            binary_source::<T>(<MulOp as BinaryExpr<CudaC>>::EXPR),
            "mul",
            ["mul."].as_slice(),
        ),
        // Negation lowers to whichever native-width form the compiler
        // chooses; each alternative keeps the half width.
        (
            unary_source::<NegOp, T>(),
            "neg",
            ["neg.", "sub."].as_slice(),
        ),
        (scalar_source::<AddOp, T>(), "add", ["add."].as_slice()),
        (scalar_source::<MulOp, T>(), "mul", ["mul."].as_slice()),
        (
            binary_shader::<T>(<AddOp as BinaryExpr<CudaC>>::EXPR),
            "add",
            ["add."].as_slice(),
        ),
        (
            binary_shader::<T>(<MulOp as BinaryExpr<CudaC>>::EXPR),
            "mul",
            ["mul."].as_slice(),
        ),
        (
            unary_shader::<NegOp, T>(),
            "neg",
            ["neg.", "sub."].as_slice(),
        ),
        (scalar_shader::<AddOp, T>(), "add", ["add."].as_slice()),
        (scalar_shader::<MulOp, T>(), "mul", ["mul."].as_slice()),
    ] {
        let ptx = compile_cuda_to_ptx(&source, device)
            .expect("invariant: supported native elementwise source compiles");
        assert!(
            ptx.lines().any(|line| line.contains(suffix)
                && (alternatives
                    .iter()
                    .any(|operation| line.contains(operation))
                    || line.contains("fma."))),
            "native {label} missing from elementwise PTX: {ptx}"
        );
        assert!(
            !ptx.lines().any(|line| line.contains(".f32")
                && ["add.", "mul.", "fma.", "mad.", "neg.", "sub."]
                    .iter()
                    .any(|operation| line.contains(operation))),
            "widened arithmetic in elementwise PTX: {ptx}"
        );
    }
}

#[test]
fn elementwise_ptx_preserves_native_half_arithmetic() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(HephaestusError::AdapterUnavailable { message })
            if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() =>
        {
            eprintln!("skip CUDA elementwise PTX: device unavailable ({message})");
            return;
        }
        Err(error) => panic!("CUDA elementwise PTX requires an available device: {error}"),
    };
    elementwise_instructions::<eunomia::F16>(&device, ".f16");
    elementwise_instructions::<eunomia::Bf16>(&device, ".bf16");
}
