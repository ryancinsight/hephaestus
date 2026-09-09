use super::super::kron::kron_shader_source;
use super::{batched_matmul_shader_source, matmul_shader_source};
use crate::CudaDevice;
use crate::infrastructure::compiler::compile_cuda_to_ptx;
use hephaestus_core::{CudaC, DialectScalar, HephaestusError};

fn product_instructions<T: DialectScalar<CudaC>>(device: &CudaDevice, suffix: &str) {
    for (source, operations) in [
        (matmul_shader_source::<T>(), ["add.", "mul."].as_slice()),
        (
            batched_matmul_shader_source::<T>(),
            ["add.", "mul."].as_slice(),
        ),
        (kron_shader_source::<T>(), ["mul."].as_slice()),
    ] {
        let ptx = compile_cuda_to_ptx(&source, device)
            .expect("invariant: supported native product source compiles");
        // CUDA SM80 implements scalar bfloat addition/multiplication using
        // native bfloat FMA with exact identity operands; SM90 has direct ops.
        for operation in operations {
            assert!(
                ptx.lines().any(|line| line.contains(suffix)
                    && (line.contains(operation) || line.contains("fma."))),
                "native {operation} missing from product PTX: {ptx}"
            );
        }
        assert!(
            !ptx.lines().any(|line| line.contains(".f32")
                && ["add.", "mul.", "fma.", "mad."]
                    .iter()
                    .any(|operation| line.contains(operation))),
            "widened arithmetic in product PTX: {ptx}"
        );
    }
}

#[test]
fn product_ptx_preserves_native_arithmetic() {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(
            tracing_subscriber::fmt::format::FmtSpan::ENTER
                | tracing_subscriber::fmt::format::FmtSpan::CLOSE,
        )
        .finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(HephaestusError::AdapterUnavailable { message })
            if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() =>
        {
            tracing::info!(%message, "CUDA adapter absent");
            return;
        }
        Err(error) => panic!("CUDA product PTX requires an available device: {error}"),
    };
    let _span =
        tracing::info_span!("product_ptx", capability = device.compute_capability()).entered();
    product_instructions::<eunomia::F16>(&device, ".f16");
    product_instructions::<eunomia::Bf16>(&device, ".bf16");
}
