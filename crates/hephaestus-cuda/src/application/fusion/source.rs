//! CUDA C source and ABI metadata for runtime-rank fusion.

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    CudaC, DialectScalar, FusedReduction, HephaestusError, IdentityToken, MaxOp, MinOp, ProdOp,
    Result, SumOp,
};
use leto::LayoutDyn;

/// Maximum runtime rank represented by the provider ABI.
pub(crate) const MAX_FUSION_RANK: usize = 8;
/// Threads per one-dimensional fusion launch block.
pub(crate) const FUSION_WORKGROUP_WIDTH: u32 = 256;

/// Layout record shared by the CUDA elementwise and reduction kernels.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct FusionLayoutInfo {
    pub(crate) offset: u64,
    pub(crate) rank: u32,
    pub(crate) length: u32,
    pub(crate) axis: u32,
    pub(crate) extent: u32,
    pub(crate) shape: [u32; MAX_FUSION_RANK],
    pub(crate) strides: [i64; MAX_FUSION_RANK],
}

const _: () = assert!(core::mem::size_of::<FusionLayoutInfo>() == 120);
const _: () = assert!(core::mem::align_of::<FusionLayoutInfo>() == 8);

impl FusionLayoutInfo {
    pub(crate) fn from_layout(layout: &LayoutDyn, axis: Option<(usize, usize)>) -> Result<Self> {
        if layout.shape.len() != layout.strides.len() {
            return Err(HephaestusError::DispatchFailed {
                message: "CUDA fusion layout shape and stride ranks differ".to_string(),
            });
        }
        if layout.ndim() > MAX_FUSION_RANK {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "CUDA fusion supports rank <= {MAX_FUSION_RANK}, got {}",
                    layout.ndim()
                ),
            });
        }
        let (_, maximum) =
            layout
                .checked_min_max_offsets()
                .map_err(|error| HephaestusError::DispatchFailed {
                    message: format!("CUDA fusion layout rejected: {error}"),
                })?;
        let _maximum = i64::try_from(maximum).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("CUDA fusion address {maximum} exceeds signed device range"),
        })?;
        let mut shape = [1u32; MAX_FUSION_RANK];
        let mut strides = [0i64; MAX_FUSION_RANK];
        for (index, (&dimension, &stride)) in layout.shape.iter().zip(&layout.strides).enumerate() {
            shape[index] =
                u32::try_from(dimension).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("CUDA fusion dimension {dimension} exceeds u32 range"),
                })?;
            strides[index] =
                i64::try_from(stride).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("CUDA fusion stride {stride} exceeds signed device range"),
                })?;
        }
        let length = u32::try_from(layout.checked_size().map_err(|error| {
            HephaestusError::DispatchFailed {
                message: format!("CUDA fusion layout rejected: {error}"),
            }
        })?)
        .map_err(|_| HephaestusError::DispatchFailed {
            message: "CUDA fusion logical length exceeds u32 range".to_string(),
        })?;
        let offset = u64::try_from(layout.offset).map_err(|_| HephaestusError::DispatchFailed {
            message: "CUDA fusion offset exceeds u64 range".to_string(),
        })?;
        let (axis, extent) = axis.map_or((0, 1), |(axis, extent)| (axis, extent));
        Ok(Self {
            offset,
            rank: u32::try_from(layout.ndim()).map_err(|_| HephaestusError::DispatchFailed {
                message: "CUDA fusion rank exceeds u32 range".to_string(),
            })?,
            length,
            axis: u32::try_from(axis).map_err(|_| HephaestusError::DispatchFailed {
                message: "CUDA fusion axis exceeds u32 range".to_string(),
            })?,
            extent: u32::try_from(extent).map_err(|_| HephaestusError::DispatchFailed {
                message: "CUDA fusion axis extent exceeds u32 range".to_string(),
            })?,
            shape,
            strides,
        })
    }
}

/// Scalar capabilities needed by provider-owned CUDA fusion.
pub trait CudaFusionScalar: DialectScalar<CudaC> + Pod + Send + Sync + 'static {
    /// CUDA header declarations required by this scalar's device type.
    const PRELUDE: &'static str;
    /// CUDA literal for the additive identity.
    const ZERO: &'static str;
    /// CUDA literal for the multiplicative identity.
    const ONE: &'static str;
    /// CUDA literal below every supported value for maximum reduction.
    const LOWEST: &'static str;
    /// CUDA literal above every supported value for minimum reduction.
    const HIGHEST: &'static str;
    /// CUDA expression converting the runtime axis length for mean reduction.
    const MEAN_DIVISOR: &'static str;
}

macro_rules! impl_fusion_scalar {
    ($ty:ty, $divisor:literal) => {
        impl CudaFusionScalar for $ty {
            const PRELUDE: &'static str = "";
            const ZERO: &'static str = <Self as IdentityToken<SumOp, CudaC>>::TOKEN;
            const ONE: &'static str = <Self as IdentityToken<ProdOp, CudaC>>::TOKEN;
            const LOWEST: &'static str = <Self as IdentityToken<MaxOp, CudaC>>::TOKEN;
            const HIGHEST: &'static str = <Self as IdentityToken<MinOp, CudaC>>::TOKEN;
            const MEAN_DIVISOR: &'static str = $divisor;
        }
    };
}

impl_fusion_scalar!(f32, "static_cast<float>(axis_length)");
impl_fusion_scalar!(i32, "static_cast<int>(axis_length)");
impl_fusion_scalar!(u32, "static_cast<unsigned int>(axis_length)");

impl CudaFusionScalar for f64 {
    const PRELUDE: &'static str = "";
    const ZERO: &'static str = "0.0";
    const ONE: &'static str = "1.0";
    const LOWEST: &'static str = "-1.7976931348623157e+308";
    const HIGHEST: &'static str = "1.7976931348623157e+308";
    const MEAN_DIVISOR: &'static str = "static_cast<double>(axis_length)";
}

impl CudaFusionScalar for eunomia::F16 {
    const PRELUDE: &'static str = "#include <cuda_fp16.h>\n";
    const ZERO: &'static str = <Self as IdentityToken<SumOp, CudaC>>::TOKEN;
    const ONE: &'static str = <Self as IdentityToken<ProdOp, CudaC>>::TOKEN;
    const LOWEST: &'static str = <Self as IdentityToken<MaxOp, CudaC>>::TOKEN;
    const HIGHEST: &'static str = <Self as IdentityToken<MinOp, CudaC>>::TOKEN;
    const MEAN_DIVISOR: &'static str = "__float2half(static_cast<float>(axis_length))";
}

impl CudaFusionScalar for eunomia::Bf16 {
    const PRELUDE: &'static str = "#include <cuda_bf16.h>\n";
    const ZERO: &'static str = <Self as IdentityToken<SumOp, CudaC>>::TOKEN;
    const ONE: &'static str = <Self as IdentityToken<ProdOp, CudaC>>::TOKEN;
    const LOWEST: &'static str = <Self as IdentityToken<MaxOp, CudaC>>::TOKEN;
    const HIGHEST: &'static str = <Self as IdentityToken<MinOp, CudaC>>::TOKEN;
    const MEAN_DIVISOR: &'static str = "__float2bfloat16(static_cast<float>(axis_length))";
}

pub(crate) fn validate_expression_source(source: &str) -> Result<()> {
    if source.trim().is_empty() {
        return Err(HephaestusError::InvalidConfiguration {
            message: "CUDA fusion expression must not be empty".to_string(),
        });
    }
    if source
        .chars()
        .any(|character| matches!(character, '\0' | '\n' | '\r' | ';' | '{' | '}' | '#'))
        || source.contains("//")
        || source.contains("/*")
        || source.contains("*/")
    {
        return Err(HephaestusError::InvalidConfiguration {
            message: "CUDA fusion expression must be one C++ expression fragment".to_string(),
        });
    }
    Ok(())
}

fn cuda_layout() -> &'static str {
    "struct FusionLayoutInfo {
    unsigned long long offset;
    unsigned int rank;
    unsigned int length;
    unsigned int axis;
    unsigned int extent;
    unsigned int shape[8];
    long long strides[8];
};
"
}

fn input_declarations<T: DialectScalar<CudaC>>(input_count: usize) -> String {
    (0..input_count)
        .map(|index| {
            format!(
                "    const {ty}* storage_input_{index},\n",
                ty = T::TYPE_TOKEN
            )
        })
        .collect()
}

fn input_loads<T: DialectScalar<CudaC>>(input_count: usize) -> String {
    (0..input_count)
        .map(|index| {
            format!(
                "    long long input_offset_{index} = static_cast<long long>(layouts[{index}].offset);\n\
    for (unsigned int input_axis_{index} = 0; input_axis_{index} < output_layout.rank; ++input_axis_{index}) {{\n\
        if (layouts[{index}].shape[input_axis_{index}] > 1u) {{\n\
            input_offset_{index} += static_cast<long long>(coordinates[input_axis_{index}]) *\
 layouts[{index}].strides[input_axis_{index}];\n\
        }}\n\
    }}\n\
    const {ty} input_{index} = storage_input_{index}[input_offset_{index}];\n",
                ty = T::TYPE_TOKEN,
            )
        })
        .collect()
}

fn output_offset() -> &'static str {
    "    long long output_offset = static_cast<long long>(output_layout.offset);\n\
    for (unsigned int output_axis = 0; output_axis < output_layout.rank; ++output_axis) {\n\
        output_offset += static_cast<long long>(coordinates[output_axis]) *\
 output_layout.strides[output_axis];\n\
    }\n"
}

/// Generate the provider-owned CUDA wrapper for an elementwise expression.
pub(crate) fn elementwise_source<T: CudaFusionScalar>(
    input_count: usize,
    expression: &str,
) -> String {
    let output_binding = input_count;
    format!(
        "{prelude}#define FUSION_WORKGROUP_WIDTH {width}\n{layout}\nextern \"C\" __global__ __launch_bounds__(FUSION_WORKGROUP_WIDTH) void fused_elementwise_kernel(\n{inputs}    {ty}* output,\n    const FusionLayoutInfo* layouts\n) {{\n    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n    const FusionLayoutInfo output_layout = layouts[{output_binding}];\n    if (index >= output_layout.length) {{\n        return;\n    }}\n    unsigned int coordinates[8] = {{}};\n    unsigned int remaining = index;\n    for (int coordinate_axis = static_cast<int>(output_layout.rank) - 1; coordinate_axis >= 0; --coordinate_axis) {{\n        const unsigned int dimension = output_layout.shape[coordinate_axis];\n        coordinates[coordinate_axis] = remaining % dimension;\n        remaining /= dimension;\n    }}\n{loads}    {output_offset}    output[output_offset] = {expression};\n}}\n",
        prelude = T::PRELUDE,
        layout = cuda_layout(),
        width = FUSION_WORKGROUP_WIDTH,
        inputs = input_declarations::<T>(input_count),
        output_binding = output_binding,
        ty = T::TYPE_TOKEN,
        loads = input_loads::<T>(input_count),
        output_offset = output_offset(),
        expression = expression,
    )
}

/// Generate the provider-owned CUDA wrapper for an expression followed by an
/// axis reduction.
pub(crate) fn reduction_source<T: CudaFusionScalar>(
    input_count: usize,
    expression: &str,
    reduction: FusedReduction,
) -> Result<String> {
    let output_binding = input_count;
    let (identity, combine, final_value) = match reduction {
        FusedReduction::Sum => (T::ZERO, "lhs + rhs", "acc".to_string()),
        FusedReduction::Product => (T::ONE, "lhs * rhs", "acc".to_string()),
        FusedReduction::Mean => (T::ZERO, "lhs + rhs", format!("acc / {}", T::MEAN_DIVISOR)),
        FusedReduction::Maximum => (T::LOWEST, "max(lhs, rhs)", "acc".to_string()),
        FusedReduction::Minimum => (T::HIGHEST, "min(lhs, rhs)", "acc".to_string()),
        _ => {
            return Err(HephaestusError::InvalidConfiguration {
                message: "unsupported CUDA fused reduction operation".to_string(),
            });
        }
    };
    Ok(format!(
        "{prelude}#define FUSION_WORKGROUP_WIDTH {width}\n{layout}\nextern \"C\" __global__ __launch_bounds__(FUSION_WORKGROUP_WIDTH) void fused_reduction_kernel(\n{inputs}    {ty}* output,\n    const FusionLayoutInfo* layouts\n) {{\n    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n    const FusionLayoutInfo output_layout = layouts[{output_binding}];\n    if (index >= output_layout.length) {{\n        return;\n    }}\n    unsigned int coordinates[8] = {{}};\n    unsigned int remaining = index;\n    for (int coordinate_axis = static_cast<int>(output_layout.rank) - 1; coordinate_axis >= 0; --coordinate_axis) {{\n        const unsigned int dimension = output_layout.shape[coordinate_axis];\n        coordinates[coordinate_axis] = remaining % dimension;\n        remaining /= dimension;\n    }}\n    {ty} acc = {identity};\n    const unsigned int axis_length = output_layout.extent;\n    for (unsigned int reduction_index = 0; reduction_index < axis_length; ++reduction_index) {{\n        coordinates[output_layout.axis] = reduction_index;\n{loads}        const {ty} expression_value = {expression};\n        const {ty} lhs = acc;\n        const {ty} rhs = expression_value;\n        acc = {combine};\n    }}\n    coordinates[output_layout.axis] = 0u;\n{output_offset}    output[output_offset] = {final_value};\n}}\n",
        prelude = T::PRELUDE,
        layout = cuda_layout(),
        width = FUSION_WORKGROUP_WIDTH,
        inputs = input_declarations::<T>(input_count),
        output_binding = output_binding,
        ty = T::TYPE_TOKEN,
        identity = identity,
        combine = combine,
        final_value = final_value,
        loads = input_loads::<T>(input_count),
        expression = expression,
        output_offset = output_offset(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expression_validation_rejects_non_expression_fragments() {
        for source in ["", "input_0; input_1", "input_0 { input_1 }", "#define X 1"] {
            assert!(
                validate_expression_source(source).is_err(),
                "source must be rejected: {source:?}"
            );
        }
    }

    #[test]
    fn generated_source_binds_runtime_layout_and_signed_offsets() {
        let source = elementwise_source::<f32>(2, "input_0 + input_1");
        assert!(source.contains("fused_elementwise_kernel"));
        assert!(source.contains("const float input_0"));
        assert!(source.contains("const FusionLayoutInfo* layouts"));
        assert!(source.contains("long long input_offset_0"));
        assert!(source.contains("output[output_offset] = input_0 + input_1"));
        assert!(source.contains("__launch_bounds__(FUSION_WORKGROUP_WIDTH)"));
    }

    #[test]
    fn f64_reduction_source_keeps_native_double_arithmetic() {
        let source = reduction_source::<f64>(1, "input_0", FusedReduction::Mean)
            .expect("mean source generation");
        assert!(source.contains("double acc"));
        assert!(source.contains("static_cast<double>(axis_length)"));
        assert!(!source.contains("float acc"));
    }

    #[test]
    fn reduced_precision_source_includes_the_eunomia_cuda_type_header() {
        let f16 = elementwise_source::<eunomia::F16>(1, "input_0");
        let bf16 = elementwise_source::<eunomia::Bf16>(1, "input_0");
        assert!(f16.starts_with("#include <cuda_fp16.h>"));
        assert!(bf16.starts_with("#include <cuda_bf16.h>"));
    }
}
