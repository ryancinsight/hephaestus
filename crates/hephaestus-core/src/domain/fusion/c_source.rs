//! C-dialect kernel source and ABI metadata for runtime-rank fusion.
//!
//! CUDA C and HIP C++ share one kernel language for the wrappers generated
//! here, so both backends monomorphize these functions over their dialect
//! instead of carrying a per-vendor copy. WGSL keeps its own generator in
//! `hephaestus-wgpu`: its binding model and address widths are genuinely a
//! different ABI, not a spelling of this one.

use eunomia::{Pod, Zeroable};
use leto::LayoutDyn;

use super::ops::{FusedReduction, MAX_FUSION_RANK};
use super::scalar::FusionScalar;
use crate::{DialectScalar, DynamicStridedView, HephaestusError, KernelDialect, Result};

/// Layout record shared by the C-dialect elementwise and reduction kernels.
///
/// The 64-bit offsets match the LP64 device model CUDA and HIP share; the
/// WGSL `u32`/`i32` record in `hephaestus-wgpu` is the separate ABI its
/// address space requires.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FusionLayoutInfo {
    offset: u64,
    rank: u32,
    length: u32,
    axis: u32,
    extent: u32,
    shape: [u32; MAX_FUSION_RANK],
    strides: [i64; MAX_FUSION_RANK],
}

const _: () = assert!(core::mem::size_of::<FusionLayoutInfo>() == 120);
const _: () = assert!(core::mem::align_of::<FusionLayoutInfo>() == 8);

impl FusionLayoutInfo {
    /// Pack a validated runtime-rank layout into the device record.
    ///
    /// # Errors
    ///
    /// Returns a typed provider error when the layout's shape/stride ranks
    /// differ, the rank exceeds [`MAX_FUSION_RANK`], or an address, dimension,
    /// stride, length, or offset exceeds the device record's range.
    pub fn from_layout(layout: &LayoutDyn, axis: Option<(usize, usize)>) -> Result<Self> {
        if layout.shape.len() != layout.strides.len() {
            return Err(HephaestusError::DispatchFailed {
                message: "fusion layout shape and stride ranks differ".to_string(),
            });
        }
        if layout.ndim() > MAX_FUSION_RANK {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "fusion supports rank <= {MAX_FUSION_RANK}, got {}",
                    layout.ndim()
                ),
            });
        }
        let (_, maximum) =
            layout
                .checked_min_max_offsets()
                .map_err(|error| HephaestusError::DispatchFailed {
                    message: format!("fusion layout rejected: {error}"),
                })?;
        let _maximum = i64::try_from(maximum).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("fusion address {maximum} exceeds signed device range"),
        })?;
        let mut shape = [1u32; MAX_FUSION_RANK];
        let mut strides = [0i64; MAX_FUSION_RANK];
        for (index, (&dimension, &stride)) in layout.shape.iter().zip(&layout.strides).enumerate() {
            shape[index] =
                u32::try_from(dimension).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("fusion dimension {dimension} exceeds u32 range"),
                })?;
            strides[index] =
                i64::try_from(stride).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("fusion stride {stride} exceeds signed device range"),
                })?;
        }
        let length = u32::try_from(layout.checked_size().map_err(|error| {
            HephaestusError::DispatchFailed {
                message: format!("fusion layout rejected: {error}"),
            }
        })?)
        .map_err(|_| HephaestusError::DispatchFailed {
            message: "fusion logical length exceeds u32 range".to_string(),
        })?;
        let offset = u64::try_from(layout.offset).map_err(|_| HephaestusError::DispatchFailed {
            message: "fusion offset exceeds u64 range".to_string(),
        })?;
        let (axis, extent) = axis.map_or((0, 1), |(axis, extent)| (axis, extent));
        Ok(Self {
            offset,
            rank: u32::try_from(layout.ndim()).map_err(|_| HephaestusError::DispatchFailed {
                message: "fusion rank exceeds u32 range".to_string(),
            })?,
            length,
            axis: u32::try_from(axis).map_err(|_| HephaestusError::DispatchFailed {
                message: "fusion axis exceeds u32 range".to_string(),
            })?,
            extent: u32::try_from(extent).map_err(|_| HephaestusError::DispatchFailed {
                message: "fusion axis extent exceeds u32 range".to_string(),
            })?,
            shape,
            strides,
        })
    }
}

/// Reject anything that is not one C++ expression fragment.
///
/// # Errors
///
/// Returns [`HephaestusError::InvalidConfiguration`] when the source is empty
/// or carries statement separators, braces, preprocessor directives, or
/// comments, any of which would escape the interpolated expression position.
pub fn validate_c_expression_source(source: &str) -> Result<()> {
    if source.trim().is_empty() {
        return Err(HephaestusError::InvalidConfiguration {
            message: "fusion expression must not be empty".to_string(),
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
            message: "fusion expression must be one C++ expression fragment".to_string(),
        });
    }
    Ok(())
}

/// Broadcast every input layout to the expression's runtime-rank shape.
///
/// This helper is layout-pure: it never touches buffer storage, so all
/// backends share it rather than restating broadcast unification.
///
/// # Errors
///
/// Returns [`HephaestusError::DispatchFailed`] when two inputs disagree on a
/// non-singleton extent.
pub fn fusion_expression_shape<B>(inputs: &[DynamicStridedView<'_, B>]) -> Result<Box<[usize]>> {
    let rank = inputs
        .iter()
        .map(|input| input.layout.ndim())
        .max()
        .unwrap_or(0);
    let mut shape = vec![1usize; rank];
    for input in inputs {
        let shift = rank - input.layout.ndim();
        for (axis, &dimension) in input.layout.shape.iter().enumerate() {
            let target_axis = axis + shift;
            let target = &mut shape[target_axis];
            if *target == 1 {
                *target = dimension;
            } else if dimension != 1 && *target != dimension {
                return Err(HephaestusError::DispatchFailed {
                    message: format!(
                        "fusion inputs have incompatible broadcast shapes {:?} and {:?}",
                        shape, input.layout.shape
                    ),
                });
            }
        }
    }
    Ok(shape.into_boxed_slice())
}

fn c_layout() -> &'static str {
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

fn input_declarations<L: KernelDialect, T: DialectScalar<L>>(input_count: usize) -> String {
    (0..input_count)
        .map(|index| {
            format!(
                "    const {ty}* storage_input_{index},\n",
                ty = T::TYPE_TOKEN
            )
        })
        .collect()
}

fn input_loads<L: KernelDialect, T: DialectScalar<L>>(input_count: usize) -> String {
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

/// Generate the provider-owned C-dialect wrapper for an elementwise expression.
///
/// Monomorphized per backend dialect: CUDA and HIP instantiations of one
/// scalar compile byte-identical kernels from this single generator.
#[must_use]
pub fn elementwise_source<L: KernelDialect, T: FusionScalar<L>>(
    input_count: usize,
    expression: &str,
) -> String {
    let output_binding = input_count;
    format!(
        "{prelude}#define FUSION_WORKGROUP_WIDTH {width}\n{layout}\nextern \"C\" __global__ __launch_bounds__(FUSION_WORKGROUP_WIDTH) void fused_elementwise_kernel(\n{inputs}    {ty}* output,\n    const FusionLayoutInfo* layouts\n) {{\n    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n    const FusionLayoutInfo output_layout = layouts[{output_binding}];\n    if (index >= output_layout.length) {{\n        return;\n    }}\n    unsigned int coordinates[8] = {{}};\n    unsigned int remaining = index;\n    for (int coordinate_axis = static_cast<int>(output_layout.rank) - 1; coordinate_axis >= 0; --coordinate_axis) {{\n        const unsigned int dimension = output_layout.shape[coordinate_axis];\n        coordinates[coordinate_axis] = remaining % dimension;\n        remaining /= dimension;\n    }}\n{loads}    {output_offset}    output[output_offset] = {expression};\n}}\n",
        prelude = T::PRELUDE,
        layout = c_layout(),
        width = super::ops::FUSION_WORKGROUP_WIDTH,
        inputs = input_declarations::<L, T>(input_count),
        output_binding = output_binding,
        ty = T::TYPE_TOKEN,
        loads = input_loads::<L, T>(input_count),
        output_offset = output_offset(),
        expression = expression,
    )
}

/// Generate the provider-owned C-dialect wrapper for an expression followed by
/// an axis reduction.
///
/// Infallible: [`FusedReduction`] is a closed vocabulary and this crate owns
/// it, so every variant maps to a combine spelling.
#[must_use]
pub fn reduction_source<L: KernelDialect, T: FusionScalar<L>>(
    input_count: usize,
    expression: &str,
    reduction: FusedReduction,
) -> String {
    let output_binding = input_count;
    let (identity, combine, final_value) = match reduction {
        FusedReduction::Sum => (T::ZERO, "lhs + rhs", "acc".to_string()),
        FusedReduction::Product => (T::ONE, "lhs * rhs", "acc".to_string()),
        FusedReduction::Mean => (T::ZERO, "lhs + rhs", format!("acc / {}", T::MEAN_DIVISOR)),
        FusedReduction::Maximum => (T::LOWEST, "max(lhs, rhs)", "acc".to_string()),
        FusedReduction::Minimum => (T::HIGHEST, "min(lhs, rhs)", "acc".to_string()),
    };
    format!(
        "{prelude}#define FUSION_WORKGROUP_WIDTH {width}\n{layout}\nextern \"C\" __global__ __launch_bounds__(FUSION_WORKGROUP_WIDTH) void fused_reduction_kernel(\n{inputs}    {ty}* output,\n    const FusionLayoutInfo* layouts\n) {{\n    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n    const FusionLayoutInfo output_layout = layouts[{output_binding}];\n    if (index >= output_layout.length) {{\n        return;\n    }}\n    unsigned int coordinates[8] = {{}};\n    unsigned int remaining = index;\n    for (int coordinate_axis = static_cast<int>(output_layout.rank) - 1; coordinate_axis >= 0; --coordinate_axis) {{\n        const unsigned int dimension = output_layout.shape[coordinate_axis];\n        coordinates[coordinate_axis] = remaining % dimension;\n        remaining /= dimension;\n    }}\n    {ty} acc = {identity};\n    const unsigned int axis_length = output_layout.extent;\n    for (unsigned int reduction_index = 0; reduction_index < axis_length; ++reduction_index) {{\n        coordinates[output_layout.axis] = reduction_index;\n{loads}        const {ty} expression_value = {expression};\n        const {ty} lhs = acc;\n        const {ty} rhs = expression_value;\n        acc = {combine};\n    }}\n    coordinates[output_layout.axis] = 0u;\n{output_offset}    output[output_offset] = {final_value};\n}}\n",
        prelude = T::PRELUDE,
        layout = c_layout(),
        width = super::ops::FUSION_WORKGROUP_WIDTH,
        inputs = input_declarations::<L, T>(input_count),
        output_binding = output_binding,
        ty = T::TYPE_TOKEN,
        identity = identity,
        combine = combine,
        final_value = final_value,
        loads = input_loads::<L, T>(input_count),
        expression = expression,
        output_offset = output_offset(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CudaC, HipC};

    fn assert_hip_matches_cuda<T>()
    where
        T: FusionScalar<CudaC> + FusionScalar<HipC>,
    {
        assert_eq!(
            elementwise_source::<HipC, T>(2, "input_0 + input_1"),
            elementwise_source::<CudaC, T>(2, "input_0 + input_1"),
            "HIP and CUDA elementwise kernels must share one source"
        );
        for reduction in [
            FusedReduction::Sum,
            FusedReduction::Product,
            FusedReduction::Mean,
            FusedReduction::Maximum,
            FusedReduction::Minimum,
        ] {
            assert_eq!(
                reduction_source::<HipC, T>(1, "input_0", reduction),
                reduction_source::<CudaC, T>(1, "input_0", reduction),
                "HIP and CUDA {reduction:?} kernels must share one source"
            );
        }
    }

    #[test]
    fn hip_and_cuda_generate_byte_identical_kernels() {
        assert_hip_matches_cuda::<f32>();
        assert_hip_matches_cuda::<i32>();
        assert_hip_matches_cuda::<u32>();
        assert_hip_matches_cuda::<f64>();
    }

    #[test]
    fn expression_validation_rejects_non_expression_fragments() {
        for source in ["", "input_0; input_1", "input_0 { input_1 }", "#define X 1"] {
            assert!(
                validate_c_expression_source(source).is_err(),
                "source must be rejected: {source:?}"
            );
        }
    }

    #[test]
    fn generated_source_binds_runtime_layout_and_signed_offsets() {
        let source = elementwise_source::<CudaC, f32>(2, "input_0 + input_1");
        assert!(source.contains("fused_elementwise_kernel"));
        assert!(source.contains("const float input_0"));
        assert!(source.contains("const FusionLayoutInfo* layouts"));
        assert!(source.contains("long long input_offset_0"));
        assert!(source.contains("output[output_offset] = input_0 + input_1"));
        assert!(source.contains("__launch_bounds__(FUSION_WORKGROUP_WIDTH)"));
    }

    #[test]
    fn f64_reduction_source_keeps_native_double_arithmetic() {
        let source = reduction_source::<CudaC, f64>(1, "input_0", FusedReduction::Mean);
        assert!(source.contains("double acc"));
        assert!(source.contains("static_cast<double>(axis_length)"));
        assert!(!source.contains("float acc"));
    }

    #[test]
    fn reduced_precision_source_includes_the_cuda_type_header() {
        let f16 = elementwise_source::<CudaC, eunomia::F16>(1, "input_0");
        let bf16 = elementwise_source::<CudaC, eunomia::Bf16>(1, "input_0");
        assert!(f16.starts_with("#include <cuda_fp16.h>"));
        assert!(bf16.starts_with("#include <cuda_bf16.h>"));
    }

    fn dynamic_layout(shape: &[usize], strides: &[isize]) -> LayoutDyn {
        LayoutDyn::new(
            shape.to_vec().into_boxed_slice(),
            strides.to_vec().into_boxed_slice(),
            0,
        )
        .expect("fusion test layout ranks match")
    }

    #[test]
    fn expression_shape_unifies_broadcast_inputs() {
        let unit = ();
        let first = dynamic_layout(&[2, 1], &[1, 1]);
        let second = dynamic_layout(&[1, 3], &[3, 1]);
        let shape = fusion_expression_shape(&[
            DynamicStridedView::new(&unit, &first),
            DynamicStridedView::new(&unit, &second),
        ])
        .expect("broadcastable inputs unify");
        assert_eq!(&*shape, &[2, 3]);
    }

    #[test]
    fn expression_shape_rejects_conflicting_extents() {
        let unit = ();
        let first = dynamic_layout(&[2], &[1]);
        let second = dynamic_layout(&[3], &[1]);
        assert!(
            fusion_expression_shape(&[
                DynamicStridedView::new(&unit, &first),
                DynamicStridedView::new(&unit, &second),
            ])
            .is_err(),
            "conflicting extents must not unify"
        );
    }
}
