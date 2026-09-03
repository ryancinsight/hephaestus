//! WGSL source and metadata for runtime-rank fusion.

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    DialectScalar, FusedReduction, IdentityToken, MaxOp, MinOp, ProdOp, SumOp, Wgsl,
};
use leto::LayoutDyn;

use crate::application::strided::{to_i32, to_u32};
use hephaestus_core::{HephaestusError, Result};

/// Provider representation bound for runtime-rank WGSL layout metadata.
pub(crate) const MAX_FUSION_RANK: usize = 8;
pub(crate) const FUSION_WORKGROUP_WIDTH: u32 = 256;

/// Layout record consumed by both the elementwise and reduction shaders.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct FusionLayoutInfo {
    pub(crate) offset: u32,
    pub(crate) rank: u32,
    pub(crate) length: u32,
    pub(crate) axis: u32,
    pub(crate) extent: u32,
    pub(crate) shape: [u32; MAX_FUSION_RANK],
    pub(crate) strides: [i32; MAX_FUSION_RANK],
}

impl FusionLayoutInfo {
    pub(crate) fn from_layout(layout: &LayoutDyn, axis: Option<(usize, usize)>) -> Result<Self> {
        if layout.shape.len() != layout.strides.len() {
            return Err(HephaestusError::DispatchFailed {
                message: "fusion layout shape and stride ranks differ".to_string(),
            });
        }
        if layout.ndim() > MAX_FUSION_RANK {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "WGPU fusion supports rank <= {MAX_FUSION_RANK}, got {}",
                    layout.ndim()
                ),
            });
        }
        let (min_offset, max_offset) =
            layout
                .checked_min_max_offsets()
                .map_err(|error| HephaestusError::DispatchFailed {
                    message: format!("fusion layout rejected: {error}"),
                })?;
        if max_offset > i32::MAX as usize {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "fusion layout address {max_offset} exceeds signed WGSL offset range"
                ),
            });
        }
        debug_assert!(min_offset <= max_offset);

        let mut shape = [1u32; MAX_FUSION_RANK];
        let mut strides = [0i32; MAX_FUSION_RANK];
        for (index, (&dimension, &stride)) in layout.shape.iter().zip(&layout.strides).enumerate() {
            shape[index] = to_u32(dimension, "fusion dimension")?;
            strides[index] = to_i32(stride, "fusion stride")?;
        }
        let length = to_u32(
            layout
                .checked_size()
                .map_err(|error| HephaestusError::DispatchFailed {
                    message: format!("fusion layout rejected: {error}"),
                })?,
            "fusion logical length",
        )?;
        let offset = to_u32(layout.offset, "fusion offset")?;
        let (axis, extent) = axis.map_or((0, 1), |(axis, extent)| (axis, extent));
        Ok(Self {
            offset,
            rank: to_u32(layout.ndim(), "fusion rank")?,
            length,
            axis: to_u32(axis, "fusion axis")?,
            extent: to_u32(extent, "fusion axis extent")?,
            shape,
            strides,
        })
    }
}

/// Scalar capabilities needed by provider-owned WGSL fusion.
///
/// A type implements this seam only when its WGSL representation and all
/// reduction identity tokens are available. This keeps unsupported dialect
/// types out of runtime shader generation.
pub trait WgpuFusionScalar: DialectScalar<Wgsl> + Pod + Send + Sync + 'static {
    /// WGSL identity literal for addition.
    const ZERO: &'static str;
    /// WGSL identity literal for multiplication.
    const ONE: &'static str;
    /// WGSL lower-bound literal for maximum reduction.
    const LOWEST: &'static str;
    /// WGSL upper-bound literal for minimum reduction.
    const HIGHEST: &'static str;
    /// WGSL expression converting the runtime axis length for mean reduction.
    const MEAN_DIVISOR: &'static str;
}

macro_rules! impl_fusion_scalar {
    ($ty:ty, $divisor:literal) => {
        impl WgpuFusionScalar for $ty {
            const ZERO: &'static str = <Self as IdentityToken<SumOp, Wgsl>>::TOKEN;
            const ONE: &'static str = <Self as IdentityToken<ProdOp, Wgsl>>::TOKEN;
            const LOWEST: &'static str = <Self as IdentityToken<MaxOp, Wgsl>>::TOKEN;
            const HIGHEST: &'static str = <Self as IdentityToken<MinOp, Wgsl>>::TOKEN;
            const MEAN_DIVISOR: &'static str = $divisor;
        }
    };
}

impl_fusion_scalar!(f32, "f32(axis_length)");
impl_fusion_scalar!(i32, "i32(axis_length)");
impl_fusion_scalar!(u32, "u32(axis_length)");

pub(crate) fn validate_expression_source(source: &str) -> Result<()> {
    if source.trim().is_empty() {
        return Err(HephaestusError::InvalidConfiguration {
            message: "fusion expression must not be empty".to_string(),
        });
    }
    if source
        .chars()
        .any(|character| matches!(character, ';' | '{' | '}'))
    {
        return Err(HephaestusError::InvalidConfiguration {
            message: "fusion expression must be one WGSL expression fragment".to_string(),
        });
    }
    Ok(())
}

fn wgsl_layout() -> &'static str {
    "struct LayoutInfo {
    offset: u32,
    rank: u32,
    length: u32,
    axis: u32,
    extent: u32,
    shape: array<u32, 8>,
    strides: array<i32, 8>,
}
"
}

fn input_declarations<T: DialectScalar<Wgsl>>(input_count: usize) -> String {
    (0..input_count)
        .map(|index| {
            format!(
                "@group(0) @binding({index}) var<storage, read> storage_input_{index}: array<{}>;\n",
                T::TYPE_TOKEN
            )
        })
        .collect()
}

fn input_loads<T: DialectScalar<Wgsl>>(input_count: usize) -> String {
    (0..input_count)
        .map(|index| {
            format!(
                "    var input_offset_{index}: i32 = i32(layouts[{index}].offset);\n\
                 for (var input_axis_{index}: u32 = 0u; input_axis_{index} < output_layout.rank; input_axis_{index} = input_axis_{index} + 1u) {{\n\
                     if (layouts[{index}].shape[input_axis_{index}] > 1u) {{\n\
                         input_offset_{index} = input_offset_{index} + i32(coordinates[input_axis_{index}]) * layouts[{index}].strides[input_axis_{index}];\n\
                     }}\n\
                 }}\n\
                 let input_{index}: {ty} = storage_input_{index}[u32(input_offset_{index})];\n",
                ty = T::TYPE_TOKEN,
            )
        })
        .collect()
}

fn output_offset() -> &'static str {
    "    var output_offset: i32 = i32(output_layout.offset);
    for (var output_axis: u32 = 0u; output_axis < output_layout.rank; output_axis = output_axis + 1u) {
        output_offset = output_offset + i32(coordinates[output_axis]) * output_layout.strides[output_axis];
    }
"
}

/// Generate the provider-owned wrapper for an elementwise expression.
pub(crate) fn elementwise_source<T: DialectScalar<Wgsl>>(
    input_count: usize,
    expression: &str,
) -> String {
    let output_binding = input_count;
    let layouts_binding = input_count + 1;
    let loads = input_loads::<T>(input_count);
    format!(
        "{layout}\n{inputs}\
@group(0) @binding({output_binding}) var<storage, read_write> output: array<{ty}>;
@group(0) @binding({layouts_binding}) var<storage, read> layouts: array<LayoutInfo>;

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let index = global_id.x;
    let output_layout = layouts[{output_binding}];
    if (index >= output_layout.length) {{
        return;
    }}
    var coordinates: array<u32, 8>;
    var remaining = index;
    for (var coordinate_axis: i32 = i32(output_layout.rank) - 1; coordinate_axis >= 0; coordinate_axis = coordinate_axis - 1) {{
        let dimension = output_layout.shape[u32(coordinate_axis)];
        coordinates[u32(coordinate_axis)] = remaining % dimension;
        remaining = remaining / dimension;
    }}
{loads}    {output_offset}    output[u32(output_offset)] = {expression};
}}
",
        layout = wgsl_layout(),
        inputs = input_declarations::<T>(input_count),
        output_binding = output_binding,
        layouts_binding = layouts_binding,
        ty = T::TYPE_TOKEN,
        width = FUSION_WORKGROUP_WIDTH,
        loads = loads,
        output_offset = output_offset(),
        expression = expression,
    )
}

/// Generate the provider-owned wrapper for an expression followed by an axis
/// reduction. The reduced extent and axis travel in the output metadata, so
/// the same pipeline serves every axis length and output view with this rank.
pub(crate) fn reduction_source<T: WgpuFusionScalar>(
    input_count: usize,
    expression: &str,
    reduction: FusedReduction,
) -> Result<String> {
    let output_binding = input_count;
    let layouts_binding = input_count + 1;
    let (identity, combine, final_value) = match reduction {
        FusedReduction::Sum => (T::ZERO, "lhs + rhs", "acc".to_string()),
        FusedReduction::Product => (T::ONE, "lhs * rhs", "acc".to_string()),
        FusedReduction::Mean => (T::ZERO, "lhs + rhs", format!("acc / {}", T::MEAN_DIVISOR)),
        FusedReduction::Maximum => (T::LOWEST, "max(lhs, rhs)", "acc".to_string()),
        FusedReduction::Minimum => (T::HIGHEST, "min(lhs, rhs)", "acc".to_string()),
        _ => {
            return Err(HephaestusError::InvalidConfiguration {
                message: "unsupported fused reduction operation".to_string(),
            });
        }
    };
    Ok(format!(
        "{layout}\n{inputs}
@group(0) @binding({output_binding}) var<storage, read_write> output: array<{ty}>;
@group(0) @binding({layouts_binding}) var<storage, read> layouts: array<LayoutInfo>;

@compute @workgroup_size({width})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let index = global_id.x;
    let output_layout = layouts[{output_binding}];
    if (index >= output_layout.length) {{
        return;
    }}
    var coordinates: array<u32, 8>;
    var remaining = index;
    for (var coordinate_axis: i32 = i32(output_layout.rank) - 1; coordinate_axis >= 0; coordinate_axis = coordinate_axis - 1) {{
        let dimension = output_layout.shape[u32(coordinate_axis)];
        coordinates[u32(coordinate_axis)] = remaining % dimension;
        remaining = remaining / dimension;
    }}
    var acc: {ty} = {identity};
    let axis_length = output_layout.extent;
    for (var reduction_index: u32 = 0u; reduction_index < axis_length; reduction_index = reduction_index + 1u) {{
        coordinates[output_layout.axis] = reduction_index;
{loads}        let expression_value = {expression};
        let lhs = acc;
        let rhs = expression_value;
        acc = {combine};
    }}
    coordinates[output_layout.axis] = 0u;
{output_offset}    output[u32(output_offset)] = {final_value};
}}
",
        layout = wgsl_layout(),
        inputs = input_declarations::<T>(input_count),
        output_binding = output_binding,
        layouts_binding = layouts_binding,
        ty = T::TYPE_TOKEN,
        width = FUSION_WORKGROUP_WIDTH,
        identity = identity,
        combine = combine,
        final_value = final_value,
        loads = input_loads::<T>(input_count),
        expression = expression,
        output_offset = output_offset(),
    ))
}
