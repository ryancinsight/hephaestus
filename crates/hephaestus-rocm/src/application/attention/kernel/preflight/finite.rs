use hephaestus_core::AttentionSemanticStatus;

use super::super::common::prelude;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::application::attention) enum FiniteOperand {
    Query,
    Key,
    Value,
    Keep,
    GradOutput,
    Weights,
}

impl FiniteOperand {
    pub(in crate::application::attention) const fn entry(self) -> &'static str {
        match self {
            Self::Query => "hephaestus_attention_finite_query",
            Self::Key => "hephaestus_attention_finite_key",
            Self::Value => "hephaestus_attention_finite_value",
            Self::Keep => "hephaestus_attention_finite_keep",
            Self::GradOutput => "hephaestus_attention_finite_grad_output",
            Self::Weights => "hephaestus_attention_finite_weights",
        }
    }

    const fn layout(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Key => "key",
            Self::Value => "value",
            Self::Keep => "keep",
            Self::GradOutput => "grad_output",
            Self::Weights => "weights",
        }
    }

    const fn rank(self) -> usize {
        match self {
            Self::Keep => 2,
            _ => 3,
        }
    }

    const fn failure(self) -> AttentionSemanticStatus {
        match self {
            Self::Query => AttentionSemanticStatus::NonFiniteQuery,
            Self::Key => AttentionSemanticStatus::NonFiniteKey,
            Self::Value => AttentionSemanticStatus::NonFiniteValue,
            Self::Keep => AttentionSemanticStatus::NonFiniteKeep,
            Self::GradOutput => AttentionSemanticStatus::NonFiniteOutputGradient,
            Self::Weights => AttentionSemanticStatus::NonFiniteWeights,
        }
    }
}

pub(in crate::application::attention) fn finite_source(
    scalar: &str,
    operand: FiniteOperand,
) -> String {
    let (elements, coordinates) = if operand.rank() == 2 {
        (
            "parameters.keep.shape[0] * parameters.keep.shape[1]",
            format!(
                "const int first = linear / parameters.{layout}.shape[1];\n    \
                 const int second = linear % parameters.{layout}.shape[1];",
                layout = operand.layout()
            ),
        )
    } else {
        (
            "parameters.LAYOUT.shape[0] * parameters.LAYOUT.shape[1] * parameters.LAYOUT.shape[2]",
            format!(
                "const int plane = parameters.{layout}.shape[1] * parameters.{layout}.shape[2];\n    \
                 const int first = linear / plane;\n    \
                 const int within = linear % plane;\n    \
                 const int second = within / parameters.{layout}.shape[2];\n    \
                 const int third = within % parameters.{layout}.shape[2];",
                layout = operand.layout()
            ),
        )
    };
    let elements = elements.replace("LAYOUT", operand.layout());
    let physical = if operand.rank() == 2 {
        format!("physical2(parameters.{}, first, second)", operand.layout())
    } else {
        format!(
            "physical3(parameters.{}, first, second, third)",
            operand.layout()
        )
    };
    format!(
        r#"{prelude}
#include <math.h>
extern "C" __global__ void {entry}(
    const {scalar}* source,
    unsigned int* status,
    AttentionMeta parameters
) {{
    const int linear = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int elements = {elements};
    if (linear >= elements) return;
    {coordinates}
    if (!isfinite(source[{physical}])) {{
        atomicMin(status, {failure}u);
    }}
}}
"#,
        prelude = prelude(),
        entry = operand.entry(),
        failure = operand.failure().code(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_are_element_parallel_and_use_core_codes() {
        let source = finite_source("float", FiniteOperand::Query);

        assert!(source.contains("blockIdx.x * blockDim.x + threadIdx.x"));
        assert!(source.contains("atomicMin(status, 1u)"));
        assert!(!source.contains("blockIdx.x != 0"));
    }
}
