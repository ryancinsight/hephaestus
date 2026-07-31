mod backward;
mod common;
mod forward;

pub(super) use backward::{backward_source, score_gradient_source};
pub(super) use forward::forward_source;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GradientKernel {
    Query,
    Key,
    Value,
}

impl GradientKernel {
    pub(super) const fn entry(self) -> &'static str {
        match self {
            Self::Query => "attention_query_gradient",
            Self::Key => "attention_key_gradient",
            Self::Value => "attention_value_gradient",
        }
    }
}
