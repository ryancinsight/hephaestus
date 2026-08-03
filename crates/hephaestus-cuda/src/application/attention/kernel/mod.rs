mod backward;
mod forward;
mod preflight;
mod prelude;

pub(super) use backward::{backward_preflight_source, backward_source, score_gradient_source};
pub(super) use forward::forward_source;
pub(super) use preflight::{backward_validation_source, forward_preflight_source};

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

    pub(super) const fn preflight_entry(self) -> &'static str {
        match self {
            Self::Query => "attention_query_gradient_preflight",
            Self::Key => "attention_key_gradient_preflight",
            Self::Value => "attention_value_gradient_preflight",
        }
    }
}
