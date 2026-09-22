//! The rendered per-dialect expressions for the runtime-parameter markers.

use super::ParameterizedUnaryExpr;
use super::marker::{
    CeluGradOp, CeluOp, HardshrinkGradOp, HardshrinkOp, HardtanhGradOp, HardtanhOp,
    LeakyReluGradOp, LeakyReluOp, SoftshrinkGradOp, SoftshrinkOp, ThresholdGradOp, ThresholdOp,
};
use crate::domain::dialect::{CudaC, HipC, Wgsl};

impl ParameterizedUnaryExpr<Wgsl> for HardtanhOp {
    const EXPR: &'static str = "select(select(x, second, x > second), first, x < first)";
}

impl ParameterizedUnaryExpr<Wgsl> for HardtanhGradOp {
    const EXPR: &'static str = "select(0.0, 1.0, (x > first) && (x < second))";
}

impl ParameterizedUnaryExpr<Wgsl> for ThresholdOp {
    const EXPR: &'static str = "select(second, x, x > first)";
}

impl ParameterizedUnaryExpr<Wgsl> for ThresholdGradOp {
    const EXPR: &'static str = "select(0.0, 1.0, x > first)";
}

impl ParameterizedUnaryExpr<Wgsl> for LeakyReluOp {
    const EXPR: &'static str = "select(first * x, x, x >= 0.0)";
}

impl ParameterizedUnaryExpr<Wgsl> for LeakyReluGradOp {
    const EXPR: &'static str = "select(first, 1.0, x > 0.0)";
}

impl ParameterizedUnaryExpr<Wgsl> for HardshrinkOp {
    const EXPR: &'static str = "select(0.0, x, abs(x) > first)";
}

impl ParameterizedUnaryExpr<Wgsl> for HardshrinkGradOp {
    const EXPR: &'static str = "select(0.0, 1.0, abs(x) > first)";
}

impl ParameterizedUnaryExpr<Wgsl> for SoftshrinkOp {
    const EXPR: &'static str = "select(select(0.0, x - first, x > first), x + first, x < -first)";
}

impl ParameterizedUnaryExpr<Wgsl> for SoftshrinkGradOp {
    const EXPR: &'static str = "select(0.0, 1.0, (x > first) || (x < -first))";
}

impl ParameterizedUnaryExpr<Wgsl> for CeluOp {
    const EXPR: &'static str = "select(first * (exp(x / first) - 1.0), x, x >= 0.0)";
}

impl ParameterizedUnaryExpr<Wgsl> for CeluGradOp {
    const EXPR: &'static str = "select(exp(x / first), 1.0, x >= 0.0)";
}

macro_rules! impl_c_family {
    ($dialect:ty) => {
        impl ParameterizedUnaryExpr<$dialect> for HardtanhOp {
            const EXPR: &'static str = "x < first ? first : (x > second ? second : x)";
        }

        impl ParameterizedUnaryExpr<$dialect> for HardtanhGradOp {
            const EXPR: &'static str = "(x > first && x < second) ? 1.0 : 0.0";
        }

        impl ParameterizedUnaryExpr<$dialect> for ThresholdOp {
            const EXPR: &'static str = "x > first ? x : second";
        }

        impl ParameterizedUnaryExpr<$dialect> for ThresholdGradOp {
            const EXPR: &'static str = "x > first ? 1.0 : 0.0";
        }

        impl ParameterizedUnaryExpr<$dialect> for LeakyReluOp {
            const EXPR: &'static str = "x >= 0.0f ? x : first * x";
        }

        impl ParameterizedUnaryExpr<$dialect> for LeakyReluGradOp {
            const EXPR: &'static str = "x > 0.0f ? 1.0f : first";
        }

        impl ParameterizedUnaryExpr<$dialect> for HardshrinkOp {
            const EXPR: &'static str = "fabsf(x) > first ? x : 0.0f";
        }

        impl ParameterizedUnaryExpr<$dialect> for HardshrinkGradOp {
            const EXPR: &'static str = "fabsf(x) > first ? 1.0f : 0.0f";
        }

        impl ParameterizedUnaryExpr<$dialect> for SoftshrinkOp {
            const EXPR: &'static str = "x > first ? x - first : (x < -first ? x + first : 0.0f)";
        }

        impl ParameterizedUnaryExpr<$dialect> for SoftshrinkGradOp {
            const EXPR: &'static str = "(x > first || x < -first) ? 1.0f : 0.0f";
        }

        impl ParameterizedUnaryExpr<$dialect> for CeluOp {
            const EXPR: &'static str = "x >= 0.0f ? x : first * (expf(x / first) - 1.0f)";
        }

        impl ParameterizedUnaryExpr<$dialect> for CeluGradOp {
            const EXPR: &'static str = "x >= 0.0f ? 1.0f : expf(x / first)";
        }
    };
}

impl_c_family!(CudaC);

impl_c_family!(HipC);
