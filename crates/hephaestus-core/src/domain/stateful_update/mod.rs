//! Stateful parameter updates owned by the accelerator provider.
//!
//! A rule marker selects one formula at compile time. Parameters and layouts
//! remain dispatch data, so changing a learning rate or tensor view reuses the
//! compiled kernel. All operands are borrowed device views; a step performs no
//! host materialization and allocates no parameter or state storage.

mod metadata;
mod operands;
mod ops;
mod parameters;
mod plan;
mod rules;

pub use metadata::StatefulUpdateMeta;
pub use operands::{StatefulUpdateAliasing, StatefulUpdateOperands};
pub use ops::StatefulUpdateOps;
pub use parameters::{
    AdaGradParameters, AdamParameters, AdamWParameters, RmsPropParameters, SgdParameters,
};
pub use plan::{StatefulUpdatePlan, plan_stateful_update};
pub use rules::{AdaGrad, Adam, AdamW, RmsProp, Sgd, StatefulUpdateRule};
