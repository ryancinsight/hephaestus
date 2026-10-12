mod adagrad;
mod adam;
mod adamw;
mod rmsprop;
mod rule;
mod sgd;
#[cfg(test)]
mod tests;

pub use adagrad::AdaGrad;
pub use adam::Adam;
pub use adamw::AdamW;
pub use rmsprop::RmsProp;
pub use rule::{StatefulUpdateRule, StatefulUpdateStep, StatefulUpdateStepF64};
pub use sgd::Sgd;

mod sealed {
    pub trait Sealed {}
}
