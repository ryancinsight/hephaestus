use bytemuck::Pod;

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
}

/// Portable scalar contract for accelerator cross-entropy.
///
/// The shared provider contract intentionally admits only `f32`, the scalar
/// supported natively by every shipped device API. Additional scalar types are
/// admitted only when every provider has native arithmetic and conformance
/// coverage.
pub trait CrossEntropyScalar: private::Sealed + Pod + Copy + Send + Sync + 'static {}

impl CrossEntropyScalar for f32 {}
