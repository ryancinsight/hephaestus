use eunomia::Pod;

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Native scalar contract for accelerator attention.
///
/// The sealed implementation set intentionally admits only `f32` and `f64`.
/// Reduced-precision types remain unsupported until every advertised provider
/// has native arithmetic and shared conformance coverage.
pub trait AttentionScalar: private::Sealed + Pod + Copy + Send + Sync + 'static {
    /// Return whether this scalar is finite.
    fn is_finite(self) -> bool;
}

impl AttentionScalar for f32 {
    #[inline]
    fn is_finite(self) -> bool {
        self.is_finite()
    }
}

impl AttentionScalar for f64 {
    #[inline]
    fn is_finite(self) -> bool {
        self.is_finite()
    }
}
