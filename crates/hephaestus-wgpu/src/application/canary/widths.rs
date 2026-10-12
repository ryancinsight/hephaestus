use std::any::TypeId;

/// WGSL math builtins with recorded or probeable driver behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MathBuiltin {
    /// `exp(x)`: degraded to f32 grade at f64 on the recorded adapter.
    Exp,
    /// `log(x)`: aborts pipeline compilation at f64 on the recorded adapter.
    Log,
    /// `pow(x, y)`: passes validation, access-violates at dispatch at f64.
    Pow,
    /// `tanh(x)`: aborts pipeline compilation at f64 on the recorded adapter.
    Tanh,
}

/// Element width under test.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CanaryWidth {
    /// 32-bit floats.
    F32,
    /// 64-bit floats.
    F64,
}

/// Map a Rust scalar to its canary width; non-float scalars return `None`.
#[must_use]
pub fn width_of<T: 'static>() -> Option<CanaryWidth> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        Some(CanaryWidth::F32)
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        Some(CanaryWidth::F64)
    } else {
        None
    }
}
