//! Unary and binary elementwise operations for the host reference device
//! (ADR 0061).
//!
//! `Host` never renders a kernel: [`HostElementwiseOps`] resolves each
//! operator through its ADR 0061 value function, captured as a plain
//! function pointer at prepare time (mirroring `crate::combine`'s
//! `combine_fn`/`unsupported_operator` pattern for the reduction and scan
//! seams). The bridge from the generic `T: Pod` seam to the two differently
//! bounded value-method families (`UnaryExpr::value`/`BinaryExpr::real_value`
//! need `T: RealField`; `BinaryExpr::value`/`TypedBinaryExpr::value` need
//! only `T: NumericElement`) is `ElementwiseDispatch` (ADR 0061
//! Decision 5): `f32`/`f64` resolve through the `RealField` methods,
//! `u32`/`i32`/`F16`/`Bf16` through the `NumericElement` methods, and a
//! real-only unary operator on a non-real scalar reports no application at
//! all — the seam turns that into the typed
//! [`HephaestusError::DispatchFailed`] naming the operator, identically to an
//! operator with no host value function whatsoever.
//!
//! # Broadcasting, aliasing, and output semantics
//!
//! Mirrors `hephaestus-wgpu`'s `WgpuElementwiseOps`: each input broadcasts to
//! the output's shape with leto's own broadcast rules (arbitrary rank `N`),
//! the output buffer must not alias any input buffer, and the output layout
//! must be non-overlapping (validated once at prepare time; re-validated by
//! [`leto::ArrayViewMut::try_iter_mut`] at dispatch, since the prepared form
//! borrows the buffers rather than copying them). A prepared dispatch re-reads
//! its bound buffers fresh on every call, so writes made after preparation are
//! observed; the broadcast shapes themselves are fixed at preparation, since
//! rebind semantics apply to buffer contents only.
//!
//! # Aliased operands
//!
//! `binary_into`/`typed_binary_into` may name the same buffer as both `lhs`
//! and `rhs` (e.g. `mul(a, a)`) with independent layouts (e.g. `a` against
//! its own transpose): [`HostBuffer`] is one `RwLock`, so a second `read()`
//! on the same lock is undefined by `std::sync::RwLock`'s own contract
//! (`crate::operands` documents this for the other host seams). Dispatch
//! detects the alias and reads the shared cells once, viewing them through
//! each operand's own broadcast layout.

mod binary;
mod dispatch;
mod scalar;
mod seam;
mod unary;
mod view;

pub use binary::HostPreparedBinary;
pub use scalar::HostPreparedScalar;
pub use seam::HostElementwiseOps;
pub use unary::HostPreparedUnary;

pub(crate) use view::{broadcast_operand, map_layout_err, write_next};
