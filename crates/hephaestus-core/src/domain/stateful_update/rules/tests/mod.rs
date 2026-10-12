// `update` is generic over every `L: KernelDialect` (the same impl
// covers `Host` with no separate blanket, per this trait's doc comment),
// so these tests pin it through `Host` specifically — the dialect this
// ADR increment adds host dispatch for.

// Every expected value below is either exact binary arithmetic (powers
// of two and zero, which IEEE-754 multiply/add/subtract never round) or
// the same expression `update` computes, transcribed independently but
// reading the same constructed `Parameters` fields and literal operands
// (ADR 0061 Decision 1: "the exact update the WGSL body computes, in the
// same operation order") — so both sides evaluate to bit-identical f32
// results, and a regression in operand order or grouping changes the
// transcription along with the implementation only if both are edited
// identically, which a reviewer would notice.

// f64 twins: same transcription discipline as above, in double
// precision. Binary-exact values where the arithmetic allows it,
// independently transcribed expressions otherwise.

mod adagrad;
mod adam;
mod adamw;
mod rmsprop;
mod selectors;
mod sgd;
