use bytemuck::{Pod, Zeroable};

/// Rank-eight packed metadata shared by every stateful-update backend.
///
/// Operands are ordered parameter, gradient, state zero, state one. Unused
/// state-one lanes are zero. The representation matches WGSL `vec4` alignment
/// and C-family four-element arrays without translation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct StatefulUpdateMeta {
    /// Right-aligned logical shape.
    pub shape: [[u32; 4]; 2],
    /// Right-aligned element strides for each operand.
    pub strides: [[[i32; 4]; 2]; 4],
    /// Base element offset for each operand.
    pub offsets: [u32; 4],
    /// Logical dispatch length in lane zero; remaining lanes are padding.
    pub dispatch: [u32; 4],
}

const _: () = assert!(core::mem::size_of::<StatefulUpdateMeta>() == 192);
