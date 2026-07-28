//! Inline host-side bind-group descriptor storage for WGPU dispatch.

use smallvec::SmallVec;

/// Common dispatches have at most four resources including their parameter
/// uniform. Larger kernels spill to the heap without changing the API.
pub(crate) const INLINE_BIND_GROUP_ENTRIES: usize = 4;

/// Bind-group entries with inline storage for the common dispatch shapes.
pub(crate) type BindGroupEntries<'a> =
    SmallVec<[wgpu::BindGroupEntry<'a>; INLINE_BIND_GROUP_ENTRIES]>;

/// Common grouped dispatches use at most four bind groups.
pub(crate) const INLINE_BIND_GROUPS: usize = 4;

/// Bind groups with inline storage for the common grouped dispatch shapes.
pub(crate) type BindGroups = SmallVec<[(u32, wgpu::BindGroup); INLINE_BIND_GROUPS]>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_bind_group_descriptors_stay_inline() {
        let entries: BindGroupEntries<'static> = BindGroupEntries::new();
        assert_eq!(entries.capacity(), INLINE_BIND_GROUP_ENTRIES);
        assert!(!entries.spilled());

        let groups = BindGroups::new();
        assert_eq!(groups.capacity(), INLINE_BIND_GROUPS);
        assert!(!groups.spilled());
    }
}
