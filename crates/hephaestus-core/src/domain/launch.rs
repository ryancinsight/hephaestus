/// Compute-kernel block width (threads per block / workgroup).
///
/// The typed parameter through which occupancy planning (atlas ADR 0002:
/// moirai intersects themis `GpuTopology` with mnemosyne
/// `KernelResourceBudget`) reaches backend dispatch, replacing baked-in
/// workgroup constants. A validating `#[repr(transparent)]` newtype over a
/// non-zero width: a zero-wide block is meaningless and would poison
/// grid-size arithmetic, so it is unrepresentable rather than checked at
/// every dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct BlockWidth(core::num::NonZeroU32);

impl BlockWidth {
    /// The default dispatch width used when no occupancy plan supplies one.
    /// 256 is the portable sweet spot: a multiple of every common warp/
    /// wavefront width (32, 64) and within every backend's workgroup limits.
    pub const DEFAULT: Self = match Self::new(256) {
        Some(width) => width,
        None => panic!("invariant: default block width is non-zero"),
    };

    /// Construct a width; `None` for zero.
    #[must_use]
    #[inline]
    pub const fn new(width: u32) -> Option<Self> {
        match core::num::NonZeroU32::new(width) {
            Some(non_zero) => Some(Self(non_zero)),
            None => None,
        }
    }

    /// The width in threads.
    #[must_use]
    #[inline]
    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// Blocks needed to cover `work_items` elements with one thread each.
    ///
    /// Returns `None` when the grid would exceed the backend-portable `u32`
    /// block-count range. Backends that report a typed dispatch error should
    /// use this checked form; [`covering_blocks`](Self::covering_blocks)
    /// remains for callers that intentionally use saturation as a signal.
    #[must_use]
    #[inline]
    pub const fn checked_covering_blocks(self, work_items: u64) -> Option<u32> {
        let blocks = work_items.div_ceil(self.0.get() as u64);
        if blocks > u32::MAX as u64 {
            None
        } else {
            Some(blocks as u32)
        }
    }

    /// Blocks needed to cover `work_items` elements with one thread each,
    /// saturating at `u32::MAX` (callers splitting such grids detect
    /// saturation by comparing covered work).
    #[must_use]
    #[inline]
    pub const fn covering_blocks(self, work_items: u64) -> u32 {
        match self.checked_covering_blocks(work_items) {
            Some(blocks) => blocks,
            None => u32::MAX,
        }
    }
}

impl Default for BlockWidth {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_width_is_unrepresentable() {
        assert_eq!(BlockWidth::new(0).map(BlockWidth::get), None);
        assert_eq!(BlockWidth::DEFAULT.get(), 256);
        assert_eq!(BlockWidth::default(), BlockWidth::DEFAULT);
    }

    #[test]
    fn covering_blocks_is_ceil_division() {
        let width = BlockWidth::new(256).unwrap();
        assert_eq!(width.covering_blocks(0), 0);
        assert_eq!(width.covering_blocks(1), 1);
        assert_eq!(width.covering_blocks(256), 1);
        assert_eq!(width.covering_blocks(257), 2);
        assert_eq!(width.covering_blocks(u64::MAX), u32::MAX);
        assert_eq!(width.checked_covering_blocks(0), Some(0));
        assert_eq!(width.checked_covering_blocks(1), Some(1));
        assert_eq!(width.checked_covering_blocks(256), Some(1));
        assert_eq!(width.checked_covering_blocks(257), Some(2));
        assert_eq!(
            width.checked_covering_blocks(u64::from(width.get()) * u64::from(u32::MAX)),
            Some(u32::MAX)
        );
        assert_eq!(
            width.checked_covering_blocks(u64::from(width.get()) * u64::from(u32::MAX) + 1),
            None
        );
    }
}
