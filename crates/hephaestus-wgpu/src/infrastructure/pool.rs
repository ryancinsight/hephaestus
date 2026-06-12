//! Bounded buffer pools for transient WGPU allocations.

use std::collections::VecDeque;

/// A buffer value whose retained allocation size is known.
pub(crate) trait PoolBuffer {
    /// Allocation size in bytes.
    fn size(&self) -> u64;
}

impl PoolBuffer for wgpu::Buffer {
    #[inline]
    fn size(&self) -> u64 {
        self.size()
    }
}

/// Count- and byte-bounded pool for transient buffers.
///
/// Reuse selects the smallest retained buffer that covers the requested
/// capacity, while count-limit eviction removes the oldest retained buffer so
/// the pool can adapt after size-regime changes.
#[derive(Debug)]
pub(crate) struct BoundedBufferPool<B> {
    buffers: VecDeque<B>,
    retained_bytes: u64,
    max_buffers: usize,
    max_bytes: u64,
}

impl<B: PoolBuffer> BoundedBufferPool<B> {
    /// Construct an empty pool with explicit retention limits.
    #[must_use]
    pub(crate) fn new(max_buffers: usize, max_bytes: u64) -> Self {
        Self {
            buffers: VecDeque::new(),
            retained_bytes: 0,
            max_buffers,
            max_bytes,
        }
    }

    /// Remove and return any retained buffer whose allocation covers `size`.
    pub(crate) fn take_at_least(&mut self, size: u64) -> Option<B> {
        let pos = self
            .buffers
            .iter()
            .enumerate()
            .filter(|(_, buffer)| buffer.size() >= size)
            .min_by_key(|(_, buffer)| buffer.size())
            .map(|(pos, _)| pos)?;
        let buffer = self
            .buffers
            .remove(pos)
            .expect("invariant: position came from VecDeque::position");
        self.retained_bytes -= buffer.size();
        Some(buffer)
    }

    /// Retain a buffer for reuse when it fits the pool limits.
    pub(crate) fn recycle(&mut self, buffer: B) {
        let size = buffer.size();
        if size > self.max_bytes || self.max_buffers == 0 {
            return;
        }
        while self.buffers.len() >= self.max_buffers && !self.buffers.is_empty() {
            let evicted = self
                .buffers
                .pop_front()
                .expect("invariant: non-empty pool has an oldest buffer");
            self.retained_bytes -= evicted.size();
        }
        while self.retained_bytes + size > self.max_bytes {
            let Some(evicted) = self.buffers.pop_back() else {
                return;
            };
            self.retained_bytes -= evicted.size();
        }
        self.retained_bytes += size;
        self.buffers.push_back(buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestBuffer(u64);

    impl PoolBuffer for TestBuffer {
        fn size(&self) -> u64 {
            self.0
        }
    }

    #[test]
    fn pool_reuses_buffer_covering_request() {
        let mut pool = BoundedBufferPool::new(4, 1024);
        pool.recycle(TestBuffer(128));
        pool.recycle(TestBuffer(512));

        let got = pool.take_at_least(256).unwrap();
        assert_eq!(got.size(), 512);
        assert!(pool.take_at_least(256).is_none());
    }

    #[test]
    fn pool_uses_smallest_sufficient_buffer() {
        let mut pool = BoundedBufferPool::new(4, 4096);
        pool.recycle(TestBuffer(2048));
        pool.recycle(TestBuffer(512));
        pool.recycle(TestBuffer(1024));

        let got = pool.take_at_least(500).unwrap();
        assert_eq!(got.size(), 512);

        let large = pool.take_at_least(2048).unwrap();
        assert_eq!(large.size(), 2048);
    }

    #[test]
    fn pool_enforces_count_and_byte_limits() {
        let mut pool = BoundedBufferPool::new(2, 512);
        pool.recycle(TestBuffer(128));
        pool.recycle(TestBuffer(256));
        pool.recycle(TestBuffer(256));

        assert!(pool.retained_bytes <= 512);
        assert!(pool.buffers.len() <= 2);

        pool.recycle(TestBuffer(1024));
        assert!(pool.take_at_least(1024).is_none());
    }

    #[test]
    fn pool_adapts_when_full_of_smaller_buffers() {
        let mut pool = BoundedBufferPool::new(2, 2048);
        pool.recycle(TestBuffer(128));
        pool.recycle(TestBuffer(256));
        pool.recycle(TestBuffer(1024));

        assert!(pool.buffers.len() <= 2);
        let got = pool.take_at_least(1024).unwrap();
        assert_eq!(got.size(), 1024);
    }

    #[test]
    fn zero_count_pool_retains_nothing() {
        let mut pool = BoundedBufferPool::new(0, 2048);
        pool.recycle(TestBuffer(128));

        assert!(pool.take_at_least(1).is_none());
        assert_eq!(pool.retained_bytes, 0);
    }
}
