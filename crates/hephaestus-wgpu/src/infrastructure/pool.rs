//! Bounded buffer pools for transient WGPU allocations.

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
/// The pool is intentionally unordered: callers only need a buffer at least as
/// large as the requested size, and recycling keeps retention bounded under
/// adversarial alternating transfer sizes.
#[derive(Debug)]
pub(crate) struct BoundedBufferPool<B> {
    buffers: Vec<B>,
    retained_bytes: u64,
    max_buffers: usize,
    max_bytes: u64,
}

impl<B: PoolBuffer> BoundedBufferPool<B> {
    /// Construct an empty pool with explicit retention limits.
    #[must_use]
    pub(crate) fn new(max_buffers: usize, max_bytes: u64) -> Self {
        Self {
            buffers: Vec::new(),
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
            .position(|buffer| buffer.size() >= size)?;
        let buffer = self.buffers.swap_remove(pos);
        self.retained_bytes -= buffer.size();
        Some(buffer)
    }

    /// Retain a buffer for reuse when it fits the pool limits.
    pub(crate) fn recycle(&mut self, buffer: B) {
        let size = buffer.size();
        if size > self.max_bytes || self.buffers.len() >= self.max_buffers {
            return;
        }
        while self.retained_bytes + size > self.max_bytes {
            let Some(evicted) = self.buffers.pop() else {
                return;
            };
            self.retained_bytes -= evicted.size();
        }
        self.retained_bytes += size;
        self.buffers.push(buffer);
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
}
