/// A typed, device-resident linear buffer of `len` elements of `T`.
///
/// The element type is carried in the type system (backends store a
/// `PhantomData<T>` next to the raw device allocation), so a buffer created
/// for one scalar type cannot be passed to a kernel expecting another —
/// dtype confusion is a compile error, not a runtime check.
pub trait DeviceBuffer<T> {
    /// Number of `T` elements in the buffer.
    fn len(&self) -> usize;

    /// Returns true when the buffer holds zero elements.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
