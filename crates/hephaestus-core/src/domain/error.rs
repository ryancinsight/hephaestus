/// Errors shared by every accelerator backend.
///
/// Backend-specific failures are carried in the message of the matching
/// variant rather than collapsed into a stringly catch-all: each variant is a
/// distinct, caller-actionable failure mode (no adapter, device init failed,
/// allocation rejected, host/device size mismatch, kernel dispatch failed).
#[derive(Debug, thiserror::Error)]
pub enum HephaestusError {
    /// No compatible adapter/physical device was found on this host.
    #[error("no compatible accelerator adapter available: {message}")]
    AdapterUnavailable {
        /// Backend-reported detail.
        message: String,
    },
    /// An adapter was found but logical-device creation failed.
    #[error("accelerator device creation failed: {message}")]
    DeviceUnavailable {
        /// Backend-reported detail.
        message: String,
    },
    /// A device allocation request was invalid or rejected before a buffer was
    /// created.
    #[error("accelerator allocation failed: {message}")]
    AllocationFailed {
        /// Backend-reported detail.
        message: String,
    },
    /// A host slice length does not match the device buffer's element count.
    #[error("host length {host_len} does not match device buffer length {device_len}")]
    LengthMismatch {
        /// Elements on the host side.
        host_len: usize,
        /// Elements in the device buffer.
        device_len: usize,
    },
    /// A kernel dispatch or device-side execution failure.
    #[error("kernel dispatch failed: {message}")]
    DispatchFailed {
        /// Backend-reported detail.
        message: String,
    },
    /// A configuration or argument value supplied to a kernel or dispatch
    /// parameter was invalid.
    #[error("invalid configuration: {message}")]
    InvalidConfiguration {
        /// Caller-facing detail.
        message: String,
    },
    /// A device-to-host or host-to-device transfer failure.
    #[error("device transfer failed: {message}")]
    TransferFailed {
        /// Backend-reported detail.
        message: String,
    },
    /// A wait for device completion reached its deadline with the submission
    /// still outstanding.
    ///
    /// Distinct from [`HephaestusError::TransferFailed`]: nothing reported a
    /// fault, the device simply did not answer within the bound. A caller acts
    /// on the two differently — a transfer failure is terminal for that
    /// operation, while an elapsed deadline names an unresponsive or
    /// over-subscribed device the caller can reacquire, re-submit against, or
    /// escalate as a hang. Collapsing it into a transfer message would leave
    /// that decision to string matching.
    #[error("device wait exceeded its {deadline:?} deadline: {message}")]
    DeviceWaitTimeout {
        /// The deadline that elapsed.
        deadline: core::time::Duration,
        /// Which wait timed out.
        message: String,
    },
}

/// Result alias for accelerator operations.
pub type Result<T> = core::result::Result<T, HephaestusError>;
