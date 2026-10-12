use hephaestus_core::{HephaestusError, Result};

use crate::RocmBuffer;

pub(super) fn reject_output_alias<T, U>(
    input_label: &'static str,
    input: &RocmBuffer<T>,
    out: &RocmBuffer<U>,
) -> Result<()> {
    if input.aliases(out) {
        return Err(HephaestusError::DispatchFailed {
            message: format!("output buffer must not alias {input_label} input"),
        });
    }
    Ok(())
}

pub(super) fn checked_work_items(len: usize) -> Result<u32> {
    u32::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("ROCm elementwise length {len} exceeds the HIP kernel argument range"),
    })
}
