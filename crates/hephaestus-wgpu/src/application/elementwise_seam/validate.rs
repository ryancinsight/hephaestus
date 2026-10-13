use hephaestus_core::{HephaestusError, Result};

use crate::application::strided::MAX_STRIDED_RANK;

pub(crate) fn validate_rank<const N: usize>() -> Result<()> {
    if N > MAX_STRIDED_RANK {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "elementwise dispatch supports rank <= {MAX_STRIDED_RANK}, got rank {N}"
            ),
        });
    }
    Ok(())
}
