use super::super::buffer::DeviceBuffer;
use super::super::error::{HephaestusError, Result};
use super::super::parameterized::validate_parameterized_output;
use super::metadata::StatefulUpdateMeta;
use super::operands::{StatefulUpdateAliasing, StatefulUpdateOperands};

/// Validated launch information for one stateful update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatefulUpdatePlan {
    meta: StatefulUpdateMeta,
}

impl StatefulUpdatePlan {
    /// Logical element count to dispatch.
    #[must_use]
    pub const fn len(self) -> usize {
        self.meta.dispatch[0] as usize
    }

    /// Whether the validated operation is an empty no-op.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.meta.dispatch[0] == 0
    }

    /// Packed backend-neutral launch metadata.
    #[must_use]
    pub const fn metadata(self) -> StatefulUpdateMeta {
        self.meta
    }
}

/// Validate all layouts, state cardinality, and aliasing before mutation.
///
/// # Errors
///
/// Returns an error when the rule's state count is unsupported or mismatched,
/// shapes differ, a storage span is invalid, a writable layout overlaps
/// itself, or any operand buffers alias.
pub fn plan_stateful_update<B, const N: usize>(
    operands: StatefulUpdateOperands<'_, B, N>,
    state_count: usize,
    aliases: StatefulUpdateAliasing,
) -> Result<StatefulUpdatePlan>
where
    B: DeviceBuffer<f32>,
{
    if N > 8 {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("stateful update supports rank <= 8, got rank {N}"),
        });
    }
    if !(1..=2).contains(&state_count) || operands.states.len() != state_count {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "stateful update requires {state_count} state views, got {}",
                operands.states.len()
            ),
        });
    }
    if aliases.any(state_count) {
        return Err(HephaestusError::DispatchFailed {
            message: "stateful update operand buffers must be pairwise distinct".to_string(),
        });
    }

    let shape = operands.parameter.layout.shape();
    if operands.gradient.layout.shape() != shape
        || operands
            .states
            .iter()
            .any(|state| state.layout.shape() != shape)
    {
        return Err(HephaestusError::DispatchFailed {
            message: "stateful update operand shapes must match exactly".to_string(),
        });
    }
    let len =
        validate_parameterized_output(operands.parameter.layout, operands.parameter.buffer.len())?;
    operands
        .gradient
        .layout
        .validate_storage_len(operands.gradient.buffer.len())
        .map_err(|error| HephaestusError::DispatchFailed {
            message: format!("gradient layout rejected: {error}"),
        })?;
    for (index, state) in operands.states.iter().enumerate() {
        validate_parameterized_output(state.layout, state.buffer.len()).map_err(|error| {
            HephaestusError::DispatchFailed {
                message: format!("state {index} layout rejected: {error}"),
            }
        })?;
    }
    let mut strides = [[[0_i32; 4]; 2]; 4];
    let mut offsets = [0_u32; 4];
    let mut shape_padded = [[1_u32; 4]; 2];
    for (axis, &extent) in shape.iter().enumerate() {
        let target = 8 - N + axis;
        shape_padded[target / 4][target % 4] =
            u32::try_from(extent).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("dimension {extent} exceeds u32 range"),
            })?;
    }
    let views = [
        Some(operands.parameter),
        Some(operands.gradient),
        operands.states.first().copied(),
        operands.states.get(1).copied(),
    ];
    for (operand, view) in views.into_iter().enumerate() {
        let Some(view) = view else { continue };
        if len != 0 {
            let (_, maximum) = view.layout.checked_min_max_offsets().map_err(|error| {
                HephaestusError::DispatchFailed {
                    message: format!("operand {operand} address range rejected: {error}"),
                }
            })?;
            if maximum > i32::MAX as usize {
                return Err(HephaestusError::DispatchFailed {
                    message: format!(
                        "operand {operand} maximum offset {maximum} exceeds device i32 address range"
                    ),
                });
            }
        }
        offsets[operand] =
            u32::try_from(view.layout.offset()).map_err(|_| HephaestusError::DispatchFailed {
                message: format!(
                    "operand {operand} offset {} exceeds u32 range",
                    view.layout.offset()
                ),
            })?;
        for (axis, &stride) in view.layout.strides().iter().enumerate() {
            let target = 8 - N + axis;
            strides[operand][target / 4][target % 4] =
                i32::try_from(stride).map_err(|_| HephaestusError::DispatchFailed {
                    message: format!("operand {operand} stride {stride} exceeds i32 range"),
                })?;
        }
    }
    let dispatch_len = u32::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("stateful update length {len} exceeds u32 range"),
    })?;
    Ok(StatefulUpdatePlan {
        meta: StatefulUpdateMeta {
            shape: shape_padded,
            strides,
            offsets,
            dispatch: [dispatch_len, 0, 0, 0],
        },
    })
}

#[cfg(test)]
mod tests {
    use leto::Layout;

    use super::*;
    use crate::StridedView;

    struct Buffer(usize);
    impl DeviceBuffer<f32> for Buffer {
        fn len(&self) -> usize {
            self.0
        }
        fn tier(&self) -> themis::MemoryTier {
            themis::MemoryTier::Dram
        }
    }

    #[test]
    fn rejects_aliasing_before_launch() {
        let layout = Layout::c_contiguous([2]).expect("valid layout");
        let parameter = Buffer(2);
        let gradient = Buffer(2);
        let state = Buffer(2);
        let states = [StridedView::new(&state, &layout)];
        let operands = StatefulUpdateOperands {
            parameter: StridedView::new(&parameter, &layout),
            gradient: StridedView::new(&gradient, &layout),
            states: &states,
        };
        let aliases = StatefulUpdateAliasing {
            parameter_gradient: true,
            ..Default::default()
        };
        assert!(plan_stateful_update(operands, 1, aliases).is_err());
    }

    #[test]
    fn accepts_distinct_strided_views() {
        let layout = Layout::try_new([2, 2], [1, 2], 0).expect("valid test layout");
        let parameter = Buffer(4);
        let gradient = Buffer(4);
        let state = Buffer(4);
        let states = [StridedView::new(&state, &layout)];
        let operands = StatefulUpdateOperands {
            parameter: StridedView::new(&parameter, &layout),
            gradient: StridedView::new(&gradient, &layout),
            states: &states,
        };
        assert_eq!(
            plan_stateful_update(operands, 1, StatefulUpdateAliasing::default())
                .expect("valid plan")
                .len(),
            4
        );
    }

    #[test]
    fn packs_rank_eight_without_losing_axes() {
        let layout =
            Layout::c_contiguous([1, 1, 1, 1, 1, 1, 2, 3]).expect("valid rank-eight layout");
        let parameter = Buffer(6);
        let gradient = Buffer(6);
        let state = Buffer(6);
        let states = [StridedView::new(&state, &layout)];
        let operands = StatefulUpdateOperands {
            parameter: StridedView::new(&parameter, &layout),
            gradient: StridedView::new(&gradient, &layout),
            states: &states,
        };
        let meta = plan_stateful_update(operands, 1, StatefulUpdateAliasing::default())
            .expect("rank-eight plan")
            .metadata();
        assert_eq!(meta.shape, [[1, 1, 1, 1], [1, 1, 2, 3]]);
        assert_eq!(meta.dispatch[0], 6);
        assert_eq!(meta.strides[0][1], [6, 6, 3, 1]);
    }
}
