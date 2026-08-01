//! Stateful parameter updates through WGPU's native Metal path.

use hephaestus_core::{StatefulUpdateOperands, StatefulUpdateOps, StatefulUpdateRule, Wgsl};
use hephaestus_wgpu::{WgpuDevice, WgpuStatefulUpdateOps};

use crate::{MetalBuffer, MetalDevice, Result};

/// Provider-owned stateful-update implementation for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalStatefulUpdateOps;

impl StatefulUpdateOps<MetalDevice> for MetalStatefulUpdateOps {
    type Dialect = Wgsl;

    fn stateful_update<Rule, const N: usize>(
        &self,
        device: &MetalDevice,
        operands: StatefulUpdateOperands<'_, MetalBuffer<f32>, N>,
        parameters: <Rule as StatefulUpdateRule<Self::Dialect>>::Parameters,
    ) -> Result<()>
    where
        Rule: StatefulUpdateRule<Self::Dialect>,
    {
        let parameter = hephaestus_core::StridedView::new(
            &operands.parameter.buffer.inner,
            operands.parameter.layout,
        );
        let gradient = hephaestus_core::StridedView::new(
            &operands.gradient.buffer.inner,
            operands.gradient.layout,
        );
        match operands.states {
            [state_zero] => {
                let states = [hephaestus_core::StridedView::new(
                    &state_zero.buffer.inner,
                    state_zero.layout,
                )];
                <WgpuStatefulUpdateOps as StatefulUpdateOps<WgpuDevice>>::stateful_update::<Rule, N>(
                    &WgpuStatefulUpdateOps,
                    device.wgpu_device(),
                    StatefulUpdateOperands {
                        parameter,
                        gradient,
                        states: &states,
                    },
                    parameters,
                )
            }
            [state_zero, state_one] => {
                let states = [
                    hephaestus_core::StridedView::new(&state_zero.buffer.inner, state_zero.layout),
                    hephaestus_core::StridedView::new(&state_one.buffer.inner, state_one.layout),
                ];
                <WgpuStatefulUpdateOps as StatefulUpdateOps<WgpuDevice>>::stateful_update::<Rule, N>(
                    &WgpuStatefulUpdateOps,
                    device.wgpu_device(),
                    StatefulUpdateOperands {
                        parameter,
                        gradient,
                        states: &states,
                    },
                    parameters,
                )
            }
            _ => Err(hephaestus_core::HephaestusError::InvalidConfiguration {
                message: format!(
                    "stateful update requires {} state views, got {}",
                    Rule::STATE_COUNT,
                    operands.states.len()
                ),
            }),
        }
    }
}
