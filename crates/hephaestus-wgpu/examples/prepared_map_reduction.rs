//! Reuse fixed WGPU resources for repeated dot and L2-norm dispatch.

use hephaestus_wgpu::{
    ComputeDevice, HephaestusError, StridedOperand, WgpuDevice, prepare_dot, prepare_norm_l2,
};
use leto::Layout;

fn main() -> hephaestus_wgpu::Result<()> {
    let device = match WgpuDevice::try_default("prepared-map-reduction-example") {
        Ok(device) => device,
        Err(HephaestusError::AdapterUnavailable { .. }) => {
            println!("no WGPU adapter available; example skipped");
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let layout = Layout::c_contiguous([4]).expect("invariant: fixed shape is valid");
    let lhs = device.upload(&[1.0f32, 2.0, 3.0, 4.0])?;
    let rhs = device.upload(&[5.0f32, 6.0, 7.0, 8.0])?;
    let lhs_view = StridedOperand {
        buffer: &lhs,
        layout: &layout,
    };
    let rhs_view = StridedOperand {
        buffer: &rhs,
        layout: &layout,
    };

    let dot = prepare_dot(&device, lhs_view, rhs_view)?;
    let norm = prepare_norm_l2(&device, lhs_view)?;
    dot.dispatch(&device)?;
    norm.dispatch(&device)?;

    let mut dot_value = [0.0f32; 1];
    let mut norm_value = [0.0f32; 1];
    device.download(dot.output(), &mut dot_value)?;
    device.download(norm.output(), &mut norm_value)?;
    println!("dot={} l2={}", dot_value[0], norm_value[0]);
    Ok(())
}
