use hephaestus_core::{
    AttentionForwardOperands, AttentionMask, AttentionOps, ComputeDevice, HephaestusError,
    StridedView,
};
use leto::Layout;

use super::device_or_skip;
use crate::WgpuAttentionOps;

#[test]
fn aliasing_is_rejected_before_any_output_mutation() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let layout = Layout::c_contiguous([1, 1, 2]).expect("tensor layout");
    let weights_layout = Layout::c_contiguous([1, 1, 1]).expect("weights layout");
    let original = [2.0_f32, -3.0];
    let shared = device.upload(&original).expect("shared upload");
    let key = device.upload(&original).expect("key upload");
    let value = device.upload(&original).expect("value upload");
    let weights = device.alloc_zeroed::<f32>(1).expect("weights allocation");
    let error = match WgpuAttentionOps.prepare_attention_forward(
        &device,
        AttentionForwardOperands {
            query: StridedView::new(&shared, &layout),
            key: StridedView::new(&key, &layout),
            value: StridedView::new(&value, &layout),
            mask: AttentionMask::unrestricted(),
            scale: 1.0,
            output: StridedView::new(&shared, &layout),
            weights: StridedView::new(&weights, &weights_layout),
        },
    ) {
        Ok(_) => panic!("aliased output must be rejected"),
        Err(error) => error,
    };
    match error {
        HephaestusError::InvalidConfiguration { message } => assert_eq!(
            message,
            "attention writable buffers must not alias readable operands or each other"
        ),
        other => panic!("expected typed invalid configuration, got {other:?}"),
    }
    let mut actual = [0.0_f32; 2];
    device
        .download(&shared, &mut actual)
        .expect("shared download");
    assert_eq!(actual, original);
}
