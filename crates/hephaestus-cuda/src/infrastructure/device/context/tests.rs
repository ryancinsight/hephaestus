//! Real driver coverage for final-owner release and thread-current context routing.

use super::CurrentContext;
use crate::CudaDevice;
use hephaestus_core::{ComputeDevice, HephaestusError};

#[test]
fn sole_context_release_preserves_immediate_reacquisition() {
    let _subscriber = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_max_level(tracing::Level::TRACE)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    );
    let first = {
        let _span = tracing::info_span!("arrange").entered();
        match CudaDevice::try_default() {
            Ok(device) => device,
            Err(HephaestusError::AdapterUnavailable { .. })
                if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() =>
            {
                return;
            }
            Err(error) => panic!("CUDA reacquisition requires a working driver: {error}"),
        }
    };
    let second = {
        let _span = tracing::info_span!("act").entered();
        // Acquisition performs a real transfer probe. Dropping its final owner
        // exercises probe-buffer release followed by context destruction.
        drop(first);
        CudaDevice::try_default()
            .expect("invariant: final context release permits immediate reacquisition")
    };
    let _span = tracing::info_span!("assert").entered();
    let expected = [0x1234_5678_u32, u32::MAX, 0, 1 << 31];
    let buffer = second
        .upload(&expected)
        .expect("invariant: the reacquired context supports device allocation and upload");
    let mut actual = [0; 4];
    second
        .download(&buffer, &mut actual)
        .expect("invariant: the reacquired context supports device readback");
    assert_eq!(actual, expected);
}

#[test]
fn final_resource_release_preserves_another_current_context() {
    let first = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(HephaestusError::AdapterUnavailable { .. })
            if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() =>
        {
            return;
        }
        Err(error) => panic!("CUDA context lifetime requires a working driver: {error}"),
    };
    let first_buffer = first
        .upload(&[11_u32, 13, 17])
        .expect("first context allocation");
    drop(first);
    let second = CudaDevice::try_default().expect("second context acquisition");
    let second_buffer = second
        .upload(&[19_u32, 23, 29])
        .expect("second context allocation");
    second.bind().expect("bind second context");
    let expected = second.cuda_context().raw;
    drop(first_buffer);
    // Query before any operation can rebind and hide a destructor routing bug.
    let actual = CurrentContext::capture(second.driver()).expect("query retained current context");
    assert_eq!(
        actual.0, expected,
        "destroying the last first-context owner preserves the second context"
    );
    let mut output = [0_u32; 3];
    second
        .download(&second_buffer, &mut output)
        .expect("second-context transfer after first release");
    assert_eq!(output, [19, 23, 29]);

    // CUDA 13.3 cuda.h:6603–6612 defines cuCtxSetCurrent(NULL) as one
    // stack pop, not a full reset. A fresh thread establishes the empty-stack
    // case without discarding contexts from the calling thread's stack.
    let third = CudaDevice::try_default().expect("third context acquisition");
    let third_buffer = third.upload(&[31_u32]).expect("third context allocation");
    drop(third);
    let driver = second.driver();
    std::thread::spawn(move || {
        let before = CurrentContext::capture(driver).expect("query fresh thread context");
        assert_eq!(
            before.0,
            core::ptr::null_mut(),
            "fresh thread starts without a CUDA context"
        );
        drop(third_buffer);
        let after =
            CurrentContext::capture(driver).expect("query context after final owner release");
        assert_eq!(
            after.0,
            core::ptr::null_mut(),
            "final owner release preserves no-current state"
        );
    })
    .join()
    .expect("empty-stack final-owner regression");
    second.bind().expect("restore live second context");
}
