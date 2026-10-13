use super::to_u32;
use hephaestus_core::HephaestusError;

#[test]
fn device_extent_preserves_boundary_words_and_rejects_overflow() {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(
            tracing_subscriber::fmt::format::FmtSpan::ENTER
                | tracing_subscriber::fmt::format::FmtSpan::CLOSE,
        )
        .finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let arrange = tracing::info_span!("arrange").entered();
    // CUDA grid.z boundary and the full unsigned kernel length/index domain.
    let words = [0_u32, 1, 65_535, 65_536, u32::MAX];
    drop(arrange);
    for word in words {
        let act = tracing::info_span!("act", word).entered();
        let extent = usize::try_from(word).expect("supported host index width");
        let got = to_u32(extent, "device extent").expect("representable extent");
        drop(act);
        let _assert = tracing::info_span!("assert", word).entered();
        assert_eq!(got, word);
    }
    #[cfg(target_pointer_width = "64")]
    {
        let extent = usize::try_from(u32::MAX).expect("wide host") + 1;
        for context in ["batched matmul batch count", "reduction length"] {
            let act = tracing::info_span!("act", context, extent).entered();
            let error = to_u32(extent, context).expect_err("unrepresentable word");
            drop(act);
            let _assert = tracing::info_span!("assert", context).entered();
            match error {
                HephaestusError::DispatchFailed { message } => {
                    assert_eq!(message, format!("{context} {extent} exceeds u32 range"));
                }
                other => panic!("wrong extent failure: {other:?}"),
            }
        }
    }
}
