//! WGPU instantiation of the shared transfer conformance clauses.

use hephaestus_conformance::assert_transfer_contract;
pub(super) fn wgpu_satisfies_the_transfer_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_transfer_contract(&device);
}

/// Sixteen readers exceed the eight retained completion slots. Each round
/// changes the values, crosses distinct byte extents, and exercises full,
/// owned, and subrange readback on the same process-cached device used by the
/// other parallel contracts. No device-wide test lock serializes this work.
#[test]
fn concurrent_readbacks_deliver_callbacks_and_preserve_values() {
    use hephaestus_core::ComputeDevice;
    use std::sync::Barrier;
    use tracing_subscriber::fmt::format::FmtSpan;

    const READERS: usize = 16;
    const ROUNDS: usize = 64;
    const LENGTHS: [usize; 4] = [1, 3, 257, 4097];

    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(FmtSpan::FULL)
        .finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let dispatch = tracing::dispatcher::get_default(Clone::clone);
    let Some(device) = super::device_or_skip() else {
        return;
    };
    let buffers = {
        let _arrange = tracing::info_span!("arrange").entered();
        (0..READERS)
            .map(|reader| {
                let len = LENGTHS[reader % LENGTHS.len()];
                let words = (1..=len)
                    .map(|index| u32::try_from(index).expect("bounded word index"))
                    .collect::<Vec<_>>();
                let halves = (1..=259_u16).collect::<Vec<_>>();
                (
                    device.upload(&words).expect("word upload"),
                    device.upload(&halves).expect("odd packed upload"),
                    len,
                )
            })
            .collect::<Vec<_>>()
    };
    let start = Barrier::new(READERS);
    std::thread::scope(|scope| {
        let threads = buffers
            .iter()
            .enumerate()
            .map(|(reader, (buffer, packed, len))| {
                let device = &device;
                let start = &start;
                let dispatch = dispatch.clone();
                scope.spawn(move || {
                    let _subscriber = tracing::dispatcher::set_default(&dispatch);
                    let _reader = tracing::info_span!("reader", reader).entered();
                    start.wait();
                    for round in 0..ROUNDS {
                        let _round = tracing::info_span!("round", round, len).entered();
                        // Tuple encoding is injective over these bounds:
                        // r < 16, k < 64, i < 4097. Every readback has a
                        // distinct generation and every element an exact oracle.
                        let generation = reader * ROUNDS + round;
                        let expected = (0..*len)
                            .map(|index| {
                                u32::try_from(generation * LENGTHS[3] + index + 1)
                                    .expect("bounded generation and index")
                            })
                            .collect::<Vec<_>>();
                        let key = u16::try_from(generation + 1)
                            .expect("bounded packed generation")
                            .rotate_left(8);
                        let packed_expected =
                            (1..=259_u16).map(|index| key ^ index).collect::<Vec<_>>();
                        let (actual, packed_actual) = {
                            let _act = tracing::info_span!("act").entered();
                            device.write_buffer(buffer, &expected).expect("word update");
                            device
                                .write_buffer(packed, &packed_expected)
                                .expect("packed update");
                            let actual = match round % 3 {
                                0 => {
                                    let mut actual = vec![u32::MAX; *len];
                                    device.download(buffer, &mut actual).expect("full readback");
                                    actual
                                }
                                1 => device.download_owned(buffer).expect("owned readback"),
                                _ => {
                                    let offset = *len / 2;
                                    let mut actual = vec![u32::MAX; *len - offset];
                                    device
                                        .download_sub_buffer(buffer, offset, &mut actual)
                                        .expect("subrange readback");
                                    actual
                                }
                            };
                            // Offset two preserves the four-byte copy alignment;
                            // 257 halves require padding in the staging allocation.
                            let mut packed_actual = vec![u16::MAX; 257];
                            device
                                .download_sub_buffer(packed, 2, &mut packed_actual)
                                .expect("odd packed readback");
                            (actual, packed_actual)
                        };
                        let _assert = tracing::info_span!("assert").entered();
                        let offset = if round % 3 == 2 { *len / 2 } else { 0 };
                        assert_eq!(actual, expected[offset..]);
                        assert_eq!(packed_actual, packed_expected[2..]);
                    }
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().expect("concurrent readback thread");
        }
    });
}
