use super::*;

pub(super) fn prepared_fft_owns_operands_and_encodes_in_existing_pass() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let shape = [2, 2, 2];
    let real_host = [0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0];
    let imaginary_host = [0.0; 8];
    let expected = direct_forward(shape, &real_host, &imaginary_host);
    let real = device.upload(&real_host).expect("invariant: real upload");
    let imaginary = device
        .upload(&imaginary_host)
        .expect("invariant: imaginary upload");
    let retained_real = real.clone();
    let retained_imaginary = imaginary.clone();
    let layout = Layout::c_contiguous(shape).expect("invariant: dense layout");
    let ops = WgpuFftOps;
    let prepared = prepare(
        &ops,
        &device,
        &real,
        &imaginary,
        &layout,
        FftDirection::Forward,
    );

    drop(real);
    drop(imaginary);

    let mut stream = device.stream().expect("invariant: command stream");
    stream
        .encode_grouped_sequence("hephaestus-fft-consumer-pass", |sequence| {
            sequence
                .raw_pass_mut()
                .insert_debug_marker("consumer-before-fft");
            prepared.encode_in_sequence(sequence)?;
            sequence
                .raw_pass_mut()
                .insert_debug_marker("consumer-after-fft");
            Ok(())
        })
        .expect("invariant: pass device owns the prepared FFT");
    stream
        .submit_with_timeout(std::time::Duration::from_secs(10))
        .expect("invariant: consumer pass submission completes");

    assert_complex_close(
        &download(&device, &retained_real),
        &download(&device, &retained_imaginary),
        &expected,
        shape.into_iter().product(),
    );
}

pub(super) fn prepared_axis_one_fft_matches_independent_row_oracles() {
    let Some(device) = device_or_skip() else {
        return;
    };
    for columns in [8, 5] {
        let rows = 3;
        let elements = rows * columns;
        let real_input = (0..elements)
            .map(|index| {
                let row = index / columns;
                let column = index % columns;
                0.25 + row as f32 * 1.75 + column as f32 * 0.125
            })
            .collect::<Vec<_>>();
        let imaginary_input = (0..elements)
            .map(|index| {
                let row = index / columns;
                let column = index % columns;
                -0.5 + row as f32 * 0.375 - column as f32 * 0.0625
            })
            .collect::<Vec<_>>();
        let real = device.upload(&real_input).expect("real upload");
        let imaginary = device.upload(&imaginary_input).expect("imaginary upload");
        let layout = Layout::c_contiguous([rows, columns]).expect("dense row layout");
        let ops = WgpuFftOps;
        let forward = prepare_axes(
            &ops,
            &device,
            &real,
            &imaginary,
            &layout,
            FftDirection::Forward,
            &[1],
        );
        let inverse = prepare_axes(
            &ops,
            &device,
            &real,
            &imaginary,
            &layout,
            FftDirection::Inverse,
            &[1],
        );
        let expected_forward = direct_axis_one(
            rows,
            columns,
            &real_input,
            &imaginary_input,
            FftDirection::Forward,
        );
        ops.dispatch_fft(&device, &forward)
            .expect("selected-axis forward dispatch");
        assert_complex_close(
            &download(&device, &real),
            &download(&device, &imaginary),
            &expected_forward,
            columns,
        );

        let real_spectrum = (0..elements)
            .map(|index| 0.75 - index as f32 * 0.03125)
            .collect::<Vec<_>>();
        let imaginary_spectrum = (0..elements)
            .map(|index| -0.25 + index as f32 * 0.046875)
            .collect::<Vec<_>>();
        device
            .write_sub_buffer(&real, 0, &real_spectrum)
            .expect("real spectrum upload");
        device
            .write_sub_buffer(&imaginary, 0, &imaginary_spectrum)
            .expect("imaginary spectrum upload");
        let expected_inverse = direct_axis_one(
            rows,
            columns,
            &real_spectrum,
            &imaginary_spectrum,
            FftDirection::Inverse,
        );
        ops.dispatch_fft(&device, &inverse)
            .expect("selected-axis inverse dispatch");
        assert_complex_close(
            &download(&device, &real),
            &download(&device, &imaginary),
            &expected_inverse,
            columns,
        );
    }
}

pub(super) fn selected_axis_validation_precedes_operand_mutation() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let real_input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let imaginary_input = [-1.0_f32, -2.0, -3.0, -4.0, -5.0, -6.0];
    let real = device.upload(&real_input).expect("real upload");
    let imaginary = device.upload(&imaginary_input).expect("imaginary upload");
    let layout = Layout::c_contiguous([2, 3]).expect("dense row layout");
    let ops = WgpuFftOps;
    for (axes, expected) in [
        (&[][..], "nonempty"),
        (&[1, 1][..], "duplicated"),
        (&[2][..], "out of range"),
    ] {
        let error = match ops.prepare_fft_axes(
            &device,
            operands(&real, &imaginary, &layout),
            FftDirection::Forward,
            axes,
        ) {
            Ok(_) => panic!("invalid selected axes must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains(expected));
        assert_eq!(download(&device, &real), real_input);
        assert_eq!(download(&device, &imaginary), imaginary_input);
    }
}
