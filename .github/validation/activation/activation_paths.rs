//! Matched activation dispatch/completion timing; default mode is CPU-only.
//!
//! Device mode requires prior root authorization and independent harness review.
//! Correctness failures remain failures even when an old baseline is timed.

use hephaestus_core::{BlockWidth, UnaryExpr, UnaryValue, Wgsl};
use hephaestus_wgpu::{
    ComputeDevice, GeluGradOp, GeluOp, GeluTanhGradOp, WgpuDevice, unary_elementwise_into,
};
use std::time::{Duration, Instant};

const SIZES: [usize; 2] = [1 << 16, 1 << 20];
const SEED_LEN: u16 = 4096;
const WARMUPS: usize = 2;
const SAMPLES: usize = 9;
const BATCH: usize = 8;
const WAIT: Duration = Duration::from_secs(15);
const REVISION: &str = env!("ACTIVATION_SOURCE_REVISION");
const EXPRESSION_DIGEST: &str = env!("ACTIVATION_EXPRESSION_SHA256");
const HARNESS_DIGEST: &str = env!("ACTIVATION_HARNESS_SHA256");
const VALIDATION_HEAD: &str = env!("ACTIVATION_VALIDATION_HEAD");

const FAILURE_CLASSES: [&str; 5] = [
    "nan",
    "infinite",
    "zero",
    "finite_relative",
    "finite_absolute",
];

#[derive(Clone, Copy, Default)]
struct SeedFailures {
    count: [usize; 5],
    first_bits: [u32; 5],
    max_ulp: [f64; 5],
}

#[derive(Clone, Copy, Debug)]
enum Accuracy {
    Gelu,
    GeluGrad,
    GeluTanhGrad,
}

impl Accuracy {
    fn acceptance(self, x: f32, band: &str) -> (f64, Option<f64>) {
        if band == "finite-saturation" {
            return (
                if matches!(self, Self::GeluGrad) {
                    16.0
                } else {
                    32.0
                },
                None,
            );
        }
        match self {
            Self::Gelu => (
                if x < -7.0 {
                    256.0
                } else if x < -3.0 {
                    128.0
                } else {
                    32.0
                },
                None,
            ),
            Self::GeluGrad if x == -1.4 || x == -0.1 => (16.0, Some(3e-7)),
            Self::GeluGrad if (-1.4..=-0.1).contains(&x) => (f64::INFINITY, Some(3e-7)),
            Self::GeluTanhGrad if x == -1.35 || x == -0.15 => (32.0, Some(3e-7)),
            Self::GeluTanhGrad if (-1.35..=-0.15).contains(&x) => (f64::INFINITY, Some(3e-7)),
            Self::GeluGrad => (
                if x < -7.0 {
                    256.0
                } else if x < -3.0 {
                    128.0
                } else {
                    16.0
                },
                None,
            ),
            Self::GeluTanhGrad => (
                if x < -6.0 {
                    256.0
                } else if x < -2.0 {
                    128.0
                } else {
                    32.0
                },
                None,
            ),
        }
    }
}

struct Dataset {
    name: &'static str,
    seed: Vec<f32>,
}

fn ranged(name: &'static str, lo: f32, hi: f32, edges: &[f32]) -> Dataset {
    let mut seed: Vec<_> = (0..SEED_LEN)
        .map(|i| lo + (hi - lo) * (f32::from(i) / f32::from(SEED_LEN - 1)))
        .collect();
    seed.extend_from_slice(edges);
    Dataset { name, seed }
}

fn saturation() -> Dataset {
    let mut seed = Vec::new();
    for edge in [20.0_f32, 40.0_f32] {
        seed.extend([
            -edge.next_up(),
            -edge,
            -edge.next_down(),
            edge.next_down(),
            edge,
            edge.next_up(),
        ]);
    }
    seed.extend([-1e20, 1e20, -1e30, 1e30, -f32::MAX, f32::MAX]);
    Dataset {
        name: "finite-saturation",
        seed,
    }
}

fn datasets(accuracy: Accuracy) -> Vec<Dataset> {
    let ordinary = ranged(
        "ordinary",
        -3.0,
        4.0,
        &[
            -3.0, -2.0, -1.4, -1.35, -0.7525, -0.15, -0.1, -0.0, 0.0, 1.42, 4.0,
        ],
    );
    let tail = match accuracy {
        Accuracy::Gelu => ranged(
            "tail",
            f32::from_bits(0xc152_5706),
            -7.0,
            &[
                -12.6,
                -13.0,
                -13.1,
                f32::from_bits(0xc152_5705),
                f32::from_bits(0xc152_5706),
            ],
        ),
        Accuracy::GeluGrad => ranged(
            "tail",
            f32::from_bits(0xc155_7859),
            -7.0,
            &[
                -12.6,
                -13.0,
                -13.1,
                -13.3,
                f32::from_bits(0xc155_7858),
                f32::from_bits(0xc155_7859),
            ],
        ),
        Accuracy::GeluTanhGrad => ranged("tail", -9.95, -6.0, &[-6.0, -9.9, -9.95]),
    };
    vec![ordinary, tail, saturation()]
}

fn ulps(got: f32, reference: f64) -> f64 {
    if reference.is_nan() {
        return if got.is_nan() { 0.0 } else { f64::INFINITY };
    }
    if reference.is_infinite() {
        return if f64::from(got) == reference {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if !got.is_finite() {
        return f64::INFINITY;
    }
    let spacing = if reference.abs() < f64::from(f32::MIN_POSITIVE) {
        2_f64.powi(-149)
    } else {
        let exponent = i32::try_from((reference.abs().to_bits() >> 52) & 0x7ff)
            .expect("binary64 exponent fits i32");
        2_f64.powi(exponent - 1023 - 23)
    };
    (f64::from(got) - reference).abs() / spacing
}

fn check(
    values: &[f32],
    dataset: &Dataset,
    references: &[f64],
    accuracy: Accuracy,
    op: &str,
    phase: &str,
) -> usize {
    let mut failures = 0_usize;
    let mut subnormal_reference = 0_usize;
    let mut max_ulp = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut checksum = 0_u64;
    let mut details = vec![SeedFailures::default(); dataset.seed.len()];
    for (i, &got) in values.iter().enumerate() {
        let seed_index = i % dataset.seed.len();
        let x = dataset.seed[seed_index];
        let expected = references[seed_index];
        checksum = checksum.wrapping_add(u64::from(got.to_bits()));
        let error = ulps(got, expected);
        let abs_error = (f64::from(got) - expected).abs();
        let (bound, absolute) = accuracy.acceptance(x, dataset.name);
        let selected = expected == 0.0 || expected.abs() >= f64::from(f32::MIN_POSITIVE);
        if !selected {
            subnormal_reference += 1;
        }
        // Existing WGSL sweeps exclude subnormal true results; a finite-input
        // NaN is nevertheless always a failure, including in those controls.
        let failed = !got.is_finite()
            || (selected && (error > bound || absolute.is_some_and(|b| abs_error > b)));
        if selected {
            max_ulp = max_ulp.max(error);
            max_abs = max_abs.max(abs_error);
        }
        if failed {
            let kind = if got.is_nan() {
                0
            } else if got.is_infinite() {
                1
            } else if got == 0.0 {
                2
            } else if absolute.is_some_and(|b| abs_error > b) {
                4
            } else {
                3
            };
            let detail = &mut details[seed_index];
            if detail.count[kind] == 0 {
                detail.first_bits[kind] = got.to_bits();
            }
            detail.count[kind] += 1;
            detail.max_ulp[kind] = detail.max_ulp[kind].max(error);
            if failures == 0 {
                println!(
                    "FIRST_FAILURE\t{op}\t{}\t{}\t{phase}\t{x:?}\t0x{:08x}\t{got:?}\t{expected:?}\t{error:?}",
                    dataset.name,
                    values.len(),
                    x.to_bits()
                );
            }
            failures += 1;
        }
    }
    for (seed_index, detail) in details.iter().enumerate() {
        for (kind, class) in FAILURE_CLASSES.iter().enumerate() {
            if detail.count[kind] != 0 {
                println!(
                    "FAILURE_DETAIL\t{op}\t{}\t{}\t{phase}\t{}\t{class}\t{}\t{}\t{}\t{:?}",
                    dataset.name,
                    values.len(),
                    dataset.seed[seed_index].to_bits(),
                    detail.count[kind],
                    detail.first_bits[kind],
                    references[seed_index].to_bits(),
                    detail.max_ulp[kind]
                );
            }
        }
    }
    println!(
        "CHECK\t{op}\t{}\t{}\t{phase}\t{failures}\t{max_ulp:?}\t{max_abs:?}\t{subnormal_reference}\t{checksum}",
        dataset.name,
        values.len()
    );
    failures
}

fn wait(device: &WgpuDevice) {
    device
        .inner()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(WAIT),
        })
        .expect("activation benchmark completion within fifteen seconds");
}

fn run_operation<Op: UnaryExpr<Wgsl> + UnaryValue>(
    device: &WgpuDevice,
    op: &str,
    accuracy: Accuracy,
) -> usize {
    let mut failures = 0;
    for dataset in datasets(accuracy) {
        let references: Vec<_> = dataset
            .seed
            .iter()
            .map(|&x| Op::apply::<f64>(f64::from(x)))
            .collect();
        for len in SIZES {
            let host: Vec<_> = dataset.seed.iter().copied().cycle().take(len).collect();
            let input = device
                .upload(&host)
                .expect("activation benchmark input upload");
            let output = device
                .alloc_zeroed::<f32>(len)
                .expect("activation benchmark output allocation");
            for _ in 0..WARMUPS {
                unary_elementwise_into::<Op, f32>(device, &input, &output, BlockWidth::DEFAULT)
                    .expect("activation benchmark warm dispatch");
                wait(device);
            }
            let mut downloaded = vec![0.0; len];
            device
                .download(&output, &mut downloaded)
                .expect("activation benchmark precheck download");
            failures += check(&downloaded, &dataset, &references, accuracy, op, "before");
            for sample in 0..SAMPLES {
                wait(device);
                let start = Instant::now();
                for _ in 0..BATCH {
                    unary_elementwise_into::<Op, f32>(device, &input, &output, BlockWidth::DEFAULT)
                        .expect("activation benchmark measured dispatch");
                }
                wait(device);
                let elapsed_ns = start.elapsed().as_nanos();
                println!(
                    "TIMING\t{op}\t{}\t{len}\t{sample}\t{BATCH}\t{elapsed_ns}",
                    dataset.name
                );
            }
            device
                .download(&output, &mut downloaded)
                .expect("activation benchmark postcheck download");
            failures += check(&downloaded, &dataset, &references, accuracy, op, "after");
        }
    }
    failures
}

fn manifest<Op: UnaryExpr<Wgsl> + UnaryValue>(op: &str, accuracy: Accuracy, include_inputs: bool) {
    println!("EXPR\t{op}\t{:?}", Op::EXPR);
    for dataset in datasets(accuracy) {
        let mut input_checksum = 0_u64;
        let mut reference_checksum = 0_u64;
        for (index, &x) in dataset.seed.iter().enumerate() {
            input_checksum = input_checksum.wrapping_add(u64::from(x.to_bits()));
            let reference = Op::apply::<f64>(f64::from(x));
            reference_checksum = reference_checksum.wrapping_add(reference.to_bits());
            if include_inputs {
                let (bound, absolute) = accuracy.acceptance(x, dataset.name);
                println!(
                    "INPUT\t{op}\t{}\t{index}\t{}\t{}\t{bound:?}\t{}",
                    dataset.name,
                    x.to_bits(),
                    reference.to_bits(),
                    absolute.map_or_else(|| "none".to_owned(), |v| v.to_string())
                );
            }
        }
        println!(
            "FIXTURE\t{op}\t{}\t{}\t{input_checksum}\t{reference_checksum}",
            dataset.name,
            dataset.seed.len()
        );
    }
}

fn main() {
    let run_device = std::env::args().any(|argument| argument == "--run-device");
    assert_eq!(
        Accuracy::Gelu.acceptance(20.0, "finite-saturation"),
        (32.0, None)
    );
    assert_eq!(
        Accuracy::GeluGrad.acceptance(20.0, "finite-saturation"),
        (16.0, None)
    );
    assert_eq!(
        Accuracy::GeluTanhGrad.acceptance(-40.0, "finite-saturation"),
        (32.0, None)
    );
    assert_eq!(
        Accuracy::GeluTanhGrad.acceptance(-3.0, "ordinary"),
        (128.0, None)
    );
    assert_eq!(
        Accuracy::GeluGrad.acceptance(-1.4, "ordinary"),
        (16.0, Some(3e-7))
    );
    assert_eq!(
        Accuracy::GeluTanhGrad.acceptance(-0.15, "ordinary"),
        (32.0, Some(3e-7))
    );
    assert_eq!(ulps(1.0, 1.0), 0.0);
    assert_eq!(ulps(1.0_f32.next_up(), 1.0), 1.0);
    assert_eq!(ulps(f32::NAN, 1.0), f64::INFINITY);
    assert_eq!(ulps(0.0, f64::from(f32::MIN_POSITIVE)), 8388608.0);
    println!("META\trevision\t{REVISION}");
    println!("META\texpression_sha256\t{EXPRESSION_DIGEST}");
    println!("META\tharness_sha256\t{HARNESS_DIGEST}");
    println!("META\tvalidation_head\t{VALIDATION_HEAD}");
    println!(
        "META\ttiming_scope\tpublic dispatch plus device completion; no upload/readback/compilation"
    );
    println!("META\tsizes\t{SIZES:?}");
    println!("META\twarmups_samples_batch\t{WARMUPS},{SAMPLES},{BATCH}");
    manifest::<GeluOp>("GeluOp", Accuracy::Gelu, !run_device);
    manifest::<GeluGradOp>("GeluGradOp", Accuracy::GeluGrad, !run_device);
    manifest::<GeluTanhGradOp>("GeluTanhGradOp", Accuracy::GeluTanhGrad, !run_device);
    if !run_device {
        println!("META\tstatus\tCPU_MANIFEST_ONLY_NO_DEVICE_ACQUISITION");
        return;
    }
    assert!(
        std::env::var("GITHUB_ACTIONS").as_deref() == Ok("true")
            && std::env::var("GITHUB_EVENT_NAME").as_deref() == Ok("pull_request")
            && std::env::consts::OS == "macos"
            && std::env::var("HEPHAESTUS_HOSTED_ACTIVATION_INTENT").as_deref()
                == Ok("reviewed-metal-validation"),
        "device execution is restricted to the reviewed automatic hosted PR context"
    );
    assert_eq!(
        std::env::var("ACTIVATION_DEVICE_AUTHORIZATION").as_deref(),
        Ok("root-approved-metal-only"),
        "root device go and independent harness review are required"
    );
    let acquisition = Instant::now();
    let device = match WgpuDevice::try_metal("hephaestus-activation-performance") {
        Ok(device) => device,
        Err(error) => {
            eprintln!(
                "ACQUISITION_FAILURE\t{:?}\t{:?}",
                acquisition.elapsed(),
                error
            );
            std::process::exit(2);
        }
    };
    println!("META\tadapter\t{:?}", device.adapter_info());
    println!("META\tfeatures\t{:?}", device.features());
    println!("META\tlimits\t{:?}", device.limits());
    println!("META\tdevice_limits\t{:?}", device.device_limits());
    assert!(
        matches!(
            device
                .adapter_info()
                .map(|info| (info.backend, info.device_type)),
            Some((
                wgpu::Backend::Metal,
                wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu
            ))
        ),
        "only an identified physical Metal adapter is accepted; no alternate route"
    );
    let mut failures = run_operation::<GeluOp>(&device, "GeluOp", Accuracy::Gelu);
    failures += run_operation::<GeluGradOp>(&device, "GeluGradOp", Accuracy::GeluGrad);
    failures += run_operation::<GeluTanhGradOp>(&device, "GeluTanhGradOp", Accuracy::GeluTanhGrad);
    println!("FINAL\tcomputed_value_failures\t{failures}");
    if failures != 0 {
        std::process::exit(1);
    }
}
