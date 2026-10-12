use super::*;

#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
pub(crate) struct StreamParams {
    pub(crate) len: u32,
    pub(crate) factor: f32,
}

pub(crate) struct StreamScaleKernel;

impl KernelSource<Wgsl> for StreamScaleKernel {
    const ENTRY: &'static str = "main";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r#"
struct Params {
    len: u32,
    factor: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < params.len) {
        output[i] = input[i] * params.factor;
    }
}
"#,
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
pub(crate) struct GroupedParams {
    pub(crate) len: u32,
    pub(crate) addend: f32,
}

pub(crate) struct GroupedAddKernel;

impl GroupedKernelSource<Wgsl> for GroupedAddKernel {
    const ENTRY: &'static str = "main";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r#"
struct Params {
    len: u32,
    addend: f32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(1) @binding(0) var<storage, read> right: array<f32>;
@group(1) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < params.len) {
        output[i] = left[i] + right[i] + params.addend;
    }
}
"#,
        )
    }
}

/// Acquire a device, or `None` to skip (no Metal device).
pub(crate) fn device(test: &str) -> Option<MetalDevice> {
    match MetalDevice::try_default() {
        Ok(d) => Some(d),
        Err(e) => {
            if std::env::var_os("HEPHAESTUS_METAL_REQUIRE_DEVICE").is_some() {
                panic!("Metal device required for {test}: {e}");
            }
            eprintln!("skip {test}: Metal device unavailable ({e})");
            None
        }
    }
}

pub(crate) fn assert_elementwise_alias_rejected(result: Result<()>) {
    match result {
        Err(HephaestusError::DispatchFailed { message }) => {
            assert!(
                message.starts_with("output buffer must not alias "),
                "unexpected alias rejection message: {message}"
            );
        }
        other => panic!("expected elementwise alias rejection, got {other:?}"),
    }
}

pub(crate) fn assert_length_mismatch<T>(result: Result<T>, host_len: usize, device_len: usize) {
    match result {
        Err(HephaestusError::LengthMismatch {
            host_len: got_host,
            device_len: got_device,
        }) => {
            assert_eq!(got_host, host_len);
            assert_eq!(got_device, device_len);
        }
        Err(error) => panic!("expected length mismatch {host_len}->{device_len}, got {error:?}"),
        Ok(_) => panic!("expected length mismatch {host_len}->{device_len}, got success"),
    }
}
