//! Typed WGPU kernel descriptors for dense split-complex FFT execution.

use std::{borrow::Cow, marker::PhantomData};

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{BindingDecl, KernelInterface, KernelSource, Wgsl};

pub(crate) const WORKGROUP_SIZE: u32 = 256;
pub(crate) const FUSED_WORKGROUP_SIZE: u32 = 64;
pub(crate) const FUSED_MAX_LENGTH: usize = 1024;
pub(crate) const FUSED_WORKGROUP_STORAGE_BYTES: u32 = 12 * 1024;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct FftParams {
    pub(crate) n: u32,
    pub(crate) stage: u32,
    pub(crate) inverse: u32,
    pub(crate) batch_count: u32,
}

const _: () = assert!(core::mem::size_of::<FftParams>() == 16);

pub(crate) trait FftEntry {
    const LABEL: &'static str;
    const ENTRY: &'static str;
}

pub(crate) struct FftKernel<E>(PhantomData<E>);

impl<E> FftKernel<E> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<E: FftEntry> KernelInterface for FftKernel<E> {
    type Params = FftParams;
    const LABEL: &'static str = E::LABEL;
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE, 1, 1];
}

impl<E: FftEntry> KernelSource<Wgsl> for FftKernel<E> {
    const ENTRY: &'static str = E::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(include_str!("shader/fft.wgsl"))
    }
}

pub(crate) struct BitReverse;
pub(crate) struct RadixFourBitReverse;
pub(crate) struct Butterfly;
pub(crate) struct RadixFourButterfly;
pub(crate) struct Scale;

impl FftEntry for BitReverse {
    const LABEL: &'static str = "hephaestus-fft-bit-reverse";
    const ENTRY: &'static str = "fft_bitrev";
}
impl FftEntry for RadixFourBitReverse {
    const LABEL: &'static str = "hephaestus-fft-radix-four-bit-reverse";
    const ENTRY: &'static str = "fft_bitrev_radix4";
}
impl FftEntry for Butterfly {
    const LABEL: &'static str = "hephaestus-fft-butterfly";
    const ENTRY: &'static str = "fft_forward";
}
impl FftEntry for RadixFourButterfly {
    const LABEL: &'static str = "hephaestus-fft-radix-four-butterfly";
    const ENTRY: &'static str = "fft_forward_radix4";
}
impl FftEntry for Scale {
    const LABEL: &'static str = "hephaestus-fft-scale";
    const ENTRY: &'static str = "fft_scale";
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct PackParams {
    pub(crate) n: u32,
    pub(crate) stage: u32,
    pub(crate) inverse: u32,
    pub(crate) batch_count: u32,
    pub(crate) nx: u32,
    pub(crate) ny: u32,
    pub(crate) nz: u32,
    pub(crate) axis: u32,
    pub(crate) fft_len: u32,
    pub(crate) padding: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<PackParams>() == 48);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct FusedParams {
    pub(crate) n: u32,
    pub(crate) log2n: u32,
    pub(crate) inverse: u32,
    pub(crate) batch_count: u32,
    pub(crate) nx: u32,
    pub(crate) ny: u32,
    pub(crate) nz: u32,
    pub(crate) axis: u32,
    pub(crate) batch_grid_x: u32,
    pub(crate) padding: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<FusedParams>() == 48);

pub(crate) struct FusedKernel;

impl KernelInterface for FusedKernel {
    type Params = FusedParams;
    const LABEL: &'static str = "hephaestus-fft-fused-radix";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_only::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [FUSED_WORKGROUP_SIZE, 1, 1];
}

impl KernelSource<Wgsl> for FusedKernel {
    const ENTRY: &'static str = "fft_fused_axis";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(include_str!("shader/fused.wgsl"))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ChirpParams {
    pub(crate) n: u32,
    pub(crate) m: u32,
    pub(crate) batch_count: u32,
    pub(crate) padding: u32,
}

const _: () = assert!(core::mem::size_of::<ChirpParams>() == 16);

pub(crate) trait PackEntry {
    const LABEL: &'static str;
    const ENTRY: &'static str;
}

pub(crate) struct PackKernel<E>(PhantomData<E>);

impl<E> PackKernel<E> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<E: PackEntry> KernelInterface for PackKernel<E> {
    type Params = PackParams;
    const LABEL: &'static str = E::LABEL;
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE, 1, 1];
}

impl<E: PackEntry> KernelSource<Wgsl> for PackKernel<E> {
    const ENTRY: &'static str = E::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(include_str!("shader/pack.wgsl"))
    }
}

pub(crate) struct Pack;
pub(crate) struct Unpack;

impl PackEntry for Pack {
    const LABEL: &'static str = "hephaestus-fft-pack";
    const ENTRY: &'static str = "fft_pack_axis";
}
impl PackEntry for Unpack {
    const LABEL: &'static str = "hephaestus-fft-unpack";
    const ENTRY: &'static str = "fft_unpack_axis";
}

pub(crate) trait ChirpEntry {
    const LABEL: &'static str;
    const ENTRY: &'static str;
}

pub(crate) struct ChirpKernel<E>(PhantomData<E>);

impl<E> ChirpKernel<E> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<E: ChirpEntry> KernelInterface for ChirpKernel<E> {
    type Params = ChirpParams;
    const LABEL: &'static str = E::LABEL;
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_only::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE, 1, 1];
}

impl<E: ChirpEntry> KernelSource<Wgsl> for ChirpKernel<E> {
    const ENTRY: &'static str = E::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(include_str!("shader/chirp.wgsl"))
    }
}

pub(crate) struct ChirpPremultiply;
pub(crate) struct ChirpPointMultiply;
pub(crate) struct ChirpScale;
pub(crate) struct ChirpPostmultiply;
pub(crate) struct ChirpNegateImaginary;

impl ChirpEntry for ChirpPremultiply {
    const LABEL: &'static str = "hephaestus-fft-chirp-premultiply";
    const ENTRY: &'static str = "chirp_premul";
}
impl ChirpEntry for ChirpPointMultiply {
    const LABEL: &'static str = "hephaestus-fft-chirp-point-multiply";
    const ENTRY: &'static str = "chirp_pointmul";
}
impl ChirpEntry for ChirpScale {
    const LABEL: &'static str = "hephaestus-fft-chirp-scale";
    const ENTRY: &'static str = "chirp_scale";
}
impl ChirpEntry for ChirpPostmultiply {
    const LABEL: &'static str = "hephaestus-fft-chirp-postmultiply";
    const ENTRY: &'static str = "chirp_postmul";
}
impl ChirpEntry for ChirpNegateImaginary {
    const LABEL: &'static str = "hephaestus-fft-chirp-negate-imaginary";
    const ENTRY: &'static str = "chirp_negate_im";
}
