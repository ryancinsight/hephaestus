//! Required-device scalar matrix for the CUDA elementwise generators.
//!
//! Every generator below is bound by `T: DialectScalar<CudaC>`, which admits
//! `F16`/`Bf16` — so each must compile those scalars, not just the machine
//! types. Fixtures use small integers and quarters, exactly representable in
//! both 16-bit formats, so every comparison is an exact equality.

#![cfg(feature = "cuda")]

use eunomia::{Bf16, F16, Pod};
use hephaestus_core::{
    AddOp, BlockWidth, ComputeDevice, CudaC, DialectScalar, DivOp, MulOp, NegOp, SqrtOp, SubOp,
};
use hephaestus_cuda::{
    CudaDevice, StridedOperand, binary_elementwise_into, binary_elementwise_strided_into,
    scalar_elementwise_into, scalar_elementwise_strided_into, unary_elementwise_into,
    unary_elementwise_strided_into,
};
use leto::Layout;

fn device_or_skip() -> Option<CudaDevice> {
    match CudaDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA elementwise scalar matrix: device unavailable ({error})");
            None
        }
        Err(error) => panic!("CUDA elementwise scalar matrix requires a physical device: {error}"),
    }
}

fn f16(values: &[f32]) -> Vec<F16> {
    values.iter().map(|&v| F16::from_f32(v)).collect()
}

fn bf16(values: &[f32]) -> Vec<Bf16> {
    values.iter().map(|&v| Bf16::from_f32(v)).collect()
}

fn assert_binary_add<T>(device: &CudaDevice)
where
    T: DialectScalar<CudaC> + Pod + PartialEq + Copy + FromHostTrio,
{
    let lhs_h = [1.0f32, 2.0, 3.0, 4.0];
    let rhs_h = [4.0f32, 3.0, 2.0, 1.0];
    let expected_h = [5.0f32, 5.0, 5.0, 5.0];
    let (lhs, rhs, expected) = T::from_host_trio(&lhs_h, &rhs_h, &expected_h);
    let lhs_buf = device.upload(&lhs).expect("lhs upload");
    let rhs_buf = device.upload(&rhs).expect("rhs upload");
    let out_buf = device.alloc_zeroed::<T>(4).expect("out alloc");
    binary_elementwise_into::<AddOp, T>(device, &lhs_buf, &rhs_buf, &out_buf, BlockWidth::DEFAULT)
        .expect("binary add dispatch");
    let mut got = vec![expected[0]; 4];
    device.download(&out_buf, &mut got).expect("download");
    assert_eq!(T::bits(&got), T::bits(&expected), "binary add mismatch");
}

/// Host-fixture bridge so one test body covers both 16-bit scalars.
trait FromHostTrio: Sized + Copy {
    fn from_host_trio(
        lhs: &[f32; 4],
        rhs: &[f32; 4],
        expected: &[f32; 4],
    ) -> (Vec<Self>, Vec<Self>, Vec<Self>);
    /// Raw bit patterns for exact comparison (`F16`/`Bf16` lack `Debug`).
    fn bits(values: &[Self]) -> Vec<u16>;
}

impl FromHostTrio for F16 {
    fn from_host_trio(
        lhs: &[f32; 4],
        rhs: &[f32; 4],
        expected: &[f32; 4],
    ) -> (Vec<Self>, Vec<Self>, Vec<Self>) {
        (f16(lhs), f16(rhs), f16(expected))
    }
    fn bits(values: &[Self]) -> Vec<u16> {
        values.iter().map(|v| v.0).collect()
    }
}

impl FromHostTrio for Bf16 {
    fn from_host_trio(
        lhs: &[f32; 4],
        rhs: &[f32; 4],
        expected: &[f32; 4],
    ) -> (Vec<Self>, Vec<Self>, Vec<Self>) {
        (bf16(lhs), bf16(rhs), bf16(expected))
    }
    fn bits(values: &[Self]) -> Vec<u16> {
        values.iter().map(|v| v.0).collect()
    }
}

#[test]
fn cuda_binary_add_compiles_admitted_half_scalars() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_binary_add::<F16>(&device);
    assert_binary_add::<Bf16>(&device);
}

fn assert_binary_mul<T>(device: &CudaDevice)
where
    T: DialectScalar<CudaC> + Pod + PartialEq + Copy + FromHostTrio,
{
    let (lhs, rhs, expected) = T::from_host_trio(
        &[1.0, 2.0, 3.0, 4.0],
        &[4.0, 3.0, 2.0, 1.0],
        &[4.0, 6.0, 6.0, 4.0],
    );
    let lhs_buf = device.upload(&lhs).expect("lhs upload");
    let rhs_buf = device.upload(&rhs).expect("rhs upload");
    let out_buf = device.alloc_zeroed::<T>(4).expect("out alloc");
    binary_elementwise_into::<MulOp, T>(device, &lhs_buf, &rhs_buf, &out_buf, BlockWidth::DEFAULT)
        .expect("binary mul dispatch");
    let mut got = vec![expected[0]; 4];
    device.download(&out_buf, &mut got).expect("download");
    assert_eq!(T::bits(&got), T::bits(&expected), "binary mul mismatch");
}

#[test]
fn cuda_binary_mul_compiles_admitted_half_scalars() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_binary_mul::<F16>(&device);
    assert_binary_mul::<Bf16>(&device);
}

fn assert_unary_neg<T>(device: &CudaDevice)
where
    T: DialectScalar<CudaC> + Pod + PartialEq + Copy + FromHostTrio,
{
    let (input, _, expected) =
        T::from_host_trio(&[1.0, -2.0, 3.0, -4.0], &[0.0; 4], &[-1.0, 2.0, -3.0, 4.0]);
    let in_buf = device.upload(&input).expect("input upload");
    let out_buf = device.alloc_zeroed::<T>(4).expect("out alloc");
    unary_elementwise_into::<NegOp, T>(device, &in_buf, &out_buf, BlockWidth::DEFAULT)
        .expect("unary neg dispatch");
    let mut got = vec![expected[0]; 4];
    device.download(&out_buf, &mut got).expect("download");
    assert_eq!(T::bits(&got), T::bits(&expected), "unary neg mismatch");
}

fn assert_binary_div<T>(device: &CudaDevice)
where
    T: DialectScalar<CudaC> + Pod + PartialEq + Copy + FromHostTrio,
{
    let (lhs, rhs, expected) = T::from_host_trio(
        &[4.0, 6.0, 8.0, 9.0],
        &[2.0, 3.0, 4.0, 3.0],
        &[2.0, 2.0, 2.0, 3.0],
    );
    let lhs_buf = device.upload(&lhs).expect("lhs upload");
    let rhs_buf = device.upload(&rhs).expect("rhs upload");
    let out_buf = device.alloc_zeroed::<T>(4).expect("out alloc");
    binary_elementwise_into::<DivOp, T>(device, &lhs_buf, &rhs_buf, &out_buf, BlockWidth::DEFAULT)
        .expect("binary div dispatch");
    let mut got = vec![expected[0]; 4];
    device.download(&out_buf, &mut got).expect("download");
    assert_eq!(T::bits(&got), T::bits(&expected), "binary div mismatch");
}

#[test]
fn cuda_binary_div_compiles_admitted_half_scalars() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_binary_div::<F16>(&device);
    assert_binary_div::<Bf16>(&device);
}

#[test]
fn cuda_unary_neg_compiles_admitted_half_scalars() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_unary_neg::<F16>(&device);
    assert_unary_neg::<Bf16>(&device);
}

fn assert_scalar_sub<T>(device: &CudaDevice)
where
    T: DialectScalar<CudaC> + Pod + PartialEq + Copy + FromHostTrio,
{
    let (input, _, expected) =
        T::from_host_trio(&[5.0, 6.0, 7.0, 8.0], &[0.0; 4], &[4.0, 5.0, 6.0, 7.0]);
    let one = T::from_host_trio(&[1.0; 4], &[0.0; 4], &[0.0; 4]).0[0];
    let in_buf = device.upload(&input).expect("input upload");
    let out_buf = device.alloc_zeroed::<T>(4).expect("out alloc");
    scalar_elementwise_into::<SubOp, T>(device, &in_buf, one, &out_buf, BlockWidth::DEFAULT)
        .expect("scalar sub dispatch");
    let mut got = vec![expected[0]; 4];
    device.download(&out_buf, &mut got).expect("download");
    assert_eq!(T::bits(&got), T::bits(&expected), "scalar sub mismatch");
}

#[test]
fn cuda_scalar_sub_compiles_admitted_half_scalars() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_scalar_sub::<F16>(&device);
    assert_scalar_sub::<Bf16>(&device);
}

fn assert_strided_binary_add<T>(device: &CudaDevice)
where
    T: DialectScalar<CudaC> + Pod + PartialEq + Copy + FromHostTrio,
{
    // 2x2 logical over a transposed (swapped-stride) view: exercises the
    // strided decode path, not just the dense one.
    let layout = Layout::try_new([2, 2], [1, 2], 0).expect("layout");
    let out_layout = Layout::c_contiguous([2, 2]).expect("output layout");
    // Logical lhs under the transposed view is [[1, 3], [2, 4]]; logical
    // rhs is [[1, 0], [0, 0]]. The asymmetric sum [[2, 3], [2, 4]] proves
    // the transpose ran rather than a dense add.
    let (lhs, rhs, expected) = T::from_host_trio(
        &[1.0, 2.0, 3.0, 4.0],
        &[1.0, 0.0, 0.0, 0.0],
        &[2.0, 3.0, 2.0, 4.0],
    );
    let lhs_buf = device.upload(&lhs).expect("lhs upload");
    let rhs_buf = device.upload(&rhs).expect("rhs upload");
    let out_buf = device.alloc_zeroed::<T>(4).expect("out alloc");
    binary_elementwise_strided_into::<AddOp, T, 2>(
        device,
        StridedOperand {
            buffer: &lhs_buf,
            layout: &layout,
        },
        StridedOperand {
            buffer: &rhs_buf,
            layout: &layout,
        },
        StridedOperand {
            buffer: &out_buf,
            layout: &out_layout,
        },
        BlockWidth::DEFAULT,
    )
    .expect("strided binary add dispatch");
    let mut got = vec![expected[0]; 4];
    device.download(&out_buf, &mut got).expect("download");
    assert_eq!(
        T::bits(&got),
        T::bits(&expected),
        "strided binary add mismatch"
    );
}

fn assert_unary_sqrt<T>(device: &CudaDevice)
where
    T: DialectScalar<CudaC> + Pod + PartialEq + Copy + FromHostTrio,
{
    let (input, _, expected) =
        T::from_host_trio(&[1.0, 4.0, 9.0, 16.0], &[0.0; 4], &[1.0, 2.0, 3.0, 4.0]);
    let in_buf = device.upload(&input).expect("input upload");
    let out_buf = device.alloc_zeroed::<T>(4).expect("out alloc");
    unary_elementwise_into::<SqrtOp, T>(device, &in_buf, &out_buf, BlockWidth::DEFAULT)
        .expect("unary sqrt dispatch");
    let mut got = vec![expected[0]; 4];
    device.download(&out_buf, &mut got).expect("download");
    assert_eq!(T::bits(&got), T::bits(&expected), "unary sqrt mismatch");
}

#[test]
fn cuda_unary_sqrt_compiles_admitted_half_scalars() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_unary_sqrt::<F16>(&device);
    assert_unary_sqrt::<Bf16>(&device);
}

#[test]
fn cuda_strided_binary_add_compiles_admitted_half_scalars() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_strided_binary_add::<F16>(&device);
    assert_strided_binary_add::<Bf16>(&device);
}

fn assert_strided_unary_neg<T>(device: &CudaDevice)
where
    T: DialectScalar<CudaC> + Pod + PartialEq + Copy + FromHostTrio,
{
    let layout = Layout::try_new([2, 2], [1, 2], 0).expect("layout");
    let out_layout = Layout::c_contiguous([2, 2]).expect("output layout");
    // Logical input under the transposed view is [[1, 3], [2, 4]].
    let (input, _, expected) =
        T::from_host_trio(&[1.0, 2.0, 3.0, 4.0], &[0.0; 4], &[-1.0, -3.0, -2.0, -4.0]);
    let in_buf = device.upload(&input).expect("input upload");
    let out_buf = device.alloc_zeroed::<T>(4).expect("out alloc");
    unary_elementwise_strided_into::<NegOp, T, 2>(
        device,
        StridedOperand {
            buffer: &in_buf,
            layout: &layout,
        },
        StridedOperand {
            buffer: &out_buf,
            layout: &out_layout,
        },
        BlockWidth::DEFAULT,
    )
    .expect("strided unary neg dispatch");
    let mut got = vec![expected[0]; 4];
    device.download(&out_buf, &mut got).expect("download");
    assert_eq!(
        T::bits(&got),
        T::bits(&expected),
        "strided unary neg mismatch"
    );
}

#[test]
fn cuda_strided_unary_neg_compiles_admitted_half_scalars() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_strided_unary_neg::<F16>(&device);
    assert_strided_unary_neg::<Bf16>(&device);
}

fn assert_strided_scalar_mul<T>(device: &CudaDevice)
where
    T: DialectScalar<CudaC> + Pod + PartialEq + Copy + FromHostTrio,
{
    let layout = Layout::try_new([2, 2], [1, 2], 0).expect("layout");
    let out_layout = Layout::c_contiguous([2, 2]).expect("output layout");
    // Logical input under the transposed view is [[1, 3], [2, 4]].
    let (input, _, expected) =
        T::from_host_trio(&[1.0, 2.0, 3.0, 4.0], &[0.0; 4], &[2.0, 6.0, 4.0, 8.0]);
    let two = T::from_host_trio(&[2.0; 4], &[0.0; 4], &[0.0; 4]).0[0];
    let in_buf = device.upload(&input).expect("input upload");
    let out_buf = device.alloc_zeroed::<T>(4).expect("out alloc");
    scalar_elementwise_strided_into::<MulOp, T, 2>(
        device,
        StridedOperand {
            buffer: &in_buf,
            layout: &layout,
        },
        two,
        StridedOperand {
            buffer: &out_buf,
            layout: &out_layout,
        },
        BlockWidth::DEFAULT,
    )
    .expect("strided scalar mul dispatch");
    let mut got = vec![expected[0]; 4];
    device.download(&out_buf, &mut got).expect("download");
    assert_eq!(
        T::bits(&got),
        T::bits(&expected),
        "strided scalar mul mismatch"
    );
}

#[test]
fn cuda_strided_scalar_mul_compiles_admitted_half_scalars() {
    let Some(device) = device_or_skip() else {
        return;
    };
    assert_strided_scalar_mul::<F16>(&device);
    assert_strided_scalar_mul::<Bf16>(&device);
}
