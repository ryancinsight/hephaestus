//! WGPU instantiation of the shared decomposition conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and
//! [`hephaestus_core::DecompositionOps`]; this file only supplies the device
//! and the backend seam value.

use hephaestus_conformance::assert_decomposition_contract;
use hephaestus_wgpu::{WgpuDecompositionOps, WgpuDevice};
use syn::visit::{self, Visit};

const OWNED_READBACK_SOURCES: &[(&str, usize, &str)] = &[
    (
        "bidiagonal",
        1,
        include_str!("../src/application/decomposition/bidiagonal.rs"),
    ),
    (
        "bunch_kaufman",
        1,
        include_str!("../src/application/decomposition/bunch_kaufman.rs"),
    ),
    (
        "col_piv_qr",
        2,
        include_str!("../src/application/decomposition/col_piv_qr.rs"),
    ),
    (
        "eigen",
        3,
        include_str!("../src/application/decomposition/eigen.rs"),
    ),
    (
        "full_piv_lu",
        2,
        include_str!("../src/application/decomposition/full_piv_lu.rs"),
    ),
    (
        "hessenberg",
        1,
        include_str!("../src/application/decomposition/hessenberg.rs"),
    ),
    (
        "schur",
        1,
        include_str!("../src/application/decomposition/schur.rs"),
    ),
    (
        "svd",
        3,
        include_str!("../src/application/decomposition/svd.rs"),
    ),
    (
        "udu",
        2,
        include_str!("../src/application/decomposition/udu.rs"),
    ),
];

#[derive(Default)]
struct DownloadCallVisitor {
    borrowed: usize,
    owned: usize,
}

impl DownloadCallVisitor {
    fn record(&mut self, method: &syn::Ident) {
        if method == "download" {
            self.borrowed += 1;
        } else if method == "download_owned" {
            self.owned += 1;
        }
    }
}

impl<'syntax> Visit<'syntax> for DownloadCallVisitor {
    fn visit_expr_method_call(&mut self, node: &'syntax syn::ExprMethodCall) {
        self.record(&node.method);
        visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_call(&mut self, node: &'syntax syn::ExprCall) {
        if let syn::Expr::Path(path) = node.func.as_ref()
            && let Some(method) = path.path.segments.last()
        {
            self.record(&method.ident);
        }
        visit::visit_expr_call(self, node);
    }
}

#[test]
fn non_blocked_decomposition_heap_readbacks_are_provider_owned() {
    let mut total_owned = 0;
    for (name, expected_owned, source) in OWNED_READBACK_SOURCES {
        let syntax = syn::parse_file(source).expect("decomposition source must parse as Rust");
        let mut calls = DownloadCallVisitor::default();
        calls.visit_file(&syntax);
        assert_eq!(
            calls.borrowed, 0,
            "WGPU {name} must not retain caller-owned decomposition readback"
        );
        assert_eq!(
            calls.owned, *expected_owned,
            "WGPU {name} provider-owned readback inventory changed"
        );
        total_owned += calls.owned;
    }
    assert_eq!(total_owned, 16, "WGPU owned readback total changed");
}

#[test]
fn wgpu_satisfies_the_decomposition_contract() {
    let device = match WgpuDevice::try_default("hephaestus-decomposition-conformance") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skip WGPU decomposition conformance: adapter unavailable ({error})");
            return;
        }
    };
    assert_decomposition_contract(&device, &WgpuDecompositionOps);
}
