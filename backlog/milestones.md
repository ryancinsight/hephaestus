# Milestone ladder ([ADR 0020](../docs/adr/0020-release-scope-1-0.md))
<a id="M-compute-contracts"></a> **M-compute-contracts** (0.29.0) needs: (none) — device contracts and core compute.
- exit: pass core contract and host tests (`cargo test -p hephaestus-core --locked --all-features` and `cargo test -p hephaestus-host --locked --all-features` pass).
- exit: run prepared map reduction example (`cargo run -p hephaestus-compute --example prepared_map_reduction --locked --features wgpu` exits successfully).
- exit: pass portable WGPU tests (`cargo test -p hephaestus-wgpu --locked --all-features` passes).
- standing: no open correctness item.
<a id="M-backend-parity"></a> **M-backend-parity** (0.30.0) needs: M-compute-contracts — backend parity and conformance.
- exit: pass conformance suite on host (`cargo test -p hephaestus-conformance --locked --all-features` passes).
- exit: resolve `HEPH-MACOS-METAL-CONFORMANCE-1` with landed evidence on the default branch.
- standing: no open correctness item.
<a id="M-1.0"></a> **M-1.0** (1.0.0) needs: M-backend-parity — stable compute substrate release (authority-gated).
- standing: `cargo fmt --all -- --check`, `cargo clippy --workspace --locked --all-targets --all-features -- -D warnings`, `cargo test --workspace --locked --all-features`, `cargo test --workspace --locked --doc --all-features`, and `cargo doc --workspace --locked --no-deps --all-features` pass; SemVer and publish dry-run gates required for declared releases pass; README and rustdoc are current; no open correctness item.
