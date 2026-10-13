# Checklist — Hephaestus #407

Target version: **0.20.3**. Change class: **[patch]**. Phase: **Closure**.
Scope: mapping callback delivery in the canonical WGPU readback path.

- [x] [patch] Recover upstream issue and implementation evidence on master.
- [x] [patch] Reproduce the callback-pending failure with real parallel Metal readbacks.
- [x] [patch] Establish guarded publication and one shared submission/callback deadline.
- [x] [patch] Preserve two-owner quarantine, overflow allocation, and staging cleanup.
- [x] [patch] Verify retained/overflow outcomes, deadlines, reuse, and Loom models (7 tests).
- [x] [patch] Verify warning-denied default all-target Clippy.
- [x] [patch] Verify eight required-device parallel runs: 64 test executions and 16,384 regression readbacks.
- [x] [patch] Verify minimal-library Clippy, four doctests, warning-denied rustdoc, formatting, and diff consistency.
- [x] [patch] Synchronize workspace/facade patch versions and changelog.
- [x] [patch] Prepare the focused pull request description with runtime evidence and residual limits.

Residual verification boundary: Metal lacks `SHADER_F64`; the unchanged
required-device aggregate has two independent FP64 acquisition failures.
Windows/Vulkan execution requires the hosted WGPU lane. No release tag or merge
belongs to this increment.

Verification: aarch64 macOS Metal, Rust 1.95.0, WGPU 30.0.1, standalone locked
graph. WGPU Nextest: **84 passed, 1 failed, 0 skipped** in 46.116 seconds on the rebased revision.
The failing aggregate names only the two existing FP64 acquisition clauses;
no callback-pending transfer failure remains. Individual Nextest deadlines
remain 60 seconds. Seven completion tests include two Loom models. All four
modified/new Rust files remain below 500 lines; external dependency identities
remain unchanged. Next action: pull-request review and hosted GPU validation.
