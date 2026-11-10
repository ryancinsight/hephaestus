# ADR 0020: Hephaestus release scope for 1.0

- Status: Accepted
- Date: 2026-11-10
- Class: `[arch]` `[minor]`

## Context

Hephaestus (workspace 0.28.0, `hephaestus-compute`) provides the shared GPU
device substrate for the Atlas stack, including compute contracts, host
reference execution, and backend implementations (WGPU, CUDA, ROCm, Metal via
WGPU). The backlog contains open correctness, verification, and backend-parity
items, and no milestone ladder exists to 1.0.

## Decision

Adopt a three-milestone ladder to 1.0 recorded in `backlog/milestones.md`:

- **M-compute-contracts** (0.29.0): users allocate buffers and execute core
  compute operations through the device contract and portable backends.
- **M-backend-parity** (0.30.0): users exercise verified backend parity for key
  operation families and conformance coverage.
- **M-1.0** (1.0.0): stable compute substrate release (authority-gated) with
  documented contracts and release gates satisfied.

## Alternatives

- Plan 1.0 as a single release: rejected because backend parity and
  verification work remains open.
- Keep roadmap implicit in backlog items: rejected because stack policy
  requires a runnable capability-oriented milestone ladder.

## Consequences

- Open items carry `milestone:` assignments and feature items name the exit
  check they serve.
- Milestones close only when exit checks pass on fetched default commits; 1.0
  release/publication remains authority-gated.
- This ADR is rewritten in place as scope evidence changes.
