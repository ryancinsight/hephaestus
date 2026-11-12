# ADR 0089: Restore published ROCm axis reduction paths

- Status: Accepted
- Date: 2026-11-12

Hephaestus ROCm 0.28.0 exposes rank-2 axis reductions through
`hephaestus_rocm::application::axis_reduction`. The kernel split in
[#517](https://github.com/ryancinsight/hephaestus/pull/517) moved those paths
into the contiguous reduction module. The release API comparison on
[#525](https://github.com/ryancinsight/hephaestus/pull/525) reports the missing
published paths.

Restore `application::axis_reduction` as the public owner of the axis
functions and their operation markers in
[#527](https://github.com/ryancinsight/hephaestus/pull/527). Keep each
operation's implementation in its own file and use the contiguous workgroup
tree for dense scalar collapses. Keep `application::reduction` for
contiguous reductions and retain the crate-root exports. This preserves the
published module layout while retaining the internal kernel split.

Git callers that use the axis functions through `application::reduction`
must change the import prefix to `application::axis_reduction`. This applies
to `reduce_axis`, `max_axis`, `mean_axis`, `min_axis`, `prod_axis`,
`sum_axis`, and their `_into` forms. The Git-only paths are removed so that
the API has one nested path for each axis operation. Callers of the
crate-root functions or the device-neutral seam keep their imports.

The contract tests import the restored module. Compare the exact candidate
against the 0.28.0 release and its Git base. The release comparison must
pass, and the Git comparison must identify only the intentional path
removals. A different API failure reopens this decision.
