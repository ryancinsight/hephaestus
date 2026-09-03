# Runtime-Rank Fusion

Fusion combines an expression and its final operation into one provider-owned
kernel. `FusedExpression` supplies one dialect expression fragment, while
`FusedElementwiseOps` and `FusedReductionOps` keep elementwise and reduction
contracts separate. The operation receives borrowed `DynamicStridedView`
values, so runtime rank and broadcasting do not require storage copies.

The WGPU provider owns the wrapper shader, layout metadata, binding validation,
and a cache keyed by provider family, scalar, rank, input count, and exact
expression source. Leto remains the single source for broadcast and output
injectivity laws. Invalid source fragments, foreign buffers, aliased outputs,
non-broadcastable inputs, and unsupported reduction axes return typed errors
before submission.

The reduction output preserves rank and keeps the reduced axis at extent one.
Sum and product use their mathematical identities for an empty axis; mean,
minimum, and maximum reject an empty axis because no identity is defined by the
contract.
