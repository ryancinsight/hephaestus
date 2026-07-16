# Provider default source and MSRV

## Context

Hephaestus consumes Atlas providers through Git source dependencies. Fixed
revision requirements in the workspace created a second source identity when a
consumer also required the provider's default branch. Leto 0.37.0, Mnemosyne
0.5.0, and Moirai 0.4.0 require Rust 1.95, making the prior 1.87 declaration
false for the resolved graph.

## Options

1. Retain fixed revisions and require consumers to align their lockfiles.
2. Add workspace-local patches to collapse the source identities.
3. Use each provider's default source and publish the actual Rust 1.95
   requirement.

## Decision

Choose option 3. The workspace has one direct requirement per provider, no
revision or patch override, and every published package inherits Rust 1.95.
The pre-1.0 MSRV change advances the workspace from 0.14.0 to 0.15.0.

## Consequences

Consumers update their lockfiles after the release and resolve one source
identity per provider. Rust 1.94 and earlier are rejected at resolution time.
The provider-source graph is verified in both Hephaestus and Apollo.
