# Tileline Licensing Strategy

This document is a practical licensing and commercialization strategy for the current Tileline
repository layout. It is not legal advice. It is the project's intended policy direction so the
open-core and commercial layers do not drift into each other by accident.

## Current Baseline

- The repository is licensed under `MPL-2.0`.
- Contributions submitted to this repository are accepted under that same license.
- Existing `MPL-2.0` source files should be treated as permanently open within this repository
  unless they are rewritten outside the covered file history.

That means the realistic path is **open core + proprietary larger works**, not "open everything now
and close it later."

## What Should Stay FOSS

These parts are the credibility layer of Tileline and should remain open-source:

- `mps/`
  - CPU scheduling, lock-free dispatch, topology awareness, SIMD/runtime scaling direction
- `paradoxpe/`
  - fixed-step physics core, collision pipeline, solver, snapshots, world model
- `nps/`
  - packet/runtime transport foundation and protocol-facing core
- `tl-core/`
  - bridge contracts, synchronization surfaces, backend-neutral integration layer
- content/runtime formats and their specifications
  - `.tlscript`
  - `.tlsprite`
  - `.tljoint`
  - `.tlpfile`
  - `.pak`
- the canonical runtime/content loading path needed to exercise those formats

## Why These Should Stay Open

- They are the technical trust layer of the project.
- They are the parts most likely to benefit from external validation, profiling, and correctness
  work.
- They are where Tileline differentiates itself architecturally.
- Closing them would weaken adoption, auditing, and contributor confidence.

## What Can Reasonably Be Commercial

These are better candidates for proprietary products or separate commercial repositories:

- advanced editor / studio tooling
  - scene editor
  - visual debugger
  - profiling UI
  - asset browser and production authoring workflows
- premium rendering/tooling layers
  - high-end postfx packs
  - tuned RT/reflection bundles
  - vendor-specific aggressive calibration packs
- cloud and hosted services
  - remote build/packaging
  - telemetry aggregation
  - crash analytics
  - collaboration and deployment services
- enterprise/platform integrations
  - certification tooling
  - commercial deployment helpers
  - platform-specific SDK packaging layers
- premium templates, sample games, content packs, and art/tool bundles

## Recommended Project Model

Tileline should behave as:

- **Open core** for runtime, physics, scheduling, protocol, and formats
- **Commercial layers** for advanced tooling, hosted services, and premium deployment/optimization
  products

This is consistent with `MPL-2.0`, because MPL allows a larger work to combine covered files with
other files under different terms, as long as the covered files remain available under MPL.

## Boundary Rule

When deciding whether something belongs in the open repository or in a commercial layer, use this
test:

- if it defines the engine/runtime contract, it should usually stay open
- if it is a premium workflow, hosted service, calibration pack, or studio-facing product layer,
  it can live outside the open repository

## Things We Should Not Quietly Close

The following should not be moved behind a proprietary wall without a deliberate public policy
change and a repository split:

- scheduler core
- physics core
- network transport core
- content format specifications
- baseline runtime loading path for those formats

## Trademarks and Branding

Open source code does not require open trademark rights.

The project should reserve:

- `Tileline`
- official logos
- official brand assets
- certification/badging language

for maintainer-approved use, even if core code remains open-source.

## Contributor Expectations

By contributing to this repository, contributors are contributing to the open-core layer.

That means:

- contributions here stay under `MPL-2.0`
- maintainers may distribute this repository as part of larger open or commercial distributions
- contribution to this repository does not imply participation in future proprietary repositories,
  revenue share, or ownership rights

## Recommended Repo Split Over Time

### Open Repositories

- `tileline`
- `mps`
- any future open protocol/spec repositories

### Potential Commercial Repositories

- `tileline-studio`
- `tileline-pro-render`
- `tileline-cloud`
- `tileline-platform-sdk`
- `tileline-enterprise-tools`

## Short Policy Summary

If we need a one-paragraph version:

> Tileline should keep its runtime, scheduling, physics, transport, and content-format foundations
> open under `MPL-2.0`, while reserving advanced tooling, hosted services, premium optimization
> layers, and branded studio products for separate commercial offerings.
