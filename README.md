# Tileline

> Parallel Compute Runtime with Game Engine Capabilities

Tileline is a runtime-first engine workspace for projects that need explicit CPU/GPU scheduling,
physics, scripting, networking, scene composition, asset packaging, and live runtime tooling.

The shortest accurate way to describe Tileline today is:

**Tileline is a parallel compute runtime with game engine capabilities.**

That wording matters. Tileline is not being built as an editor-first clone of a traditional game
engine. It is being built as a **scheduler-aware runtime stack** where rendering, physics,
scripting, networking, and asset flow are all treated as execution systems that can be measured,
balanced, and evolved deliberately.

## Recommended Positioning

If we want to market Tileline differently, this is the direction that fits the codebase best.

### Primary Positioning

- **Parallel compute runtime with game engine capabilities**

### Short Alternatives

- **A runtime-first engine for simulation-heavy games**
- **A scheduler-aware engine stack for physics, graphics, and networking**
- **A Linux-first interactive runtime for games, sandboxes, and compute-heavy scenes**
- **An engine architecture where CPU/GPU orchestration is part of the product**

### What Tileline Is Not Trying To Be

- not a Unity or Unreal clone
- not just a renderer benchmark lab
- not only a physics sandbox
- not only a scripting language experiment

Tileline is strongest when described as a system for **interactive workloads that benefit from
parallel execution, explicit scheduling, telemetry, and engine-grade runtime composition**.

## Why Tileline Is Different

Most engines expose scenes, editors, and assets first, and keep their scheduling model buried under
the hood.

Tileline does the opposite:

- it treats **parallel execution** as a product feature
- it exposes **runtime telemetry and scaling behavior** as first-class signals
- it builds **game engine capabilities on top of that runtime model**
- it aims to keep the stack understandable from CPU topology all the way to scene presentation

That makes Tileline especially interesting for:

- simulation-heavy games
- sandbox and systems-driven prototypes
- performance-focused engine R&D
- Linux-first runtime experimentation
- projects that want tighter control over CPU/GPU/physics/network behavior than mainstream engines

## Current Identity

Tileline currently combines these roles in one workspace:

- a **parallel compute runtime**
- a **scene runtime** for 2D/3D content
- a **physics-backed game loop**
- a **scriptable execution environment**
- a **networking/runtime packaging toolchain**
- a **live debug and telemetry shell** via TLApp

The result is not a single monolithic engine binary. It is a collection of focused subsystems that
compose into a runtime-driven engine stack.

## Core Subsystems

| Subsystem | Role | Notes |
| --- | --- | --- |
| `mps/` | Multi Processing Scaler | CPU scheduling, topology awareness, lock-free dispatch, SIMD direction, separate repo/submodule |
| `gms/` | Graphics Multi Scaler | GPU discovery, scoring, workload planning, multi-GPU direction |
| `mgs/` | Mobile Graphics Scheduler | Mobile/TBDR-aware scheduling and fallback path |
| `paradoxpe/` | Physics core | Fixed-step physics, SoA body storage, broadphase, narrowphase, solver, joints, sleep, snapshots |
| `tl-core/` | Bridge and render-core layer | Runtime bridge logic, multigpu sync abstractions, Vulkan transition work |
| `runtime/` | Canonical app/runtime layer | TLApp, scene runtime, project loading, console, render integration, sprite/script systems |
| `nps/` | Network Packet Scaler | Low-level packet path, reliability, authority/snapshot transport direction |
| `MAS` | Multi Audio Synthesizer | Runtime-owned audio direction, early integration path |

## Content And Runtime Path

Tileline already has an engine-owned content/runtime path rather than only example glue.

### Content Formats

- `.tlscript`: runtime scripting and scene control
- `.tlsprite`: sprite/HUD/light-oriented authoring format
- `.tljoint`: composition layer for scenes that combine scripts and sprite programs
- `.tlpfile`: project manifest / runtime project root
- `.pak`: runtime packaging format for distributable asset bundles

### Runtime Surfaces

- `tlapp`: canonical demo/runtime executable
- in-app console overlay (`Ctrl+F1`) for runtime control, metrics, and diagnostics
- project GUI path for `.tlpfile`
- runtime scene pipeline for 3D and early 2D foundation work

## What Exists Today

The codebase already includes real, non-trivial engine/runtime functionality:

- parallel CPU scheduling infrastructure through MPS integration
- GPU planning work through GMS and MGS
- ParadoxPE fixed-step physics with snapshots/interpolation and contact telemetry
- TLApp runtime with scene build, telemetry HUD, live console, and runtime controls
- `.tlscript`, `.tlsprite`, `.tljoint`, and `.tlpfile` authoring/loading path
- `.pak` pack/list/unpack tooling
- render-path experimentation including hybrid RT/FSR-facing work and raw Vulkan migration work
- early 2D foundation planning and runtime integration direction

## Current Reality And Honesty Check

Tileline is ambitious, but the README should stay honest about where the project is.

### True Today

- Tileline is already more than a prototype benchmark workspace
- it has a real runtime shape
- it has real physics, scripting, scene, console, packaging, and bridge layers
- it is increasingly coherent as a runtime-first engine stack

### Also True Today

- the project is still evolving quickly
- render backend transition work is ongoing
- the stable runtime path still leans on `wgpu` while raw Vulkan cutover work continues
- some systems are mature enough for experimentation, not yet for mass-market stability
- several independence goals (`wgpu`, `rayon`, `bevy`) are active roadmap work, not finished facts

In other words:

**Tileline is credible as an engine architecture and runtime platform today, and it is still in the
phase where architectural choices matter more than polish.**

## What To Call Tileline Publicly

If the goal is to market Tileline more effectively, this is the wording I would recommend using in
public-facing places:

> Tileline is a parallel compute runtime with game engine capabilities, built for
> simulation-heavy interactive software and games that benefit from explicit CPU/GPU scheduling,
> physics, scripting, telemetry, and runtime composition.

That line fits the repo better than simply calling it “a game engine,” because it highlights the
real differentiator instead of underselling the architecture.

## Licensing Model

Tileline is currently licensed under `MPL-2.0`, and the most realistic long-term model for this
project is **open core + proprietary larger works**, not "close the engine later."

### What Should Stay Open

- `mps/`, `paradoxpe/`, `nps/`, and `tl-core/`
- core runtime/content contracts
- format specifications and baseline loaders:
  - `.tlscript`
  - `.tlsprite`
  - `.tljoint`
  - `.tlpfile`
  - `.pak`

These pieces are the trust layer of Tileline. They are where external validation, performance
work, and architectural credibility matter most.

### What Can Be Commercial

- studio/editor tooling
- premium renderer/tooling packs
- hosted telemetry/build/deployment services
- enterprise/platform integration layers
- premium templates, samples, and content bundles

The practical rule is:

> keep the runtime contracts open, and monetize advanced tooling, hosted services, and premium
> workflow layers around them.

For the fuller policy, see [LICENSE-STRATEGY.md](LICENSE-STRATEGY.md).

## Repository Layout

```text
.
├── gms/         GPU planning, scoring, and multi-GPU direction
├── mgs/         mobile graphics scheduling path
├── mps/         CPU scheduling runtime (vendored as submodule)
├── nps/         network packet/runtime transport foundation
├── paradoxpe/   physics engine core
├── runtime/     TLApp, scene runtime, content loading, console, integration
├── tl-core/     bridge, sync, and render-core transition layer
├── docs/        design docs, roadmaps, release notes, demos
├── scripts/     packaging, release, and helper scripts
└── dist/        generated release/package artifacts
```

## Quick Start

### 1. Clone And Initialize Submodules

```bash
git submodule update --init --recursive
```

This matters because `mps/` is tracked as its own repository and vendored here as a submodule.

### 2. Check The Workspace

```bash
cargo check
```

### 3. Run TLApp

```bash
cargo run -p runtime --bin tlapp -- --fps-cap 60 --vsync auto
```

Legacy example entrypoint also exists:

```bash
cargo run -p runtime --example tlapp -- --fps-cap 60 --vsync auto
```

### 4. Open The Project GUI

```bash
cargo run -p runtime --bin tlproject_gui -- --project docs/demos/tlapp/tlapp_project.tlpfile
```

### 5. Open The `.tlsprite` Editor

```bash
cargo run -p runtime --bin tlsprite_editor -- --file docs/demos/tlapp/bounce_hud.tlsprite
```

### 6. Package Demo Assets

```bash
./scripts/package_prebeta_pak.sh
```

## Useful Development Commands

### Workspace Checks

```bash
cargo check
cargo test -p tl-core
cargo test -p runtime
```

### GMS Benchmark

```bash
cargo run -p gms --example render_benchmark -- --mode max --vsync off --warmup 2 --duration 10 --resolution 1280x720
```

### MGS Benchmark

```bash
cargo run -p mgs --example render_benchmark -- --mode stable --vsync on --warmup 2 --duration 10 --resolution 1280x720
```

### NPS Starter Example

```bash
cargo run -p nps --example starter_packet
```

## Documentation Guide

Start here:

- `docs/README.md`: documentation index
- `docs/tileline-v0.5.0-roadmap.md`: current major roadmap (`Heimdall Update`)
- `docs/tileline-v0.5.5-roadmap.md`: upcoming MPS SIMD + standalone extraction track
- `docs/tileline-v0.6.0-roadmap.md`: MLS runtime stack with unified RAM+VRAM / UMA-first direction
- `docs/tileline-v0.6.5-roadmap.md`: GGUF + NVFP4 follow-up for MLS
- `docs/runtime-tlapp-console.md`: in-app runtime console and live controls
- `docs/runtime-pak.md`: `.pak` packaging flow
- `docs/runtime-tlpfile-gui.md`: `.tlpfile` runtime/project shell
- `docs/paradoxpe-foundation.md`: physics architecture notes
- `docs/nps-protocol.md`: networking/runtime transport notes
- `LICENSE-STRATEGY.md`: open-core / proprietary-layer policy direction
- `MPS-BENCHMARK.md`: benchmark notes and comparative performance thinking

## Roadmap Direction

### v0.5.0: Heimdall Update

The major direction for `v0.5.0` is not cosmetic polish. It is about making Tileline behave like a
coherent runtime-first engine core.

That roadmap centers on:

- render-stack optimization
- effects and texture support
- ParadoxPE + MPS revision
- MAS runtime integration
- stronger GPU planning and scaling
- reduced dependence on `rayon`, `bevy`, and eventually `wgpu` in shipping paths

### v0.5.5: MPS Expansion

The `v0.5.5` direction extends the CPU runtime identity even further:

- SIMD work (`AVX-512`, `NEON`, `VMX/AltiVec` direction)
- stronger runtime-dispatched MPS kernels
- MPS living as a truly independent library/repo while still powering Tileline

### v0.6.0: MLS Runtime Stack

The `v0.6.0` direction turns MLS into a first-class runtime layer:

- inference + local training
- runtime-owned control surfaces across `.tlpfile`, CLI, and `.tlscript`
- advisory-only physics assist
- unified RAM+VRAM / UMA-first memory model instead of a user-visible PyTorch-style split

### v0.6.5: MLS Format + Precision Expansion

The `v0.6.5` follow-up extends that MLS base with:

- `GGUF` model support
- `NVFP4`-class low-precision NVIDIA inference paths
- stronger format/precision telemetry without breaking the v0.6.0 MLS contract

## Why This Repo Can Be Marketed Differently

Tileline has enough unique structure now that marketing it as merely “another engine prototype” is
underselling it.

A better framing is:

- **engine architecture for parallel runtime ownership**
- **compute-oriented runtime that can power games**
- **scheduler-first stack for physics, graphics, scripts, and packets**

That is not just branding. It reflects the actual code layout:

- separate CPU and GPU scaling layers
- a real physics engine core
- runtime-native scene and asset formats
- live runtime tooling
- packaging and project composition
- a growing bridge layer between simulation and rendering

## Status Summary

Tileline is currently best understood as:

- **experimental** in architecture
- **serious** in direction
- **usable for internal demos and runtime experiments**
- **not yet pretending to be a finished mainstream engine**

That combination is a strength, not a weakness, if we describe it correctly.

## License

Tileline is licensed under the Mozilla Public License 2.0 (`MPL-2.0`).
The root `LICENSE` file is the source of truth.
