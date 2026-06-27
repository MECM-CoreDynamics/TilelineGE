# Tileline v0.7.0 Roadmap

Codename: `Ironclad`

## Release Goal

`v0.7.0` is the release where Tileline becomes a **fully backend-independent parallel engine**
with no remaining dependence on external generic GPU or CPU scheduling crates in shipping paths.

Key themes:

- **WGPU exit is complete** — `tl-core`, `runtime`, and all shipping crates are free of `wgpu`
  types, `wgpu`-backed render loops, and `wgpu`-derived adapter metadata.
- **Native Vulkan ownership is audited** — while removing WGPU dependence, the raw Vulkan path
  explicitly revisits `ash` + VMA usage so allocation, lifetime, synchronization, and telemetry
  become engine-owned rather than inherited from the bootstrap layer.
- **GMS is the canonical GPU path** — all GPU scheduling, multi-GPU sync, and render backend
  decisions flow through `gms` and `tl-core` native backends (Vulkan, Metal), not through a
  portable abstraction layer.
- **Parallel-by-default execution** — MPS is the default CPU scheduler; ParadoxPE phases run
  parallel unless an explicit, telemetry-visible serial fallback is triggered.
- **MGS is production-complete** — the mobile graphics scaler path is fully validated on
  Android/Mali/Adreno and no longer treated as a secondary track.
- **BerrySR frame generation** — AI-driven frame generation (Stable Diffusion 1.5 + LoRA)
  integrated as a complementary path to FSR, targeting Tensor Core / Neural Engine / NPU.
- **`.pak` executable standard** — `.pak` becomes the standard packaging format for executable
  Tileline programs, not just asset archives.

---

## Workstream V1: WGPU Independence (Complete Exit)

**Why:** `wgpu` was a valuable bootstrap layer, but it blocks explicit multi-GPU control,
backend-native feature access, and clean dependency hygiene. `v0.7.0` finishes the exit.

### V1.1. tl-core Multi-GPU Sync Abstraction

Target:

- replace `wgpu::{Device, PollError, PollStatus, SubmissionIndex, BufferView, BufferViewMut}`
  in `tl-core/src/graphics/multigpu/sync.rs` with backend-neutral traits and handles
- define `GpuDevice`, `GpuPollStatus`, `GpuSubmissionIndex`, `GpuBufferView` as engine-owned
  abstractions with native Vulkan and Metal implementations
- keep the same bounded-wait semantics; do not regress the sub-millisecond compose budget

Acceptance:

- `multigpu/sync.rs` compiles without `use wgpu::` anywhere in production code
- Vulkan backend implements the new traits using `ash`/`vma` primitives
- Metal backend implements the new traits using `metal-rs` primitives
- multi-GPU compose wait stays under the 1 ms budget on desktop gates

### V1.2. tl-core Bridge Backend-Neutral Enum

Target:

- replace `wgpu::Backend` and `wgpu::DeviceType` usage in `tl-core/src/core/bridge.rs`
- introduce `GraphicsBackendKind` and `GraphicsDeviceClass` enums in `tl-core`
- map Vulkan -> `GraphicsBackendKind::Vulkan`, Metal -> `GraphicsBackendKind::Metal`, etc.
- replace `wgpu::BufferUsages::MAP_WRITE` checks with backend-neutral memory topology checks

Acceptance:

- `bridge.rs` compiles without `use wgpu::` anywhere in production code
- scheduler path selection still correctly identifies Metal vs Vulkan vs other backends
- Apple UMA detection remains accurate without `wgpu` adapter metadata

### V1.3. Runtime Struct Decoupling

Target:

- remove `wgpu::Instance`, `wgpu::Surface`, `wgpu::Device`, `wgpu::Queue`,
  `wgpu::SurfaceConfiguration`, `wgpu::PresentMode` from `TlAppRuntime` struct fields
- introduce backend-neutral renderer handles owned by each renderer variant
- `TlAppRuntime` stores an opaque `RendererBackend` enum instead of raw `wgpu` types
- `runtime_init.rs` bootstraps each backend (Vulkan, Metal, WGPU legacy) through its own path

Acceptance:

- `TlAppRuntime` struct definition contains no `wgpu::` types
- `runtime/src/tlapp_app/mod.rs` compiles without `use wgpu::` in production code
- each renderer variant (Wgpu, Metal, Vulkan) owns its own backend-specific state

### V1.4. Scheduler Path Decoupling

Target:

- remove `wgpu::AdapterInfo` and `wgpu::Backend` usage from `runtime/src/scheduler_path.rs`
- `RuntimeAdapterInfo` (already introduced in `v0.5.x`) becomes the sole adapter metadata type
- delete `wgpu`-to-`RuntimeAdapterInfo` conversion wrappers

Acceptance:

- `scheduler_path.rs` compiles without `use wgpu::` anywhere
- auto-scheduler path selection (`GMS` vs `MGS`) still works correctly on all platforms

### V1.5. Final Audited Dependency Cleanup

Target:

- remove `wgpu` and `egui-wgpu` from `runtime/Cargo.toml` shipping dependencies
- remove `wgpu` from `tl-core/Cargo.toml` if no longer needed by any module
- move any remaining `wgpu` usage to `dev-dependencies` or example-only crates
- `cargo tree` for `runtime` and `tl-core` must not show `wgpu` in the production dependency graph

Acceptance:

- `cargo tree -p runtime | grep wgpu` returns nothing
- `cargo tree -p tl-core | grep wgpu` returns nothing
- all existing tests pass
- runtime smoke test on macOS (Metal) and Linux (Vulkan) passes

### V1.6. Ash + VMA Native Vulkan Audit

Target:

- review the raw Vulkan backend's `ash` and VMA integration while WGPU ownership is removed
- make Vulkan object lifetime, allocator ownership, memory budgets, mapped-buffer behavior, and
  queue/sync primitives explicit in `tl-core`
- replace any remaining WGPU-derived assumptions in Vulkan telemetry, adapter probing, and
  frame-resource recycling with native `ash` + VMA data
- document the boundary between engine-owned abstractions and backend-owned Vulkan handles

Acceptance:

- Vulkan backend initialization, resize, shutdown, and frame-resource recycling paths are covered
  by smoke tests or renderer-logic tests
- VMA allocation failures and budget pressure report structured fallback diagnostics instead of
  panics or silent corruption
- native Vulkan telemetry can explain allocation pressure, queue ownership, and sync waits without
  referencing WGPU terms

---

## Workstream V2: GMS as Canonical GPU Path

**Why:** Once `wgpu` is gone, GMS must own every GPU scheduling decision without a fallback to a
portable layer.

### V2.1. GMS Native SM/CU Scaler as Default

Target:

- GMS Native SM/CU scaler becomes the default GPU scheduling path on desktop
- no silent fallback to a generic scheduler
- all domain budgets (`render|physics|ai_ml|postfx|ui`) are actively enforced
- `gms` crate is the source of truth for GPU topology, adapter profiling, and workload assignment

Acceptance:

- TLApp boots with GMS scaler active on both Metal and Vulkan paths
- domain budget telemetry is always present in HUD/console
- overload behavior degrades through explicit GMS guardrails, not through backend fallback

### V2.2. Backend-Neutral GPU Topology Everywhere

Target:

- `GpuAdapterProfile`, `GpuTopology`, and `MemoryTopology` are the canonical types across all
  crates
- no crate imports `wgpu::AdapterInfo` or `wgpu::Backend` for scheduling decisions
- GMS exports backend-neutral profiling helpers that Vulkan and Metal backends populate

Acceptance:

- `grep -r "wgpu::AdapterInfo" runtime/src tl-core/src gms/src` returns nothing
- `grep -r "wgpu::Backend" runtime/src tl-core/src gms/src` returns nothing

### V2.3. WGPU Render Loop Removal

Target:

- delete `runtime/src/wgpu_render_loop.rs` or move it to `examples/` as historical reference
- `TlAppRuntime::render_frame()` no longer has a `wgpu`-backed branch
- the canonical frame loop uses backend-neutral `RenderFrame` -> `BackendSnapshot` -> `Present`
  flow

Acceptance:

- `wgpu_render_loop.rs` does not exist in `runtime/src/`
- runtime compiles and runs on macOS Metal without `wgpu` in dependency tree
- runtime compiles and runs on Linux Vulkan without `wgpu` in dependency tree

---

## Workstream V3: Parallel-by-Default Execution

**Why:** `v0.5.0` removed `rayon` and `bevy`; `v0.7.0` makes parallel execution the silent default
with serial as an explicit, telemetry-visible exception.

### V3.1. MPS Default CPU Scheduling

Target:

- MPS (`mps::MpsScheduler` / `mps::TaskDispatcher`) is the default CPU scheduler for all
  runtime jobs
- no global thread pool from any external crate is created at runtime startup
- `PhysicsMpsRunner` is the only physics async path in shipping code

Acceptance:

- runtime startup does not spawn `rayon` or `bevy_tasks` threads
- MPS worker count and topology are visible in startup logs
- `cargo tree -p runtime | grep -E "rayon|bevy_tasks"` returns nothing

### V3.2. ParadoxPE Parallel-First Policy

Target:

- `broadphase|narrowphase|solver|integrate` run parallel by default on all shipping presets
- serial fallback is opt-in via explicit flags (`--physics-serial-fallback`) or small-N
  automatic detection
- `serial_fallback_reason` telemetry is mandatory (already implemented; policy enforcement now)
- under `8k` and `30k` gates, no phase silently falls back to serial without logging

Acceptance:

- default runtime preset produces `serial_fallback_reason: None` for all four phases under `8k`
- `30k` gate may trigger bounded serial tails, but reasons are always logged
- no hidden `for body in bodies { serial_work() }` loops remain in hot paths

### V3.3. TlscriptParallelRuntimeCoordinator Default Enable

Target:

- `TlscriptParallelRuntimeCoordinator` is enabled by default in the live frame loop
- scripts with `@parallel(domain="bodies")` contracts route through MPS chunk dispatch
- serial fallback (`evaluate_frame` on main thread) only happens when:
  - no parallel contract exists
  - workload is below chunk threshold
  - MPS is unavailable
- parallel dispatch metrics are always visible in HUD

Acceptance:

- showcase scripts with `@parallel` decorators show `dispatch_parallel_batches > 0` in metrics
- scripts without `@parallel` fall back gracefully with `dispatch_main_thread_required > 0`
- no script phase silently bypasses the coordinator

---

## Workstream V4: MGS Completion

**Why:** MGS is the mobile/TBDR-first GPU path. It must be production-complete, not a
secondary track.

### V4.1. Android + Mali/Panthor + Adreno Validation

Target:

- MGS path passes acceptance gates on:
  - Android / Mali-G710
  - Android / Mali-Panthor
  - Android / Adreno 740
- TBDR-aware tile scheduling works correctly
- mobile-safe presets (`MgsPerformanceProfile::Balanced` and `PowerSaver`) are stable

Acceptance:

- `cargo test -p mgs` passes on all target devices
- TLApp showcase runs at `30+ FPS` on mobile-safe presets without severe chopping
- memory bandwidth budgeting does not trigger GPU watchdog kills

### V4.2. Complete MGS Scene Workload Mapping

Target:

- all runtime scene density levels map correctly to MGS bridge hints
- sprite, tile, and 3D workload mixture is budgeted correctly on TBDR
- `mgs_scene_workload.md` documentation matches implementation

Acceptance:

- `mgs/src/scene_workload.rs` tests cover all density levels
- no runtime scene type crashes or corrupts output on MGS path

### V4.3. Runtime Scheduler Auto-Selection

Target:

- runtime auto-selects `MGS` path on Android and genuine mobile hardware
- `GMS` path is never silently selected on TBDR-class hardware
- scheduler selection telemetry reports the correct path and reason

Acceptance:

- `cargo run -p runtime --bin tlapp` on Android boots with `MGS` path
- `TILELINE_SCHEDULER=gms` override still works for explicit testing
- scheduler reason string is accurate (`"mali_tbdr_detected"`, `"adreno_detected"`, etc.)

---

## Workstream V5: BerrySR Frame Generation

**Why:** BerrySR is an AI-driven frame generation layer (Stable Diffusion 1.5 + LoRA) that
complements FSR/DLSS-style spatial upscaling with temporal frame synthesis.

### V5.1. BerrySR Core Integration

Target:

- integrate Stable Diffusion 1.5 + LoRA inference into the MLS runtime path
- frame generation operates on motion-vector-aware latent diffusion, not naive image-to-image
- BerrySR is treated as a **complementary** path to FSR: FSR upscales spatially, BerrySR
  synthesizes intermediate frames temporally
- input: previous frame + motion vectors + depth + current frame low-res
- output: intermediate synthesized frame

Acceptance:

- BerrySR inference pipeline compiles and runs through `MlsRuntime`
- frame-gen produces visually coherent intermediate frames under motion
- pipeline latency is budgeted and does not destabilize frame pacing

### V5.2. Hardware Backend Targets

Target:

- **NVIDIA**: Tensor Core path via CUDA/cuDNN, targeting RTX 30-series+
- **Apple**: Neural Engine path via Core ML, targeting M1/M2/M3/M4
- **Rockchip**: NPU path via RKNN, targeting RK3588-class SoCs
- AMD and Intel are **not** targeted in `v0.7.0` to keep validation scope bounded

Acceptance:

- NVIDIA Tensor Core path passes quality/perf gate on RTX 4060-class hardware
- Apple Neural Engine path passes quality/perf gate on M3-class hardware
- Rockchip NPU path passes quality gate on Orange Pi 5 / RK3588S
- each backend reports its inference latency and fallback reason in telemetry

### V5.3. Frame-Gen Budget and Control Surface

Target:

- BerrySR is toggleable and budgeted through all canonical control surfaces:
  - CLI: `--berrysr on|off`, `--berrysr-quality low|medium|high`, `--berrysr-latency-budget <ms>`
  - `.tlscript`: `berrysr_set_enabled()`, `berrysr_set_quality()`, `berrysr_set_latency_budget()`
  - `.tlpfile`: `[berrysr]` section
  - TLApp console: `berrysr.*` commands
- budget precedence: `CLI > .tlscript > .tlpfile`
- frame-gen is **disabled by default**; explicit opt-in required

Acceptance:

- BerrySR can be enabled/disabled at runtime without restarting
- quality preset changes do not crash or corrupt the render loop
- latency budget clamping prevents frame-gen from overshooting the frame time

### V5.4. FSR + BerrySR Compositing

Target:

- define the canonical compositing order:
  1. render native-resolution scene
  2. FSR spatial upscale (if enabled)
  3. BerrySR temporal synthesis (if enabled) on the upscaled stream
  4. post-FX and present
- BerrySR can run on the FSR-upscaled stream or the native stream, engine-configurable
- overload policy: if BerrySR budget is exceeded, disable BerrySR first, keep FSR active

Acceptance:

- FSR-only, BerrySR-only, and FSR+BerrySR combined paths all render correctly
- overload degrades through explicit policy, not through uncontrolled frame collapse
- compositing order is documented and consistent across Metal and Vulkan paths

---

## Workstream V6: `.pak` Executable Packaging Standard

**Why:** `.pak` is currently an asset archive. `v0.7.0` promotes it to the standard executable
program package format for Tileline.

### V6.1. Self-Contained `.pak` Executables

Target:

- `.pak` can contain:
  - embedded runtime binary (platform-native ELF/Mach-O/PE)
  - `.tlpfile` project manifest
  - all scene assets (`.tlscript`, `.tlsprite`, `.tljoint`, textures, meshes, audio)
  - BerrySR model weights (`.pak`-internal, referenced via MLS artifact binding)
  - GMS/MGS scaler profiles
- `.pak` header includes an `executable` flag and entry-point metadata
- runtime can boot directly from a `.pak` without external file references
- **default executable `.pak` name is `prima.pak`**; runtime looks for `prima.pak` in the working
  directory when no `--pak` argument is provided

Acceptance:

- `scripts/build_pak_executable.sh` produces a self-contained `.pak` from a project directory
- `tlapp --pak mygame.pak` boots the game without additional CLI arguments
- `tlapp` (no arguments) in a directory containing `prima.pak` boots from `prima.pak`
- `.pak` size is bounded by shard partitioning (existing 5GB default cap still applies)

### V6.2. `.pak` Manifest for Executable Entry Points

Target:

- `.pak` internal manifest (`pak://manifest.toml`) defines:
  - `entry_binary`: relative path to embedded runtime
  - `entry_scene`: default scene to load on boot
  - `entry_dimension`: `2d` or `3d`
  - `required_extensions`: `[berrysr]`, `[nps]`, etc.
  - `target_platforms`: `[linux-x86_64, macos-aarch64, android-aarch64]`
- runtime reads the manifest on boot and validates platform compatibility

Acceptance:

- `.pak` with executable manifest boots to the correct scene on first launch
- platform mismatch is detected early with a clear error message
- missing required extensions are reported before runtime initialization

### V6.3. Cross-Platform `.pak` Execution

Target:

- `.pak` is portable across platforms that share the same asset format
- binary shards are platform-specific but asset shards are shared
- `pak_tool` can list, verify, and repack executable `.pak` archives

Acceptance:

- `pak_tool list mygame.pak` shows binary shard + asset shard layout
- `pak_tool verify mygame.pak` checks checksums and manifest consistency
- repack preserves executable metadata

---

## Test Gates

### WGPU Exit Gate

- `cargo tree -p runtime | grep wgpu` → empty
- `cargo tree -p tl-core | grep wgpu` → empty
- runtime smoke test on macOS Metal passes
- runtime smoke test on Linux Vulkan passes
- runtime smoke test on Android MGS passes

### Parallel-by-Default Gate

- `cargo tree -p runtime | grep -E "rayon|bevy"` → empty
- ParadoxPE `8k` gate: all phases parallel by default
- ParadoxPE `30k` gate: serial tails are explicit, logged, and bounded
- MPS is the default scheduler in startup logs

### MGS Completion Gate

- `cargo test -p mgs` passes
- Android showcase `30+ FPS` on mobile-safe preset
- no GPU watchdog kills under stress

### BerrySR Gate

- NVIDIA Tensor Core path: inference latency < 8ms per frame at medium quality
- Apple Neural Engine path: inference latency < 12ms per frame at medium quality
- Rockchip NPU path: inference latency < 25ms per frame at low quality
- frame-gen output is visually coherent (no tearing, ghosting, or corruption)
- BerrySR disable/fallback does not crash the render loop

### `.pak` Executable Gate

- `scripts/build_pak_executable.sh` produces valid `.pak`
- `tlapp --pak <file>` boots without additional arguments
- `pak_tool verify` passes on executable `.pak`

---

## Scope Locks

- **WGPU exit is hard-gated** — no shipping crate may depend on `wgpu` in `v0.7.0`
- **BerrySR targets only NVIDIA Tensor Core, Apple Neural Engine, and Rockchip NPU** — AMD and
  Intel are out of scope for this release to keep validation bounded
- **BerrySR is complementary to FSR, not a replacement** — FSR spatial upscale remains the
  primary path; BerrySR adds temporal synthesis on top
- **BerrySR is disabled by default** — explicit opt-in required via CLI/script/config
- **MGS completion means Android/Mali/Adreno only** — iOS validation is not a `v0.7.0` gate
- **`.pak` executable standard covers ELF/Mach-O/PE** — WASM or web-target packaging is out of
  scope
- **parallel-by-default does not mean reckless parallelism** — serial fallback is still allowed
  when explicitly triggered, logged, and bounded

---

## Rollback / Degrade Behavior

If regressions are detected:

- `--scheduler=serial` forces serial physics execution for debugging
- `--berrysr=off` disables frame generation without affecting FSR
- `--gms-mode=conservative` reduces GPU aggressiveness without disabling GMS
- `--pak-legacy` falls back to loose-directory asset loading instead of `.pak` executable

---

## Cross-Release Alignment

| Release | Theme | Primary Deliverables |
|---------|-------|----------------------|
| `v0.5.0` | Heimdall Update | Render optimization, effects/textures, ParadoxPE + MPS revision, Rayon/Bevy/WGPU independence start |
| `v0.6.0` | MLS + Decentralized NPS | ML runtime stack, peer-mesh networking, NAT traversal |
| `v0.6.5` | GGUF + NVFP4 | Low-precision MLS execution, model format expansion |
| **`v0.7.0`** | **Ironclad** | **WGPU exit complete, GMS canonical, parallel-by-default, MGS complete, BerrySR, `.pak` executables** |
