# Tileline v0.6.0 Roadmap

Codename: `MLS Runtime Stack`

## Release Goal

`v0.6.0` promotes `MLS` (`Machine Learning Scaler`) into a first-class Tileline layer. MLS is
not treated as a side feature under GMS; it becomes a canonical runtime subsystem with shared
contracts across `tl-core`, `runtime`, `TLApp`, `.tlpfile`, `.tlscript`, and future native
vendor adapters.

Chosen defaults:

- delivery level: `inference + training`
- first workloads: `upscale + agent + physics_assist`
- rollout priority: `AMD/NVIDIA first`
- control surface: `.tlpfile + CLI + .tlscript`
- determinism rule: `physics_assist` remains advisory only
- memory model: `UMA-first unified RAM+VRAM`

## Workstream F1: MLS Core Contract + Packaging

Target:

- add canonical `tl-core::mls`
- define engine-owned backend, device, workload, precision, artifact, request, training, and
  telemetry types
- reuse `.pak` as the default shipping model container in `v0.6.0`
- make MLS memory ownership **unified-memory first** instead of PyTorch-style hard host/device
  separation where possible
- fail soft on unsupported backends, missing models, and training-unavailable adapters

Acceptance:

- `tl_core::mls` exports stable public types
- `MlsRuntime` resolves a backend without crashing when native adapters are unavailable
- `.pak` references and loose model paths use one shared artifact binding surface
- MLS runtime exposes one canonical memory model where host RAM + device-visible VRAM/UMA are
  treated as one engine-owned residency problem rather than separate user-managed pools

## Workstream F2: Native Backend Adapters

Target:

- add native adapter scaffolding for:
  - AMD (`ROCm/HIP + rocBLAS/MIOpen` direction)
  - NVIDIA (`CUDA + cuBLAS + TensorRT` direction)
  - Apple (`Core ML` first, `Metal/MPSGraph` fallback)
  - Rockchip (`RKNN` inference path)
- keep adapter policy on one contract with fail-soft behavior
- do not make Vulkan cooperative matrices the primary MLS path in `v0.6.0`
- backend adapters must report whether they are operating in:
  - true UMA / unified memory
  - staged discrete VRAM path
  - fail-soft CPU fallback

Acceptance:

- adapter inventory resolves through one MLS contract
- AMD/NVIDIA remain the first hard-gate backends for release validation
- Apple/Rockchip initialize contract state cleanly and degrade without crashing when unsupported
- unified-memory capable devices (Apple Silicon / shared-memory APUs / similar UMA-class targets)
  do not require a separate user-visible host/device memory management model

## Workstream F3: Upscale + Graphics MLS Path

Target:

- treat `upscale` as the first engine-owned MLS graphics workload
- route its controls through runtime rather than example-only glue
- make overload degrade cleanly back to non-MLS render behavior

Acceptance:

- MLS upscale state is inspectable from runtime telemetry
- overload clamps without uncontrolled frame collapse
- fallback path is explicit and reported

## Workstream F4: Agent Inference + Runtime Hooks

Target:

- expose agent-oriented inference control through runtime and `.tlscript`
- keep model/session binding engine-owned rather than ad hoc example code
- support local model binding via `.pak` or loose paths

Acceptance:

- `mls.model.bind` and script-side `mls_bind_model()` drive the same runtime surface
- agent inference can be toggled and budgeted through CLI/script/config
- no bespoke example-only binding path is required

## Workstream F5: Physics-Assist Advisory Path

Target:

- add `physics_assist` as an advisory MLS workload
- route only hints into ParadoxPE-facing runtime logic
- never let MLS mutate authoritative fixed-step world state directly

Acceptance:

- physics assist can propose hints or budgets
- fixed-step world state remains CPU/physics authoritative
- disable/fallback is explicit and deterministic-safe

## Workstream F6: Local Training + Checkpointing

Target:

- support local single-device train-step execution in `v0.6.0`
- add checkpoint save/load surface for MLS sessions
- keep distributed/cloud training out of scope for this release

Acceptance:

- `mls.train <slot> <steps>` works through the canonical runtime surface
- checkpoint requests fail soft when unsupported on the active backend
- training remains optional and explicitly gated

## Workstream F7: Telemetry + CLI + `.tlscript` + `.tlpfile`

Target:

- extend runtime bridge telemetry with MLS state:
  - `mls_backend`
  - `mls_device`
  - `mls_memory_mode`
  - `mls_uma_active`
  - `mls_active_workloads`
  - `mls_infer_queue_depth`
  - `mls_train_queue_depth`
  - `mls_drop_rate`
  - `mls_fallback_reason`
  - `mls_precision`
  - `mls_step_time_ms`
- extend `.tlpfile` with `[mls]`
- extend TLApp CLI with `mls.*`
- extend `.tlscript` with `mls_*` built-ins
- keep override precedence fixed:
  1. CLI
  2. `.tlscript`
  3. `.tlpfile`

Acceptance:

- `mls.status` works in TLApp
- script-side MLS built-ins compile and emit runtime overrides
- `.tlpfile [mls]` parses and feeds runtime defaults
- telemetry values are visible in runtime status output and script metrics
- runtime telemetry makes the unified-memory path explicit instead of hiding whether the workload
  is UMA-native, staged-to-VRAM, or CPU fallback

## Test Gates

- unit tests:
  - backend capability detection / fallback mapping
  - `.tlpfile [mls]` parsing
  - `.tlscript` MLS built-ins and precedence behavior
  - model binding resolution from `.pak` and loose paths
- integration tests:
  - `auto` backend selection with mock/native inventories
  - Apple/Rockchip fail-soft behavior
  - runtime override precedence (`CLI > script > tlpfile`)
- workload gates:
  - `upscale` degrades cleanly to non-MLS render path
  - `agent` inference uses canonical runtime content path
  - `physics_assist` never mutates authoritative world state directly
  - `training` can step, checkpoint, and resume in local mode

## Scope Locks

- `v0.6.0` training means `local single-device training/finetune`
- MLS reuses `.pak` instead of introducing a separate model-container format
- `physics_assist` is advisory only in this release
- AMD/NVIDIA are the first hard validation targets
- `v0.6.0` MLS is **not** planned around a PyTorch-like split between system RAM and VRAM; the
  engine contract should present unified residency semantics and hide staging details behind the
  backend
- Apple Neural Engine and Rockchip NPU support land on the same contract and can continue deeper
  hardware validation in `v0.6.x`
