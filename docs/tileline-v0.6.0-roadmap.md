# Tileline v0.6.0 Roadmap

Codename: `MLS Runtime Stack + Decentralized NPS`

## Release Goal

`v0.6.0` promotes both `MLS` (`Machine Learning Scaler`) and `NPS` (`Network Packet Scaler`)
into first-class Tileline subsystems.

- **MLS** becomes the canonical ML runtime stack with inference, training, and engine-owned
  backend contracts across `tl-core`, `runtime`, `TLApp`, `.tlpfile`, `.tlscript`, and native
  vendor adapters.
- **NPS** evolves from basic UDP transport into a **decentralized connectivity layer** with
  peer-mesh topology, NAT traversal, authenticated sessions, delta-sync, and peer discovery.

Decentralized connectivity is the primary NPS priority for this release. The goal is to make
Tileline multiplayer work without a hard dependency on a central dedicated server for
session establishment or steady-state snapshot relay.

Chosen defaults:

- MLS delivery level: `inference + training`
- MLS first workloads: `upscale + agent + physics_assist`
- MLS rollout priority: `AMD/NVIDIA first`
- NPS topology default: `PeerMesh` with deterministic bounded fanout
- NPS session model: `cryptographic identity + STUN/ICE bootstrap + TURN relay fallback`
- Control surface for both: `.tlpfile + CLI + .tlscript`
- MLS determinism rule: `physics_assist` remains advisory only
- MLS memory model: `UMA-first unified RAM+VRAM`

---

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

---

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

---

## Workstream F3: Upscale + Graphics MLS Path

Target:

- treat `upscale` as the first engine-owned MLS graphics workload
- route its controls through runtime rather than example-only glue
- make overload degrade cleanly back to non-MLS render behavior

Acceptance:

- MLS upscale state is inspectable from runtime telemetry
- overload clamps without uncontrolled frame collapse
- fallback path is explicit and reported

---

## Workstream F4: Agent Inference + Runtime Hooks

Target:

- expose agent-oriented inference control through runtime and `.tlscript`
- keep model/session binding engine-owned rather than ad hoc example code
- support local model binding via `.pak` or loose paths

Acceptance:

- `mls.model.bind` and script-side `mls_bind_model()` drive the same runtime surface
- agent inference can be toggled and budgeted through CLI/script/config
- no bespoke example-only binding path is required

---

## Workstream F5: Physics-Assist Advisory Path

Target:

- add `physics_assist` as an advisory MLS workload
- route only hints into ParadoxPE-facing runtime logic
- never let MLS mutate authoritative fixed-step world state directly

Acceptance:

- physics assist can propose hints or budgets
- fixed-step world state remains CPU/physics authoritative
- disable/fallback is explicit and deterministic-safe

---

## Workstream F6: Local Training + Checkpointing

Target:

- support local single-device train-step execution in `v0.6.0`
- add checkpoint save/load surface for MLS sessions
- keep distributed/cloud training out of scope for this release

Acceptance:

- `mls.train <slot> <steps>` works through the canonical runtime surface
- checkpoint requests fail soft when unsupported on the active backend
- training remains optional and explicitly gated

---

## Workstream F7: MLS Telemetry + CLI + `.tlscript` + `.tlpfile`

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

---

## Workstream N1: Decentralized Topology + Peer Mesh Core

Target:

- promote `PeerMesh` from foundation primitive to default runtime topology
- implement bounded deterministic fanout with configurable `MeshFanoutConfig`
- add RTT/loss-aware peer scoring for relay and snapshot target selection
- support hybrid `ListenHost` fallback for sessions that prefer a well-known entry point
- keep `ClientServer` as an explicit but non-default topology option

Acceptance:

- `PeerMesh` is the default when no dedicated server is configured
- snapshot routing uses scored peer selection instead of naive full broadcast
- topology metrics (`snapshot_skipped_topology`, `last_snapshot_target_peers`) are exposed in
  telemetry
- deterministic fanout bounds do not explode under high peer counts

---

## Workstream N2: NAT Traversal + P2P Handshake

Target:

- integrate STUN/ICE for NAT hole punching between peers
- add TURN relay fallback when direct P2P is impossible
- keep relay usage explicit in telemetry and CLI (`nps.relay.status`)
- implement lightweight session handshake after ICE candidate negotiation
- bootstrap packets (`BootstrapHello` / `BootstrapWelcome`) carry ICE metadata

Acceptance:

- two peers behind typical home NATs can establish direct UDP without a dedicated server
- TURN fallback activates automatically and logs the relay path
- ICE negotiation does not block the simulation tick loop
- handshake completes within a bounded number of bootstrap rounds

---

## Workstream N3: Peer Discovery + Identity

Target:

- add decentralized peer discovery (DHT or mDNS-based, engine-configurable)
- bind peer identity to cryptographic key pairs (Ed25519 or similar)
- expose `PeerIdentity` in `nps/src/model.rs` with stable `peer_id` derivation from public key
- keep discovery fail-soft: if no DHT/mDNS peers are found, CLI bootstrap addresses still work

Acceptance:

- peers can discover each other without a central matchmaking server
- peer IDs are stable across reconnects and NAT address changes
- discovery does not leak sensitive topology info in untrusted LANs
- mDNS variant works for LAN party scenarios without external infrastructure

---

## Workstream N4: Authenticated Session Lifecycle

Target:

- harden `Connecting -> Negotiating -> Ready -> Disconnecting` state machine
- add per-session cryptographic authentication token derived from peer identity
- sign bootstrap packets to prevent spoofed `BootstrapHello` injection
- add session timeout and keepalive with explicit disconnect reasons
- keep `NetworkTransportRuntime` session table thread-safe without heavy locks

Acceptance:

- spoofed bootstrap packets are rejected at the transport layer
- session tokens rotate safely on reconnect
- disconnect reasons propagate to `.tlscript` via event channel
- session table remains O(1) for the hot read path during snapshot emission

---

## Workstream N5: Bandwidth Budgeting + Delta Sync

Target:

- add per-channel bandwidth budgets enforced by `NetworkPacketManager`
- implement delta-compression for `PhysicsState` snapshots (`TransformBatch` deltas)
- bind `.tlscript` `@net(sync="on_change")` metadata to delta emission decisions
- add snapshot downsampling/divisor when bandwidth budget is exceeded
- keep full keyframe snapshots every `N` ticks for delta recovery

Acceptance:

- bandwidth limits are configurable per topology and per lane
- delta snapshots reduce bytes-per-tick under low-churn scenes
- dropped delta bursts recover automatically at the next keyframe
- `@net(sync="on_change")` functions emit deltas instead of full state

---

## Workstream N6: Snapshot Interpolation + Rollback

Target:

- add `PhysicsInterpolationBuffer` for received remote snapshots
- implement snapshot interpolation for visual smoothing of remote peer state
- add rollback hooks in ParadoxPE for authority correction on misprediction
- keep rollback bounded: max rewind window is a configurable tick count
- expose interpolation quality to telemetry (buffer depth, stall events)

Acceptance:

- remote peer bodies render smoothly without visible teleportation
- rollback rewinds only authoritative bodies owned by the correcting peer
- rewind window is bounded and does not grow unboundedly under packet loss
- interpolation buffer recovers from transient packet loss without permanent stalls

---

## Workstream N7: `.tlscript @net()` Runtime Binding

Target:

- map `.tlscript` `@net(...)` decorators to canonical NPS lanes at runtime
- emit `LifecycleEvent` packets for `@net(reliable)` functions
- emit `PhysicsState` / `Input` packets for `@net(unreliable, domain="bodies")`
- add script-side event queue for incoming network events (spawn, authority transfer, state sync)
- keep binding one-directional: script declares intent, runtime enforces lane semantics

Acceptance:

- `@net(sync="on_change")` variables emit delta packets on mutation
- `@net(unreliable)` functions map to `UnreliableSequenced` policy
- `@net(reliable)` functions map to `ReliableOrdered` policy
- incoming network events are observable from `.tlscript` without polling the socket directly

---

## Workstream N8: NPS Telemetry + Control Surface

Target:

- extend runtime bridge telemetry with NPS state:
  - `nps_topology`
  - `nps_peer_count`
  - `nps_relay_active`
  - `nps_rtt_ms`
  - `nps_jitter_ms`
  - `nps_loss_estimate`
  - `nps_bandwidth_used_bps`
  - `nps_snapshot_bytes_per_sec`
  - `nps_decode_failures`
- extend `.tlpfile` with `[nps]`
- extend TLApp CLI with `nps.*`
- extend `.tlscript` with `nps_*` built-ins
- add per-peer HUD meters alongside existing network health bar

Acceptance:

- `nps.status` works in TLApp
- script-side NPS built-ins compile and emit runtime overrides
- `.tlpfile [nps]` parses and feeds runtime defaults
- per-peer RTT/jitter/loss is visible in HUD and console
- telemetry distinguishes direct P2P from relay paths

---

## Test Gates

### MLS

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

### NPS

- unit tests:
  - ICE candidate parsing and priority sorting
  - peer scoring with synthetic RTT/loss inputs
  - delta compression roundtrip (`TransformBatch` -> delta -> reconstruct)
  - session state machine transitions and timeout handling
  - `@net(...)` metadata to lane policy mapping
- integration tests:
  - loopback `PeerMesh` with 2–4 peers (localhost, no NAT)
  - simulated NAT scenario with TURN fallback (using CLI relay override)
  - snapshot interpolation buffer under synthetic packet loss
  - runtime override precedence (`CLI > script > tlpfile`)
- workload gates:
  - `PeerMesh` fanout stays bounded at 16+ peers
  - delta sync reduces bandwidth vs full snapshots under low churn
  - rollback rewind window stays within configured bound under loss
  - relay fallback does not crash or deadlock the transport loop

---

## Scope Locks

### MLS

- `v0.6.0` training means `local single-device training/finetune`
- MLS reuses `.pak` instead of introducing a separate model-container format
- `physics_assist` is advisory only in this release
- AMD/NVIDIA are the first hard validation targets
- `v0.6.0` MLS is **not** planned around a PyTorch-like split between system RAM and VRAM; the
  engine contract should present unified residency semantics and hide staging details behind the
  backend
- Apple Neural Engine and Rockchip NPU support land on the same contract and can continue deeper
  hardware validation in `v0.6.x`

### NPS

- `v0.6.0` NPS focus is **decentralized connectivity**, not centralized dedicated-server hosting
- encryption/auth hardening is session-level; full end-to-end encrypted mesh is out of scope
- voice/chat stack is out of scope for `v0.6.0`
- generalized replication for every engine subsystem is out of scope; NPS targets physics/input
  and `.tlscript @net()` lanes only
- rollback polish beyond snapshot interpolation + bounded rewind is deferred to `v0.6.x`
- peer discovery defaults to mDNS for LAN and a lightweight DHT scaffold for WAN; full
  production-grade DHT hardening is `v0.6.x`

---

## Rollback / Degrade Behavior

If NPS regressions are detected:

- `--nps-topology=client_server` forces central-server mode and bypasses peer-mesh logic
- `--nps-discovery=off` disables mDNS/DHT and falls back to CLI address list
- `--nps-relay=forced` bypasses ICE and routes everything through TURN for debug stability
- `--nps-delta=off` disables delta compression and falls back to full quantized snapshots

If MLS regressions are detected:

- `--mls-mode=off` disables all MLS workloads and reverts to non-MLS render/physics paths
- `--mls-backend=cpu` forces CPU fallback without touching GPU state
