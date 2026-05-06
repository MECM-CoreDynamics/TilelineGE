# Tileline v0.8.0 Roadmap

Codename: `Artisan`

## Release Goal

`v0.8.0` is the **last major release before v1.0 maturity**. It is where Tileline transitions from
an engine/runtime core into a **complete, authorable platform** with first-class editor support,
modern input ergonomics, radical UI/UX overhaul, and multi-platform kernel portability.

Key themes:

- **Editor** — a native, runtime-integrated editor for scenes, scripts, sprites, and materials
- **Controller / gamepad as first-class input** — no longer keyboard-and-mouse-first
- **UI/UX revolution** — HXNU-native UI layer replaces ad-hoc immediate-mode overlays
- **HXNU kernel support** — Tileline runs natively on the HXNU kernel, not just Linux/macOS
- **Partial ownership crates** — open-core / commercial-layer crate split begins
- **v0.8.x LTS** — the `v0.8.x` line becomes the stability bridge to `v1.0`

`v0.8.0` is intentionally **feature-heavy** because `v1.0` must be a maturity freeze, not a
surprise delivery.

---

## Workstream E1: Tileline Editor

**Why:** Tileline cannot reach v1.0 maturity without a native editor. Bespoke example glue and
hand-edited text files are not a production content pipeline.

### E1.1. Editor Core Architecture

Target:

- build the editor as a **runtime-integrated** tool, not a separate application
- the editor is a special `.tlscript` / `.tlsprite` / `.tljoint` editing mode inside TLApp
- activate via CLI (`--editor`), `.tlpfile` flag (`[project] editor = true`), or in-app hotkey
- editor scene graph mirrors the runtime scene graph; edits are live-previewed
- editor UI is built on the HXNU UI layer (see E3), not `egui` or immediate-mode fallback

Acceptance:

- `tlapp --editor` boots into editor mode on Metal, Vulkan, and HXNU paths
- editor does not require a separate build target or asset tree
- editor state is serializable to `.tlpfile` / `.tljoint` without data loss

### E1.2. Scene Editor

Target:

- viewport camera: fly, orbit, and first-person modes
- entity selection, transform gizmos (translate, rotate, scale)
- hierarchy panel with parent/child relationships
- property inspector for transforms, materials, lights, physics bodies
- undo/redo stack with deterministic replay

Acceptance:

- a scene can be opened, edited, and saved without exiting TLApp
- undo/redo works across transform, spawn, and despawn operations
- live preview shows changes in real time without requiring a manual refresh

### E1.3. Script / Sprite / Material Editors

Target:

- `.tlscript` editor with syntax highlighting, error diagnostics, and live eval
- `.tlsprite` editor with visual tile/slot layout and hot-reload preview
- material editor with node-based or property-sheet layout
- all editors support hot-reload: changes are reflected in the running scene immediately

Acceptance:

- editing a `.tlscript` file updates the running scene without restart
- `.tlsprite` texture slot changes appear in the viewport within one frame
- material edits are visible immediately on all affected meshes

### E1.4. Editor Stability and Performance

Target:

- editor mode does not degrade runtime performance by more than 15 %
- editor UI does not block the simulation tick loop
- editor state autosave every 30 seconds with crash recovery
- editor can be exited cleanly back to runtime play mode without restart

Acceptance:

- `8k` scene in editor mode stays above `50 FPS` on desktop gate
- autosave does not cause frame spikes
- crash recovery restores unsaved changes on next boot

---

## Workstream E2: Controller / Gamepad First-Class Support

**Why:** Tileline targets sandbox and multiplayer experiences where keyboard-and-mouse is not the
only input model. Controllers must be equal citizens.

### E2.1. Input Abstraction Refactor

Target:

- replace keyboard-and-mouse-first input model with a unified `InputDevice` abstraction
- `InputDevice` variants: `KeyboardMouse`, `Gamepad`, `Touch`, `Motion`
- all game actions (move, look, interact, menu, pause) map to logical actions, not raw keys
- action mapping is configurable per device and per game profile

Acceptance:

- `InputDevice` enum exists and is used by all runtime input consumers
- no hardcoded `KeyCode::W` or `MouseButton::Left` in gameplay logic
- action mapping is serializable to `.tlpfile`

### E2.2. Gamepad Integration

Target:

- support SDL2 / Gilrs / platform-native gamepad APIs on Linux, macOS, and HXNU
- support XInput on Windows (if Windows port exists)
- support DualSense / Xbox / Switch Pro controller layouts out of the box
- rumble / haptic feedback API (advisory; actual haptic execution is backend-dependent)
- controller hot-plug: connect/disconnect is handled gracefully without crashing

Acceptance:

- gamepad input works on Linux, macOS, and HXNU
- controller disconnect does not crash the runtime
- default mappings are sensible for FPS, sandbox, and platformer genres

### E2.3. Controller-Driven UI Navigation

Target:

- all editor and in-game UI is navigable via gamepad
- focus model: D-pad / left stick moves focus, A/Cross confirms, B/Circle cancels
- no feature is keyboard-mouse-only in shipping UI

Acceptance:

- editor can be fully operated with a gamepad
- in-game menus work with a gamepad without mouse fallback

---

## Workstream E3: UI/UX Revolution + HXNU Integration

**Why:** The current HUD/console path is functional but ad-hoc. `v0.8.0` replaces it with a
structured, HXNU-native UI layer that works across desktop, mobile, and embedded.

### E3.1. HXNU UI Layer Foundation

Target:

- integrate `HXNU` (https://github.com/berrycomp/hxnu) as a supported kernel platform
- Tileline runtime compiles and runs natively on HXNU, not through a POSIX compatibility layer
- HXNU UI subsystem provides the foundation for all Tileline UI (editor, HUD, menus, console)
- on HXNU, UI is rendered through the HXNU framebuffer / compositor path
- on Linux/macOS, HXNU UI layer is emulated through a compatibility backend

Acceptance:

- `tlapp` boots on HXNU under QEMU and real hardware
- UI renders correctly on HXNU framebuffer without `wgpu`, `Vulkan`, or `Metal`
- Linux/macOS compatibility backend passes the same UI tests

### E3.2. Structured UI Framework

Target:

- replace immediate-mode HUD sprites with a retained-mode UI scene graph
- UI elements: panels, buttons, sliders, text fields, viewports, scroll regions, tabs
- layout system: flexbox-like constraints with runtime theme support
- theming: light, dark, and high-contrast themes; customizable accent colors
- UI is authored in `.tlui` files (JSON/TOML-like declarative format)

Acceptance:

- `.tlui` files parse and produce a visible UI layout
- theme changes apply without restart
- UI layout adapts to 720p, 1080p, 1440p, and 4K without manual adjustment

### E3.3. In-App Console Revolution

Target:

- replace `Ctrl+F1` CLI overlay with a full terminal-style in-app console
- command history, tab completion, syntax highlighting for `.tlscript`
- multi-pane layout: command input, output log, live variable inspector
- console is usable with gamepad (E2.3) and keyboard

Acceptance:

- console opens in < 100 ms
- tab completion works for all built-in commands
- command history persists across sessions

---

## Workstream E4: Partial Ownership Crates (Open Core / Commercial Layer)

**Why:** Tileline needs a sustainable development model. `v0.8.0` begins the explicit crate-level
split between fully open-source components and commercially licensed extensions.

### E4.1. Crate Ownership Classification

Target:

- classify every crate into one of three tiers:
  - **Tier 1: Fully Open Source** (`tl-core`, `paradoxpe`, `mps`, `nps`, `gms`, `mgs`, `runtime`)
  - **Tier 2: Open Core** (source available, free to use, commercially licensed extensions exist)
  - **Tier 3: Commercial** (closed source, licensed per seat or per project)
- document the classification in `LICENSE-STRATEGY.md` with per-crate rationale
- Tier 1 crates remain under the existing project license with no change

Acceptance:

- `LICENSE-STRATEGY.md` lists every crate with its tier and license
- no Tier 1 crate depends on a Tier 3 crate (preventing accidental commercial lock-in)
- Tier 2 crates have explicit extension points documented

### E4.2. Tier 2 Crate Examples

Target:

- identify initial Tier 2 candidates:
  - `editor` (open core; advanced features like collaborative editing are commercial)
  - `berrysr` (open core; base inference free, high-quality LoRA packs commercial)
  - `hxnu_compat` (open core; basic compatibility free, advanced kernel integration commercial)
- Tier 2 crates build and pass tests without Tier 3 dependencies
- commercial extensions are loaded dynamically or gated behind compile-time flags

Acceptance:

- Tier 2 crates compile and run without Tier 3 code
- commercial extension API is documented and versioned
- users can build and ship games with Tier 1 + Tier 2 only

### E4.3. Contribution and Licensing Clarity

Target:

- `CONTRIBUTING.md` updated with tier-specific contribution rules
- CLA (Contributor License Agreement) for Tier 2 and Tier 3 contributions
- clear policy on what happens to community patches that touch Tier 2 boundaries

Acceptance:

- new contributors can determine their patch's tier from file paths alone
- CLA is documented and linked from PR templates

---

## Workstream E5: v0.8.x LTS Preparation

**Why:** `v0.8.x` is the last line before `v1.0`. It must be stable enough to serve as a long-term
platform for commercial projects that do not want to ride the bleeding edge.

### E5.1. API Freeze Planning

Target:

- declare a provisional API freeze date (target: 3 months after `v0.8.0`)
- `v0.8.x` patch releases fix bugs but do not add new public APIs
- deprecated APIs from `v0.5.x`–`v0.7.x` are removed in `v0.8.0`; no new deprecations in `v0.8.x`
- public API surface is documented with stability markers (`stable`, `unstable`, `deprecated`)

Acceptance:

- `cargo doc -p tl-core` shows stability markers on all public items
- no `#[deprecated]` item is added after `v0.8.0`
- `v0.8.1`, `v0.8.2`, etc. do not change public API signatures

### E5.2. Long-Term Support Policy

Target:

- `v0.8.x` receives security fixes and critical bug fixes for 12 months after `v1.0` release
- LTS branch: `release/v0.8.x`
- backport policy: only critical fixes (security, data loss, crash) are backported
- commercial licensees get extended LTS (18–24 months) as a Tier 3 benefit

Acceptance:

- `release/v0.8.x` branch exists and CI passes
- backport process is documented in `RELEASE.md`

### E5.3. v1.0 Migration Guide

Target:

- publish a migration guide from `v0.8.x` to `v1.0`
- document all breaking changes, removed APIs, and replacement paths
- provide automated migration scripts where possible

Acceptance:

- migration guide is complete before `v1.0` release candidate 1
- automated scripts pass on the `bounce_tank_showcase` example without manual intervention

---

## Test Gates

### Editor Gate

- editor boots on Metal, Vulkan, and HXNU
- scene can be opened, edited, saved, and reloaded without data loss
- undo/redo passes a 100-operation stress test
- editor performance degradation vs play mode is < 15 %

### Controller Gate

- gamepad input works on Linux, macOS, and HXNU
- controller hot-plug does not crash
- all UI is navigable via gamepad

### HXNU Gate

- `tlapp` boots on HXNU under QEMU
- UI renders on HXNU framebuffer
- physics and render pipelines run without POSIX compatibility layer

### UI/UX Revolution Gate

- `.tlui` files parse and render correctly
- theme switching works without restart
- console opens in < 100 ms with tab completion

### Partial Ownership Gate

- `LICENSE-STRATEGY.md` documents every crate tier
- Tier 1 crates build without Tier 3 dependencies
- Tier 2 crates build and pass tests without Tier 3 code

### LTS Gate

- `cargo doc` stability markers are present
- `release/v0.8.x` branch CI passes
- migration guide is complete

---

## Scope Locks

- **v0.8.0 is the last feature release before v1.0** — no new subsystems after `v0.8.0`
- **Editor is runtime-integrated, not a separate application** — no standalone editor binary
- **HXNU support is native, not POSIX emulation** — Linux/macOS remain primary, HXNU is a
  first-class third target
- **Tier 3 crates do not block Tier 1 builds** — open-source path must always compile without
  commercial code
- **UI revolution replaces HUD sprites, but HUD telemetry remains** — BerrySR/FSR/GMS meters are
  still visible; they are rendered through the new UI layer
- **Gamepad support does not remove keyboard-and-mouse** — all input devices remain supported

---

## Rollback / Degrade Behavior

If regressions are detected:

- `--input=keyboard_mouse` bypasses gamepad abstraction
- `--ui=legacy` falls back to sprite-based HUD instead of HXNU UI layer
- `--editor=off` disables editor mode and boots directly to play mode
- `--tier3=off` builds without Tier 3 commercial extensions

---

## Cross-Release Alignment

| Release | Theme | Primary Deliverables |
|---------|-------|----------------------|
| `v0.5.0` | Heimdall Update | Render optimization, effects/textures, ParadoxPE + MPS revision, Rayon/Bevy/WGPU independence start |
| `v0.6.0` | MLS + Decentralized NPS | ML runtime stack, peer-mesh networking, NAT traversal |
| `v0.6.5` | GGUF + NVFP4 | Low-precision MLS execution, model format expansion |
| `v0.7.0` | Ironclad | WGPU exit complete, GMS canonical, parallel-by-default, MGS complete, BerrySR, `.pak` executables |
| **`v0.8.0`** | **Artisan** | **Editor, controller support, UI/UX revolution, HXNU kernel support, partial ownership crates, LTS bridge** |
| `v1.0` | Maturity | API freeze, stability, long-term support |
