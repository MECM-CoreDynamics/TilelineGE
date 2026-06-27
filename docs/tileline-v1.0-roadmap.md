# Tileline v1.0 Roadmap

Codename: `Maturity`

## Release Goal

`v1.0` is the **first long-term stable release** of Tileline. It is not a feature release. It is a
**commitment**: the public API, the runtime behavior, and the platform support matrix are frozen
and guaranteed to remain stable for the supported lifetime of the release.

`v1.0` exists so that commercial projects, educational institutions, and modding communities can
build on Tileline without fear of breaking changes.

---

## What v1.0 Is

- **API freeze** — no breaking changes to public APIs in `tl-core`, `runtime`, `paradoxpe`, `mps`,
  `nps`, `gms`, or `mgs` for the supported lifetime
- **Behavioral stability** — physics determinism, render output, and network semantics are
  guaranteed to be consistent across patch releases
- **Platform guarantee** — supported platforms are documented and tested; new platforms may be
  added in `v1.x`, but existing platforms are not dropped
- **LTS foundation** — `v1.0.x` receives security fixes and critical bug fixes for a minimum of
  24 months

## What v1.0 Is Not

- **Not a marketing event** — `v1.0` does not imply "finished" or "no more work"; it implies
  "stable enough to build on"
- **Not a feature dump** — all major features land in `v0.8.x`; `v1.0` is the freeze that follows
- **Not the end of innovation** — `v1.1`, `v1.2`, etc. will add features, but they will not break
  `v1.0` APIs

---

## Definition Of Done

`v1.0` cannot ship until all of the following are true:

### API Stability

- every public type, function, and trait in Tier 1 crates has a stability marker
- no `#[deprecated]` item was added after `v0.8.0`
- `cargo doc` builds without warnings on all Tier 1 crates
- SemVer is enforced strictly: `v1.0.x` = patches only, `v1.x.0` = additions only, `v2.0.0` = breaks

### Test Coverage

- `cargo test --workspace` passes on all supported platforms
- integration tests cover:
  - `8k` and `30k` physics scenarios
  - Metal, Vulkan, and HXNU render paths
  - MGS mobile path on at least one Android device
  - NPS peer-mesh with 2–8 peers
  - BerrySR on NVIDIA Tensor Core and Apple Neural Engine
- no test is allowed to be `#[ignore]` without a linked issue and a target fix version

### Documentation

- every public API has rustdoc with at least one example
- `docs/` index is complete and cross-referenced
- migration guide from `v0.8.x` to `v1.0` is published and validated
- `README.md` explains how to get started, how to contribute, and how to license

### Platform Validation

| Platform | Backend | Minimum Gate |
|----------|---------|--------------|
| Linux x86_64 | Vulkan + GMS | `8k` @ 60 FPS, `30k` @ 30 FPS |
| macOS aarch64 | Metal + GMS | `8k` @ 60 FPS, `30k` @ 30 FPS |
| HXNU x86_64 | Vulkan + GMS | boots, renders, physics stable |
| Android aarch64 | MGS | `8k` @ 30 FPS on Mali-G710 / Adreno 740 |

### Licensing and Governance

- `LICENSE-STRATEGY.md` is current and legally reviewed
- Tier 1, Tier 2, and Tier 3 crate classifications are frozen
- CLA process is documented and enforced for Tier 2+ contributions
- commercial licensing portal is operational (if applicable)

---

## Release Lifecycle

### v1.0.0-rc1 → v1.0.0-rcN

- 4–6 week release candidate period
- only critical fixes merged
- community testing encouraged; reported issues block release

### v1.0.0

- tagged and announced
- binary artifacts published for all supported platforms
- `.pak` executable standard is the primary distribution format

### v1.0.x (LTS)

- security fixes and critical bug fixes only
- no new features, no API additions, no behavioral changes
- supported for 24 months minimum
- commercial licensees may receive extended support (36 months) as a Tier 3 benefit

### v1.x.0 (Feature Releases)

- additive changes only
- new features, new platforms, new backends
- no breaking changes to `v1.0` APIs
- examples: `v1.1.0` might add Windows support, `v1.2.0` might add VR backends

### v2.0.0 (Future Major Release)

- breaking changes allowed
- migration guide from `v1.x` to `v2.0` required
- no earlier than 24 months after `v1.0.0`

---

## Rollback / Long-Term Support

If a critical vulnerability or data-loss bug is discovered in `v1.0.x`:

- fix is developed on `main` and backported to `release/v1.0.x`
- patch release (`v1.0.x+1`) is issued within 72 hours for critical security issues
- patch release is issued within 14 days for critical stability issues
- users are notified via GitHub Security Advisories and the commercial licensee portal

---

## Cross-Release Alignment

| Release | Theme | Primary Deliverables |
|---------|-------|----------------------|
| `v0.5.0` | Heimdall Update | Render optimization, effects/textures, ParadoxPE + MPS revision, Rayon/Bevy/WGPU independence start |
| `v0.6.0` | MLS + Decentralized NPS | ML runtime stack, peer-mesh networking, NAT traversal |
| `v0.6.5` | GGUF + NVFP4 | Low-precision MLS execution, model format expansion |
| `v0.7.0` | Ironclad | WGPU exit complete, GMS canonical, parallel-by-default, MGS complete, BerrySR, `.pak` executables |
| `v0.8.0` | Artisan | Editor, controller support, UI/UX revolution, HXNU kernel support, partial ownership crates, LTS bridge |
| **`v1.0`** | **Maturity** | **API freeze, behavioral stability, platform guarantee, 24-month LTS, SemVer commitment** |
