# Tileline v0.6.5 Roadmap

Codename: `MLS Model Formats + Low-Precision Inference`

## Release Goal

`v0.6.5` extends the `MLS` foundation from `v0.6.0` with production-facing model-format and
low-precision execution support. The two headline additions are:

- `GGUF` model packaging / runtime loading
- `NVFP4`-class low-precision inference support on capable NVIDIA paths

This release is not meant to replace the `v0.6.0` MLS contract. It builds on that contract and
keeps the same unified engine-owned control surfaces: `tl-core`, `runtime`, `TLApp`, `.tlpfile`,
`.tlscript`, and `.pak`.

## Workstream G1: GGUF Runtime Support

Target:

- add `GGUF` as an engine-supported MLS model format
- support GGUF artifact binding through the canonical runtime path
- keep `.pak` as the outer shipping container when desired; `GGUF` is the model payload format,
  not a replacement for Tileline packaging

Acceptance:

- `mls.model.bind` can bind GGUF artifacts
- loose GGUF paths and `.pak#...` GGUF references resolve through one code path
- runtime diagnostics clearly distinguish `missing model`, `unsupported format`, and `backend
  unsupported`

## Workstream G2: NVFP4 / Low-Precision NVIDIA Path

Target:

- add `NVFP4`-class low-precision execution mode for supported NVIDIA backends
- keep `fp16`, `bf16`, `int8`, and `fp32` fallback behavior intact
- expose low-precision selection through the existing MLS precision/runtime controls rather than
  adding a parallel configuration surface

Acceptance:

- NVIDIA MLS backend reports whether NVFP4 is available
- unsupported devices degrade cleanly to existing precisions
- precision fallback is explicit in telemetry and console status

## Workstream G3: Format + Precision Telemetry

Target:

- extend MLS telemetry so runtime can report:
  - model format (`pak`, `gguf`, other future types)
  - active precision (`nvfp4`, `fp16`, `bf16`, `int8`, `fp32`)
  - fallback reason when requested precision/format is unsupported
- keep CLI / `.tlscript` / `.tlpfile` precedence aligned with the `v0.6.0` MLS rules

Acceptance:

- `mls.status` can show GGUF/NVFP4 state when active
- runtime logs and HUD telemetry make precision/format fallback visible
- no silent downgrade occurs when the requested precision is unavailable

## Test Gates

- unit tests:
  - GGUF artifact binding and validation
  - precision fallback mapping for NVIDIA backends
  - telemetry formatting for model-format + precision state
- integration tests:
  - GGUF model binding through runtime-owned surfaces
  - NVFP4 request on unsupported NVIDIA device -> deterministic fallback
  - `.tlpfile` / CLI / `.tlscript` precedence still holds
- stability gates:
  - no crash on unsupported GGUF payloads
  - no crash when NVFP4 is requested on non-NVIDIA backends
  - no silent stall when low-precision path initialization fails

## Scope Locks

- `v0.6.5` builds on the `v0.6.0` unified-memory MLS contract; it does not reintroduce a
  user-visible RAM/VRAM split
- `GGUF` is added as a supported model payload format, not as a replacement for `.pak`
- `NVFP4` is NVIDIA-specific and must always fail soft to another precision when unavailable
- AMD / Apple / Rockchip paths remain first-class MLS backends even when specific `NVFP4` support
  is not applicable
