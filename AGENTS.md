# Tileline Agent Guidelines (`AGENTS.md`)

Welcome to the **Tileline** repository! If you are an AI Assistant, Agent, or an automated workflow operating on this codebase, you **MUST** read and strictly adhere to the rules outlined in this document before writing any code.

## 1. Project Scope & Architecture
This specific repository is the **root engine project** and will ultimately house ONLY the core foundational layers:
- **`tl-core`**: The orchestration and integration boundary. Contains the core data structures, bridge logic, and core abstractions.
- **`runtime`**: The actual render-loop, application entry point (`tlapp`), and windowing system integration.

> **Note:** Systems like **GMS (Graphics Multi Scaler)**, **MPS (CPU Scheduling / WASM)**, and **ParadoxPE (Physics Engine)** are intended to be independent, modular projects. Do NOT tightly couple `tl-core` or `runtime` to these external modules in a way that creates circular dependencies.

## 2. Strict Architectural Rules
- **No Circular Dependencies:** `runtime` depends on `tl-core`. `tl-core` does NOT depend on `runtime`. External crates (like `gms` or `ParadoxPE`) should be treated as external dependencies or submodules.
- **Modularity:** Design APIs so that other users can easily extract and use modules like ParadoxPE in completely different projects.
- **Ask Before Adding Crates:** Never add heavy external dependencies to `Cargo.toml` without explicitly asking the user first.

## 3. Performance & Memory Rules (CRITICAL)
Tileline is built for extreme performance, predictable frame-times, and high tick rates.
- **Zero Allocations in the Hot Path:** Do NOT use heap allocations (`Vec::new()`, `Box::new()`, `String::new()`, etc.) inside the render loop, update loops, or physics ticks unless absolutely necessary and approved. Use object pools, static arrays, or pre-allocated buffers.
- **Tick Rate vs. FPS:** Understand the difference between the physics/engine tick rate and the rendering FPS. They run asynchronously. Do not bind logic updates to frame rendering.
- **Bypass rendering:** Features like `--no-render` completely disable GPU workloads. Any engine code must survive and run perfectly even when no rendering backend is active. 

## 4. Code Editing Guidelines
- **Precision:** Use specific file replacement/editing tools. Do NOT replace entire files for a 2-line change. 
- **Preserve Existing Logic:** Tileline has complex mathematical and hardware-specific optimizations (like Apple Silicon / UMA adaptive buffers, SXRC compression). If you don't understand a block of code, **ask**. Do not accidentally delete or "clean up" seemingly unused complex heuristic math.
- **Logging:** When editing logs (especially `tlapp fps`), maintain the dense, single-line format (`inst: 118.9 | ema: ...`). If a subsystem like the GPU is disabled, output a clear `[Render disabled]` message instead of letting systems spam errors.

## 5. Communication with the User
- Tileline is a heavily managed project. Focus exactly on what the user asks. 
- If a user says "Don't debug this, we'll do it later", stop your debugging loop immediately.
- Suggest architectural improvements, but wait for explicit "OK" before rewriting fundamental systems.

---
*By reading this, you acknowledge the high-performance constraints of the Tileline architecture. Keep your code fast, safe, and modular.*
