//! Raw Metal renderer modules: render graph, resource manager, mesh slots, and shader library.

#![cfg(target_os = "macos")]

pub mod mesh_slot;
pub mod render_graph;
pub mod resource_manager;
pub mod shader_library;
