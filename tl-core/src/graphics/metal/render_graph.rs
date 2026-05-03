//! Lightweight render-graph abstraction for Metal.
//!
//! Phase-0 skeleton: sequential pass encoding.  Phase 2 will add parallel
//! `MTLParallelRenderCommandEncoder` dispatch and async command-buffer pools.

#![cfg(target_os = "macos")]

use metal::{MTLClearColor, MTLLoadAction, MTLStoreAction, RenderPassDescriptor};

use super::resource_manager::ResourceManager;

/// Reference to a color or depth attachment inside the graph.
#[derive(Debug, Clone)]
pub enum AttachmentRef {
    /// The `CAMetalLayer` drawable texture (resolved at encode time).
    Drawable,
    /// A named texture owned by `ResourceManager`.
    Texture(String),
}

/// One render pass node in the graph.
pub struct RenderPassNode {
    pub label: &'static str,
    pub color_attachments: Vec<(AttachmentRef, ColorLoadStore)>,
    pub depth_attachment: Option<(AttachmentRef, DepthLoadStore)>,
    /// Closure that records commands into the encoder.
    /// `FnMut` so the same node can be re-encoded across frames.
    pub encode_fn: Box<dyn FnMut(&metal::RenderCommandEncoderRef) + Send>,
}

/// How to load/store a color attachment.
#[derive(Debug, Clone, Copy)]
pub struct ColorLoadStore {
    pub load: MTLLoadAction,
    pub store: MTLStoreAction,
    pub clear: Option<MTLClearColor>,
}

/// How to load/store a depth attachment.
#[derive(Debug, Clone, Copy)]
pub struct DepthLoadStore {
    pub load: MTLLoadAction,
    pub store: MTLStoreAction,
    pub clear_depth: f64,
}

/// A simple render graph that executes passes in insertion order.
///
/// In Phase 2 this will be upgraded with topological sorting,
/// `MTLParallelRenderCommandEncoder` sub-passes, and parallel command-buffer
/// submission.
pub struct RenderGraph {
    passes: Vec<RenderPassNode>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add_pass(&mut self, pass: RenderPassNode) {
        self.passes.push(pass);
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Encode every pass into the supplied command buffer.
    ///
    /// `drawable_texture` is the current `CAMetalLayer` drawable (only valid
    /// inside the frame callback).
    pub fn encode(
        &mut self,
        command_buffer: &metal::CommandBufferRef,
        drawable_texture: Option<&metal::TextureRef>,
        resources: &ResourceManager,
    ) {
        for pass in &mut self.passes {
            let desc = RenderPassDescriptor::new();

            for (slot, (attachment, ops)) in pass.color_attachments.iter().enumerate() {
                let color = desc.color_attachments().object_at(slot as u64).unwrap();
                match attachment {
                    AttachmentRef::Drawable => {
                        if let Some(tex) = drawable_texture {
                            color.set_texture(Some(tex));
                        }
                    }
                    AttachmentRef::Texture(name) => {
                        if let Some(tex) = resources.texture(name) {
                            color.set_texture(Some(tex));
                        }
                    }
                }
                color.set_load_action(ops.load);
                color.set_store_action(ops.store);
                if let Some(c) = ops.clear {
                    color.set_clear_color(c);
                }
            }

            if let Some((attachment, ops)) = &pass.depth_attachment {
                let depth = desc.depth_attachment().unwrap();
                match attachment {
                    AttachmentRef::Drawable => {
                        if let Some(tex) = drawable_texture {
                            depth.set_texture(Some(tex));
                        }
                    }
                    AttachmentRef::Texture(name) => {
                        if let Some(tex) = resources.texture(name) {
                            depth.set_texture(Some(tex));
                        }
                    }
                }
                depth.set_load_action(ops.load);
                depth.set_store_action(ops.store);
                depth.set_clear_depth(ops.clear_depth);
            }

            let encoder = command_buffer.new_render_command_encoder(&desc);
            (pass.encode_fn)(encoder);
            encoder.end_encoding();
        }
    }
}
