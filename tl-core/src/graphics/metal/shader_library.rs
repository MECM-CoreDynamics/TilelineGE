//! MSL shader compilation and render-pipeline state cache.

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use metal::{
    Device, Function, MTLBlendFactor, MTLBlendOperation, MTLPixelFormat, MTLVertexFormat,
    MTLVertexStepFunction, RenderPipelineColorAttachmentDescriptorRef,
    RenderPipelineDescriptor, RenderPipelineState, VertexDescriptor,
};

/// How the color attachment should blend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendMode {
    None,
    Alpha,
    Additive,
}

/// Unique key for a cached render pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub vertex_function: String,
    pub fragment_function: Option<String>,
    pub color_format: MTLPixelFormat,
    pub depth_format: MTLPixelFormat,
    pub sample_count: u32,
    pub blend_mode: BlendMode,
    pub vertex_layout_hash: u64,
}

/// Description of one vertex attribute inside a buffer layout.
#[derive(Debug, Clone, Copy)]
pub struct VertexAttributeDesc {
    pub format: MTLVertexFormat,
    pub offset: usize,
    pub buffer_index: usize,
}

/// Description of a single vertex buffer slot.
#[derive(Debug, Clone)]
pub struct VertexBufferLayoutDesc {
    pub stride: usize,
    pub step_function: MTLVertexStepFunction,
    pub attributes: Vec<VertexAttributeDesc>,
}

/// Full vertex-input layout for a pipeline.
#[derive(Debug, Clone)]
pub struct VertexLayout {
    pub buffer_layouts: Vec<VertexBufferLayoutDesc>,
}

impl VertexLayout {
    pub fn hash_key(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        for layout in &self.buffer_layouts {
            layout.stride.hash(&mut hasher);
            (layout.step_function as usize).hash(&mut hasher);
            for attr in &layout.attributes {
                (attr.format as usize).hash(&mut hasher);
                attr.offset.hash(&mut hasher);
                attr.buffer_index.hash(&mut hasher);
            }
        }
        hasher.finish()
    }
}

/// Owns a compiled `MTLLibrary` and caches `MTLRenderPipelineState`s.
pub struct ShaderLibrary {
    device: Device,
    library: metal::Library,
    pipeline_cache: HashMap<PipelineKey, RenderPipelineState>,
}

impl ShaderLibrary {
    pub fn new(device: &Device, source: &str) -> Result<Self, String> {
        let compile_opts = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(source, &compile_opts)
            .map_err(|e| format!("shader compilation failed: {e}"))?;
        Ok(Self {
            device: device.clone(),
            library,
            pipeline_cache: HashMap::new(),
        })
    }

    pub fn get_function(&self, name: &str) -> Result<Function, String> {
        self.library
            .get_function(name, None)
            .map_err(|e| format!("missing function '{name}': {e}"))
    }

    /// Obtain (or create) a cached pipeline for the given key + vertex layout.
    pub fn get_pipeline(
        &mut self,
        key: &PipelineKey,
        vertex_layout: &VertexLayout,
    ) -> Result<&RenderPipelineState, String> {
        if !self.pipeline_cache.contains_key(key) {
            let vert = self.get_function(&key.vertex_function)?;
            let frag = key
                .fragment_function
                .as_ref()
                .map(|f| self.get_function(f))
                .transpose()?;
            let pipeline = build_pipeline(&self.device, &vert, frag.as_ref(), key, vertex_layout)?;
            self.pipeline_cache.insert(key.clone(), pipeline);
        }
        Ok(self.pipeline_cache.get(key).unwrap())
    }
}

fn build_pipeline(
    device: &Device,
    vertex_function: &Function,
    fragment_function: Option<&Function>,
    key: &PipelineKey,
    vertex_layout: &VertexLayout,
) -> Result<RenderPipelineState, String> {
    let desc = RenderPipelineDescriptor::new();
    desc.set_vertex_function(Some(vertex_function));
    if let Some(frag) = fragment_function {
        desc.set_fragment_function(Some(frag));
    }

    let color_attachment = desc.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(key.color_format);
    configure_blend(color_attachment, key.blend_mode);

    desc.set_depth_attachment_pixel_format(key.depth_format);
    if key.sample_count > 1 {
        desc.set_sample_count(key.sample_count as u64);
    }

    let vertex_descriptor = VertexDescriptor::new();
    for (buffer_index, layout) in vertex_layout.buffer_layouts.iter().enumerate() {
        let buffer_layout_desc = vertex_descriptor.layouts().object_at(buffer_index as u64).unwrap();
        buffer_layout_desc.set_stride(layout.stride as u64);
        buffer_layout_desc.set_step_function(layout.step_function);
        for (attr_index, attr) in layout.attributes.iter().enumerate() {
            let attr_desc = vertex_descriptor.attributes().object_at(attr_index as u64).unwrap();
            attr_desc.set_format(attr.format);
            attr_desc.set_offset(attr.offset as u64);
            attr_desc.set_buffer_index(attr.buffer_index as u64);
        }
    }
    desc.set_vertex_descriptor(Some(&vertex_descriptor));

    device
        .new_render_pipeline_state(&desc)
        .map_err(|e| format!("pipeline creation failed: {e}"))
}

fn configure_blend(
    color_attachment: &RenderPipelineColorAttachmentDescriptorRef,
    mode: BlendMode,
) {
    match mode {
        BlendMode::None => {
            color_attachment.set_blending_enabled(false);
        }
        BlendMode::Alpha => {
            color_attachment.set_blending_enabled(true);
            color_attachment.set_rgb_blend_operation(MTLBlendOperation::Add);
            color_attachment.set_alpha_blend_operation(MTLBlendOperation::Add);
            color_attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
            color_attachment.set_source_alpha_blend_factor(MTLBlendFactor::SourceAlpha);
            color_attachment
                .set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
            color_attachment
                .set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        }
        BlendMode::Additive => {
            color_attachment.set_blending_enabled(true);
            color_attachment.set_rgb_blend_operation(MTLBlendOperation::Add);
            color_attachment.set_alpha_blend_operation(MTLBlendOperation::Add);
            color_attachment.set_source_rgb_blend_factor(MTLBlendFactor::One);
            color_attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
            color_attachment.set_destination_rgb_blend_factor(MTLBlendFactor::One);
            color_attachment.set_destination_alpha_blend_factor(MTLBlendFactor::One);
        }
    }
}
