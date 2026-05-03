//! macOS raw Metal backend with Z-prepass + Early-Z occlusion culling.
//!
//! Scope:
//! - single GPU only
//! - explicit command queue submission
//! - CPU-side snapshot ring with strict capacity guards
//! - two-pass rendering: Z-prepass (depth-only) + forward shading (equal-depth test)
//! - fail-soft telemetry surface for runtime integration

#![cfg(target_os = "macos")]

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use core_graphics_types::geometry::CGSize;
use metal::{
    foreign_types::ForeignType,
    Buffer, CommandQueue, CompileOptions, DepthStencilDescriptor, DepthStencilState, Device,
    Function, MTLClearColor, MTLCompareFunction, MTLIndexType, MTLLoadAction,
    MTLPixelFormat, MTLPrimitiveType, MTLResourceOptions, MTLStoreAction, MTLTextureUsage,
    MTLVertexFormat, MTLVertexStepFunction, MetalLayer, RenderPassDescriptor,
    RenderPipelineDescriptor, RenderPipelineState, Texture, TextureDescriptor,
};
use objc::{msg_send, sel, sel_impl};
use objc::runtime::Object;
use winit::dpi::PhysicalSize;
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::Window;

use crate::graphics::frame_snapshot::RenderStateSnapshot;

/// Runtime configuration for the raw Metal backend.
#[derive(Debug, Clone)]
pub struct MetalBackendConfig {
    /// Number of ring slots kept in-flight for snapshot uploads.
    pub frames_in_flight: usize,
    /// Maximum transform records accepted per frame.
    pub max_instances: usize,
    /// Enable Z-prepass + Early-Z occlusion culling.
    pub occlusion_culling_enabled: bool,
}

impl Default for MetalBackendConfig {
    fn default() -> Self {
        Self {
            frames_in_flight: 2,
            max_instances: 32_768,
            occlusion_culling_enabled: true,
        }
    }
}

/// Snapshot metadata exposed for debugging / telemetry.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalSnapshotSlotState {
    pub frame_id: u64,
    pub instance_count: u32,
    pub material_count: u32,
    pub texture_count: u32,
    pub light_count: u32,
    pub byte_len: usize,
}

/// Submit telemetry for one recorded frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalFrameSubmissionTelemetry {
    pub frame_slot: usize,
    pub frame_id: u64,
    pub instance_count: u32,
    pub material_count: u32,
    pub texture_count: u32,
    pub light_count: u32,
}

/// Per-frame command queue execution telemetry.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalFrameExecutionTelemetry {
    pub submission: MetalFrameSubmissionTelemetry,
    pub snapshot_state: MetalSnapshotSlotState,
    pub command_buffer_submitted: bool,
    pub presented: bool,
    pub primary_submission_serial: u64,
    pub prepass_draw_calls: u32,
    pub main_pass_draw_calls: u32,
    pub early_z_reject_estimate: u32,
}

#[derive(Debug)]
struct SnapshotSlot {
    instance_capacity: usize,
    material_capacity: usize,
    texture_capacity: usize,
    light_capacity: usize,
    last_state: MetalSnapshotSlotState,
}

/// Errors produced by the raw Metal backend.
#[derive(Debug)]
pub enum MetalBackendError {
    NoMetalDevice,
    InvalidConfig(&'static str),
    SnapshotCapacityExceeded { requested: usize, capacity: usize },
    SnapshotMaterialCapacityExceeded { requested: usize, capacity: usize },
    SnapshotTextureCapacityExceeded { requested: usize, capacity: usize },
    SnapshotLightCapacityExceeded { requested: usize, capacity: usize },
    ShaderCompilation(String),
    PipelineCreation(String),
}

impl Display for MetalBackendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoMetalDevice => write!(f, "no Metal system device found"),
            Self::InvalidConfig(message) => write!(f, "invalid Metal backend config: {message}"),
            Self::SnapshotCapacityExceeded { requested, capacity } => write!(
                f,
                "snapshot capacity exceeded: requested {requested} instances, capacity {capacity}"
            ),
            Self::SnapshotMaterialCapacityExceeded { requested, capacity } => write!(
                f,
                "snapshot material capacity exceeded: requested {requested} materials, capacity {capacity}"
            ),
            Self::SnapshotTextureCapacityExceeded { requested, capacity } => write!(
                f,
                "snapshot texture capacity exceeded: requested {requested} textures, capacity {capacity}"
            ),
            Self::SnapshotLightCapacityExceeded { requested, capacity } => write!(
                f,
                "snapshot light capacity exceeded: requested {requested} lights, capacity {capacity}"
            ),
            Self::ShaderCompilation(err) => write!(f, "Metal shader compilation failed: {err}"),
            Self::PipelineCreation(err) => write!(f, "Metal pipeline creation failed: {err}"),
        }
    }
}

impl Error for MetalBackendError {}

const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
};

struct ZPrepassOut {
    float4 position [[position]];
};

vertex ZPrepassOut z_prepass_vertex(
    VertexIn in [[stage_in]],
    constant float4x4 &view_proj [[buffer(1)]],
    constant float4x4 *model_matrices [[buffer(2)]],
    uint instance_id [[instance_id]]
) {
    ZPrepassOut out;
    float4 world_pos = model_matrices[instance_id] * float4(in.position, 1.0);
    out.position = view_proj * world_pos;
    return out;
}

fragment void z_prepass_fragment() {}

struct ForwardOut {
    float4 position [[position]];
    float3 world_normal;
    float2 uv;
};

vertex ForwardOut forward_vertex(
    VertexIn in [[stage_in]],
    constant float4x4 &view_proj [[buffer(1)]],
    constant float4x4 *model_matrices [[buffer(2)]],
    uint instance_id [[instance_id]]
) {
    ForwardOut out;
    float4 world_pos = model_matrices[instance_id] * float4(in.position, 1.0);
    out.position = view_proj * world_pos;
    out.world_normal = (model_matrices[instance_id] * float4(in.normal, 0.0)).xyz;
    out.uv = in.uv;
    return out;
}

fragment float4 forward_fragment(ForwardOut in [[stage_in]]) {
    float3 color = float3(0.5, 0.5, 0.5);
    float3 light_dir = normalize(float3(1.0, 1.0, 1.0));
    float ndotl = max(dot(normalize(in.world_normal), light_dir), 0.0);
    float3 lit = color * (0.2 + 0.8 * ndotl);
    return float4(lit, 1.0);
}
"#;

/// Raw Metal backend MVP used by runtime integration layers.
pub struct MetalBackend {
    config: MetalBackendConfig,
    _window: Option<Arc<Window>>,
    device: Device,
    command_queue: CommandQueue,
    device_name: String,
    frame_slots: Vec<SnapshotSlot>,
    current_frame_slot: usize,
    surface_size: PhysicalSize<u32>,
    primary_submission_serial: u64,
    layer: MetalLayer,
    depth_texture: Texture,
    z_prepass_pipeline: RenderPipelineState,
    forward_pipeline: RenderPipelineState,
    depth_state_write: DepthStencilState,
    depth_state_equal: DepthStencilState,
    vertex_buffer: Option<Buffer>,
    index_buffer: Option<Buffer>,
    transform_buffer: Option<Buffer>,
    view_proj_buffer: Option<Buffer>,
}

impl MetalBackend {
    /// Create a new raw Metal backend bound to the provided window.
    pub fn new(window: Arc<Window>, config: MetalBackendConfig) -> Result<Self, MetalBackendError> {
        if config.frames_in_flight == 0 {
            return Err(MetalBackendError::InvalidConfig(
                "frames_in_flight must be greater than zero",
            ));
        }
        if config.max_instances == 0 {
            return Err(MetalBackendError::InvalidConfig(
                "max_instances must be greater than zero",
            ));
        }

        let device = Device::system_default().ok_or(MetalBackendError::NoMetalDevice)?;
        let command_queue = device.new_command_queue();
        let device_name = device.name().to_owned();

        let max_materials = config.max_instances.max(256);
        let max_textures = config.max_instances.max(128);
        let max_lights = 64;
        let frame_slots = (0..config.frames_in_flight)
            .map(|_| SnapshotSlot {
                instance_capacity: config.max_instances,
                material_capacity: max_materials,
                texture_capacity: max_textures,
                light_capacity: max_lights,
                last_state: MetalSnapshotSlotState::default(),
            })
            .collect::<Vec<_>>();

        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .map_err(MetalBackendError::ShaderCompilation)?;

        let z_prepass_vertex = library
            .get_function("z_prepass_vertex", None)
            .map_err(MetalBackendError::ShaderCompilation)?;
        let z_prepass_fragment = library
            .get_function("z_prepass_fragment", None)
            .map_err(MetalBackendError::ShaderCompilation)?;
        let forward_vertex = library
            .get_function("forward_vertex", None)
            .map_err(MetalBackendError::ShaderCompilation)?;
        let forward_fragment = library
            .get_function("forward_fragment", None)
            .map_err(MetalBackendError::ShaderCompilation)?;

        let z_prepass_pipeline = build_pipeline(
            &device,
            &z_prepass_vertex,
            Some(&z_prepass_fragment),
            MTLPixelFormat::BGRA8Unorm,
            MTLPixelFormat::Depth32Float,
        )
        .map_err(MetalBackendError::PipelineCreation)?;

        let forward_pipeline = build_pipeline(
            &device,
            &forward_vertex,
            Some(&forward_fragment),
            MTLPixelFormat::BGRA8Unorm,
            MTLPixelFormat::Depth32Float,
        )
        .map_err(MetalBackendError::PipelineCreation)?;

        let depth_state_write = build_depth_state(&device, MTLCompareFunction::Less, true)
            .map_err(MetalBackendError::PipelineCreation)?;
        let depth_state_equal = build_depth_state(&device, MTLCompareFunction::Equal, false)
            .map_err(MetalBackendError::PipelineCreation)?;

        let surface_size = window.inner_size();
        let depth_texture = create_depth_texture(&device, surface_size.width, surface_size.height);

        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_drawable_size(CGSize::new(surface_size.width as f64, surface_size.height as f64));

        let view = {
            let handle = window
                .window_handle()
                .map_err(|_| MetalBackendError::InvalidConfig("failed to get window handle"))?;
            match handle.as_raw() {
                RawWindowHandle::AppKit(appkit) => appkit.ns_view.as_ptr() as *mut Object,
                _ => return Err(MetalBackendError::InvalidConfig("not a macOS window")),
            }
        };
        unsafe {
            let () = msg_send![view, setWantsLayer: true];
            let () = msg_send![view, setLayer: layer.as_ptr()];
        }

        let max_verts = config.max_instances.max(1) * 64 * 8;
        let max_indices = config.max_instances.max(1) * 64 * 12;
        let max_transforms = config.max_instances.max(1);

        let vertex_buffer = Some(device.new_buffer(
            (max_verts * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        ));
        let index_buffer = Some(device.new_buffer(
            (max_indices * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        ));
        let transform_buffer = Some(device.new_buffer(
            (max_transforms * std::mem::size_of::<[[f32; 4]; 4]>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        ));
        let view_proj_buffer = Some(device.new_buffer(
            std::mem::size_of::<[[f32; 4]; 4]>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        ));

        Ok(Self {
            config,
            _window: Some(Arc::clone(&window)),
            device,
            command_queue,
            device_name,
            frame_slots,
            current_frame_slot: 0,
            surface_size,
            primary_submission_serial: 0,
            layer,
            depth_texture,
            z_prepass_pipeline,
            forward_pipeline,
            depth_state_write,
            depth_state_equal,
            vertex_buffer,
            index_buffer,
            transform_buffer,
            view_proj_buffer,
        })
    }

    /// Report the active Metal device name.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Current frame extent in pixels.
    pub fn surface_size(&self) -> PhysicalSize<u32> {
        self.surface_size
    }

    /// Active backend config snapshot.
    pub fn config(&self) -> &MetalBackendConfig {
        &self.config
    }

    /// Update current surface size and recreate depth texture.
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) -> Result<(), MetalBackendError> {
        if new_size.width == 0 || new_size.height == 0 {
            return Ok(());
        }
        self.surface_size = new_size;
        self.depth_texture = create_depth_texture(&self.device, new_size.width, new_size.height);
        self.layer.set_drawable_size(CGSize::new(new_size.width as f64, new_size.height as f64));
        Ok(())
    }

    /// Write one render-visible snapshot into the selected frame slot.
    pub fn upload_state_snapshot(
        &mut self,
        frame_slot: usize,
        snapshot: RenderStateSnapshot<'_>,
    ) -> Result<MetalSnapshotSlotState, MetalBackendError> {
        let slot = self
            .frame_slots
            .get_mut(frame_slot)
            .ok_or(MetalBackendError::InvalidConfig("frame slot out of range"))?;

        if snapshot.transforms.len() > slot.instance_capacity {
            return Err(MetalBackendError::SnapshotCapacityExceeded {
                requested: snapshot.transforms.len(),
                capacity: slot.instance_capacity,
            });
        }
        if snapshot.materials.len() > slot.material_capacity {
            return Err(MetalBackendError::SnapshotMaterialCapacityExceeded {
                requested: snapshot.materials.len(),
                capacity: slot.material_capacity,
            });
        }
        if snapshot.textures.len() > slot.texture_capacity {
            return Err(MetalBackendError::SnapshotTextureCapacityExceeded {
                requested: snapshot.textures.len(),
                capacity: slot.texture_capacity,
            });
        }
        if snapshot.lights.len() > slot.light_capacity {
            return Err(MetalBackendError::SnapshotLightCapacityExceeded {
                requested: snapshot.lights.len(),
                capacity: slot.light_capacity,
            });
        }

        let byte_len = std::mem::size_of_val(snapshot.transforms)
            + std::mem::size_of_val(snapshot.materials)
            + std::mem::size_of_val(snapshot.textures)
            + std::mem::size_of_val(snapshot.lights);

        slot.last_state = MetalSnapshotSlotState {
            frame_id: snapshot.frame_id,
            instance_count: snapshot.transforms.len() as u32,
            material_count: snapshot.materials.len() as u32,
            texture_count: snapshot.textures.len() as u32,
            light_count: snapshot.lights.len() as u32,
            byte_len,
        };

        Ok(slot.last_state)
    }

    /// Submit one frame worth of work to the Metal command queue.
    pub fn render_n(
        &mut self,
        snapshot: RenderStateSnapshot<'_>,
    ) -> Result<MetalFrameExecutionTelemetry, MetalBackendError> {
        let frame_slot = self.current_frame_slot;
        let snapshot_state = self.upload_state_snapshot(frame_slot, snapshot)?;

        let drawable = self.layer.next_drawable();
        let command_buffer = self.command_queue.new_command_buffer();

        let mut prepass_draw_calls = 0u32;
        let mut main_pass_draw_calls = 0u32;
        let mut early_z_reject_estimate = 0u32;

        // Upload vertex / index / transform / view-proj data if provided
        if let (Some(vb), Some(ib), Some(tb), Some(vpb)) = (
            self.vertex_buffer.as_ref(),
            self.index_buffer.as_ref(),
            self.transform_buffer.as_ref(),
            self.view_proj_buffer.as_ref(),
        ) {
            if !snapshot.vertices.is_empty() {
                let vert_bytes = std::mem::size_of_val(snapshot.vertices);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        snapshot.vertices.as_ptr() as *const u8,
                        vb.contents() as *mut u8,
                        vert_bytes.min(vb.length() as usize),
                    );
                }
            }
            if !snapshot.indices.is_empty() {
                let idx_bytes = std::mem::size_of_val(snapshot.indices);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        snapshot.indices.as_ptr() as *const u8,
                        ib.contents() as *mut u8,
                        idx_bytes.min(ib.length() as usize),
                    );
                }
            }
            if !snapshot.transforms.is_empty() {
                let trans_bytes = std::mem::size_of_val(snapshot.transforms);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        snapshot.transforms.as_ptr() as *const u8,
                        tb.contents() as *mut u8,
                        trans_bytes.min(tb.length() as usize),
                    );
                }
            }
            {
                let vp_bytes = std::mem::size_of_val(&snapshot.camera_view_proj);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        snapshot.camera_view_proj.as_ptr() as *const u8,
                        vpb.contents() as *mut u8,
                        vp_bytes.min(vpb.length() as usize),
                    );
                }
            }

            if let Some(drawable_ref) = drawable.as_ref() {
                let (p, m, e) = self.encode_frame(
                    &command_buffer,
                    &snapshot,
                    drawable_ref.texture(),
                    &self.depth_texture,
                    vb,
                    ib,
                    tb,
                    vpb,
                );
                prepass_draw_calls = p;
                main_pass_draw_calls = m;
                early_z_reject_estimate = e;
            }
        }

        if let Some(drawable_ref) = drawable {
            command_buffer.present_drawable(drawable_ref);
        }
        command_buffer.commit();

        self.primary_submission_serial = self.primary_submission_serial.saturating_add(1);
        self.current_frame_slot = (self.current_frame_slot + 1) % self.frame_slots.len();

        let submission = MetalFrameSubmissionTelemetry {
            frame_slot,
            frame_id: snapshot_state.frame_id,
            instance_count: snapshot_state.instance_count,
            material_count: snapshot_state.material_count,
            texture_count: snapshot_state.texture_count,
            light_count: snapshot_state.light_count,
        };

        Ok(MetalFrameExecutionTelemetry {
            submission,
            snapshot_state,
            command_buffer_submitted: true,
            presented: drawable.is_some(),
            primary_submission_serial: self.primary_submission_serial,
            prepass_draw_calls,
            main_pass_draw_calls,
            early_z_reject_estimate,
        })
    }

    fn encode_frame(
        &self,
        command_buffer: &metal::CommandBufferRef,
        snapshot: &RenderStateSnapshot<'_>,
        color_texture: &metal::TextureRef,
        depth_texture: &metal::TextureRef,
        vb: &Buffer,
        ib: &Buffer,
        tb: &Buffer,
        vpb: &Buffer,
    ) -> (u32, u32, u32) {
        let index_count = snapshot.indices.len();
        let instance_count = snapshot.transforms.len();
        let mut prepass_draw_calls = 0u32;
        let main_pass_draw_calls;
        let mut early_z_reject_estimate = 0u32;

        if self.config.occlusion_culling_enabled && !snapshot.indices.is_empty() && instance_count > 0 {
            // Pass 1 — Z-Prepass (depth-only)
            let pass_desc = RenderPassDescriptor::new();
            let color_attachment = pass_desc.color_attachments().object_at(0).unwrap();
            color_attachment.set_texture(Some(color_texture));
            color_attachment.set_load_action(MTLLoadAction::Clear);
            color_attachment.set_store_action(MTLStoreAction::DontCare);
            color_attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));

            let depth_attachment = pass_desc.depth_attachment().unwrap();
            depth_attachment.set_texture(Some(depth_texture));
            depth_attachment.set_load_action(MTLLoadAction::Clear);
            depth_attachment.set_store_action(MTLStoreAction::Store);
            depth_attachment.set_clear_depth(1.0);

            let encoder = command_buffer.new_render_command_encoder(&pass_desc);
            encoder.set_render_pipeline_state(&self.z_prepass_pipeline);
            encoder.set_depth_stencil_state(&self.depth_state_write);
            encoder.set_vertex_buffer(0, Some(vb), 0);
            encoder.set_vertex_buffer(1, Some(vpb), 0);
            encoder.set_vertex_buffer(2, Some(tb), 0);
            encoder.draw_indexed_primitives_instanced(
                MTLPrimitiveType::Triangle,
                index_count as u64,
                MTLIndexType::UInt32,
                ib,
                0,
                instance_count as u64,
            );
            prepass_draw_calls = 1;
            encoder.end_encoding();

            // Pass 2 — Forward Shading (equal-depth, no write)
            let pass_desc = RenderPassDescriptor::new();
            let color_attachment = pass_desc.color_attachments().object_at(0).unwrap();
            color_attachment.set_texture(Some(color_texture));
            color_attachment.set_load_action(MTLLoadAction::Load);
            color_attachment.set_store_action(MTLStoreAction::Store);

            let depth_attachment = pass_desc.depth_attachment().unwrap();
            depth_attachment.set_texture(Some(depth_texture));
            depth_attachment.set_load_action(MTLLoadAction::Load);
            depth_attachment.set_store_action(MTLStoreAction::DontCare);

            let encoder = command_buffer.new_render_command_encoder(&pass_desc);
            encoder.set_render_pipeline_state(&self.forward_pipeline);
            encoder.set_depth_stencil_state(&self.depth_state_equal);
            encoder.set_vertex_buffer(0, Some(vb), 0);
            encoder.set_vertex_buffer(1, Some(vpb), 0);
            encoder.set_vertex_buffer(2, Some(tb), 0);
            encoder.draw_indexed_primitives_instanced(
                MTLPrimitiveType::Triangle,
                index_count as u64,
                MTLIndexType::UInt32,
                ib,
                0,
                instance_count as u64,
            );
            main_pass_draw_calls = 1;
            encoder.end_encoding();

            early_z_reject_estimate = prepass_draw_calls.saturating_sub(main_pass_draw_calls);
        } else {
            // Single pass forward when occlusion culling is disabled or no geometry
            let pass_desc = RenderPassDescriptor::new();
            let color_attachment = pass_desc.color_attachments().object_at(0).unwrap();
            color_attachment.set_texture(Some(color_texture));
            color_attachment.set_load_action(MTLLoadAction::Clear);
            color_attachment.set_store_action(MTLStoreAction::Store);
            color_attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));

            let depth_attachment = pass_desc.depth_attachment().unwrap();
            depth_attachment.set_texture(Some(depth_texture));
            depth_attachment.set_load_action(MTLLoadAction::Clear);
            depth_attachment.set_store_action(MTLStoreAction::Store);
            depth_attachment.set_clear_depth(1.0);

            let encoder = command_buffer.new_render_command_encoder(&pass_desc);
            encoder.set_render_pipeline_state(&self.forward_pipeline);
            encoder.set_depth_stencil_state(&self.depth_state_write);
            encoder.set_vertex_buffer(0, Some(vb), 0);
            encoder.set_vertex_buffer(1, Some(vpb), 0);
            encoder.set_vertex_buffer(2, Some(tb), 0);
            encoder.draw_indexed_primitives_instanced(
                MTLPrimitiveType::Triangle,
                index_count as u64,
                MTLIndexType::UInt32,
                ib,
                0,
                instance_count as u64,
            );
            main_pass_draw_calls = 1;
            encoder.end_encoding();
        }

        (prepass_draw_calls, main_pass_draw_calls, early_z_reject_estimate)
    }

    #[cfg(test)]
    pub fn new_for_test(config: MetalBackendConfig) -> Result<Self, MetalBackendError> {
        if config.frames_in_flight == 0 {
            return Err(MetalBackendError::InvalidConfig(
                "frames_in_flight must be greater than zero",
            ));
        }
        if config.max_instances == 0 {
            return Err(MetalBackendError::InvalidConfig(
                "max_instances must be greater than zero",
            ));
        }

        let device = Device::system_default().ok_or(MetalBackendError::NoMetalDevice)?;
        let command_queue = device.new_command_queue();
        let device_name = device.name().to_owned();

        let max_materials = config.max_instances.max(256);
        let max_textures = config.max_instances.max(128);
        let max_lights = 64;
        let frame_slots = (0..config.frames_in_flight)
            .map(|_| SnapshotSlot {
                instance_capacity: config.max_instances,
                material_capacity: max_materials,
                texture_capacity: max_textures,
                light_capacity: max_lights,
                last_state: MetalSnapshotSlotState::default(),
            })
            .collect::<Vec<_>>();

        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .map_err(MetalBackendError::ShaderCompilation)?;

        let z_prepass_vertex = library
            .get_function("z_prepass_vertex", None)
            .map_err(MetalBackendError::ShaderCompilation)?;
        let z_prepass_fragment = library
            .get_function("z_prepass_fragment", None)
            .map_err(MetalBackendError::ShaderCompilation)?;
        let forward_vertex = library
            .get_function("forward_vertex", None)
            .map_err(MetalBackendError::ShaderCompilation)?;
        let forward_fragment = library
            .get_function("forward_fragment", None)
            .map_err(MetalBackendError::ShaderCompilation)?;

        let z_prepass_pipeline = build_pipeline(
            &device,
            &z_prepass_vertex,
            Some(&z_prepass_fragment),
            MTLPixelFormat::BGRA8Unorm,
            MTLPixelFormat::Depth32Float,
        )
        .map_err(MetalBackendError::PipelineCreation)?;

        let forward_pipeline = build_pipeline(
            &device,
            &forward_vertex,
            Some(&forward_fragment),
            MTLPixelFormat::BGRA8Unorm,
            MTLPixelFormat::Depth32Float,
        )
        .map_err(MetalBackendError::PipelineCreation)?;

        let depth_state_write = build_depth_state(&device, MTLCompareFunction::Less, true)
            .map_err(MetalBackendError::PipelineCreation)?;
        let depth_state_equal = build_depth_state(&device, MTLCompareFunction::Equal, false)
            .map_err(MetalBackendError::PipelineCreation)?;

        let surface_size = PhysicalSize::new(64, 64);
        let depth_texture = create_depth_texture(&device, surface_size.width, surface_size.height);

        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_drawable_size(CGSize::new(surface_size.width as f64, surface_size.height as f64));

        let max_verts = config.max_instances.max(1) * 64 * 8;
        let max_indices = config.max_instances.max(1) * 64 * 12;
        let max_transforms = config.max_instances.max(1);

        let vertex_buffer = Some(device.new_buffer(
            (max_verts * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        ));
        let index_buffer = Some(device.new_buffer(
            (max_indices * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        ));
        let transform_buffer = Some(device.new_buffer(
            (max_transforms * std::mem::size_of::<[[f32; 4]; 4]>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        ));
        let view_proj_buffer = Some(device.new_buffer(
            std::mem::size_of::<[[f32; 4]; 4]>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        ));

        Ok(Self {
            config,
            _window: None,
            device,
            command_queue,
            device_name,
            frame_slots,
            current_frame_slot: 0,
            surface_size,
            primary_submission_serial: 0,
            layer,
            depth_texture,
            z_prepass_pipeline,
            forward_pipeline,
            depth_state_write,
            depth_state_equal,
            vertex_buffer,
            index_buffer,
            transform_buffer,
            view_proj_buffer,
        })
    }

    #[cfg(test)]
    pub fn render_n_headless(
        &mut self,
        snapshot: RenderStateSnapshot<'_>,
    ) -> Result<MetalFrameExecutionTelemetry, MetalBackendError> {
        let frame_slot = self.current_frame_slot;
        let snapshot_state = self.upload_state_snapshot(frame_slot, snapshot)?;

        let command_buffer = self.command_queue.new_command_buffer();

        let mut prepass_draw_calls = 0u32;
        let mut main_pass_draw_calls = 0u32;
        let mut early_z_reject_estimate = 0u32;

        if let (Some(vb), Some(ib), Some(tb), Some(vpb)) = (
            self.vertex_buffer.as_ref(),
            self.index_buffer.as_ref(),
            self.transform_buffer.as_ref(),
            self.view_proj_buffer.as_ref(),
        ) {
            if !snapshot.vertices.is_empty() {
                let vert_bytes = std::mem::size_of_val(snapshot.vertices);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        snapshot.vertices.as_ptr() as *const u8,
                        vb.contents() as *mut u8,
                        vert_bytes.min(vb.length() as usize),
                    );
                }
            }
            if !snapshot.indices.is_empty() {
                let idx_bytes = std::mem::size_of_val(snapshot.indices);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        snapshot.indices.as_ptr() as *const u8,
                        ib.contents() as *mut u8,
                        idx_bytes.min(ib.length() as usize),
                    );
                }
            }
            if !snapshot.transforms.is_empty() {
                let trans_bytes = std::mem::size_of_val(snapshot.transforms);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        snapshot.transforms.as_ptr() as *const u8,
                        tb.contents() as *mut u8,
                        trans_bytes.min(tb.length() as usize),
                    );
                }
            }
            {
                let vp_bytes = std::mem::size_of_val(&snapshot.camera_view_proj);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        snapshot.camera_view_proj.as_ptr() as *const u8,
                        vpb.contents() as *mut u8,
                        vp_bytes.min(vpb.length() as usize),
                    );
                }
            }

            let color_texture = create_offscreen_color_texture(
                &self.device,
                self.surface_size.width,
                self.surface_size.height,
            );

            let (p, m, e) = self.encode_frame(
                &command_buffer,
                &snapshot,
                &color_texture,
                &self.depth_texture,
                vb,
                ib,
                tb,
                vpb,
            );
            prepass_draw_calls = p;
            main_pass_draw_calls = m;
            early_z_reject_estimate = e;
        }

        command_buffer.commit();

        self.primary_submission_serial = self.primary_submission_serial.saturating_add(1);
        self.current_frame_slot = (self.current_frame_slot + 1) % self.frame_slots.len();

        let submission = MetalFrameSubmissionTelemetry {
            frame_slot,
            frame_id: snapshot_state.frame_id,
            instance_count: snapshot_state.instance_count,
            material_count: snapshot_state.material_count,
            texture_count: snapshot_state.texture_count,
            light_count: snapshot_state.light_count,
        };

        Ok(MetalFrameExecutionTelemetry {
            submission,
            snapshot_state,
            command_buffer_submitted: true,
            presented: false,
            primary_submission_serial: self.primary_submission_serial,
            prepass_draw_calls,
            main_pass_draw_calls,
            early_z_reject_estimate,
        })
    }
}

fn create_depth_texture(device: &Device, width: u32, height: u32) -> Texture {
    let desc = TextureDescriptor::new();
    desc.set_pixel_format(MTLPixelFormat::Depth32Float);
    desc.set_width(width as u64);
    desc.set_height(height as u64);
    desc.set_usage(MTLTextureUsage::RenderTarget);
    desc.set_storage_mode(metal::MTLStorageMode::Private);
    device.new_texture(&desc)
}

#[cfg(test)]
fn create_offscreen_color_texture(device: &Device, width: u32, height: u32) -> Texture {
    let desc = TextureDescriptor::new();
    desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    desc.set_width(width as u64);
    desc.set_height(height as u64);
    desc.set_usage(MTLTextureUsage::RenderTarget);
    desc.set_storage_mode(metal::MTLStorageMode::Private);
    device.new_texture(&desc)
}

fn build_pipeline(
    device: &Device,
    vertex_function: &Function,
    fragment_function: Option<&Function>,
    color_format: MTLPixelFormat,
    depth_format: MTLPixelFormat,
) -> Result<RenderPipelineState, String> {
    let desc = RenderPipelineDescriptor::new();
    desc.set_vertex_function(Some(vertex_function));
    if let Some(frag) = fragment_function {
        desc.set_fragment_function(Some(frag));
    }

    let color_attachment = desc.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(color_format);

    desc.set_depth_attachment_pixel_format(depth_format);

    // Vertex layout: position (3 floats) + normal (3 floats) + uv (2 floats) = 8 floats
    let vertex_descriptor = metal::VertexDescriptor::new();
    let layout = vertex_descriptor.layouts().object_at(0).unwrap();
    layout.set_stride((8 * std::mem::size_of::<f32>()) as u64);
    layout.set_step_function(MTLVertexStepFunction::PerVertex);

    let attrs = vertex_descriptor.attributes();
    let attr0 = attrs.object_at(0).unwrap();
    attr0.set_format(MTLVertexFormat::Float3);
    attr0.set_offset(0);
    attr0.set_buffer_index(0);

    let attr1 = attrs.object_at(1).unwrap();
    attr1.set_format(MTLVertexFormat::Float3);
    attr1.set_offset((3 * std::mem::size_of::<f32>()) as u64);
    attr1.set_buffer_index(0);

    let attr2 = attrs.object_at(2).unwrap();
    attr2.set_format(MTLVertexFormat::Float2);
    attr2.set_offset((6 * std::mem::size_of::<f32>()) as u64);
    attr2.set_buffer_index(0);

    desc.set_vertex_descriptor(Some(&vertex_descriptor));

    device.new_render_pipeline_state(&desc)
}

fn build_depth_state(
    device: &Device,
    compare: MTLCompareFunction,
    write: bool,
) -> Result<DepthStencilState, String> {
    let desc = DepthStencilDescriptor::new();
    desc.set_depth_compare_function(compare);
    desc.set_depth_write_enabled(write);
    Ok(device.new_depth_stencil_state(&desc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::frame_snapshot::FrameInstanceTransform;

    #[test]
    fn default_config_is_valid() {
        let cfg = MetalBackendConfig::default();
        assert!(cfg.frames_in_flight >= 1);
        assert!(cfg.max_instances >= 1);
        assert!(cfg.occlusion_culling_enabled);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn shader_compilation_succeeds() {
        let device = Device::system_default().expect("no Metal device");
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .expect("shader compilation failed");
        assert!(library.get_function("z_prepass_vertex", None).is_ok());
        assert!(library.get_function("forward_vertex", None).is_ok());
        assert!(library.get_function("forward_fragment", None).is_ok());
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn depth_state_creation() {
        let device = Device::system_default().expect("no Metal device");
        assert!(build_depth_state(&device, MTLCompareFunction::Less, true).is_ok());
        assert!(build_depth_state(&device, MTLCompareFunction::Equal, false).is_ok());
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn pipeline_creation() {
        let device = Device::system_default().expect("no Metal device");
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .unwrap();
        let vert = library.get_function("z_prepass_vertex", None).unwrap();
        let frag = library.get_function("z_prepass_fragment", None).unwrap();
        assert!(build_pipeline(
            &device,
            &vert,
            Some(&frag),
            MTLPixelFormat::BGRA8Unorm,
            MTLPixelFormat::Depth32Float,
        )
        .is_ok());
    }

    fn dummy_snapshot_with_instances(instance_count: usize) -> RenderStateSnapshot<'static> {
        let transforms: Vec<FrameInstanceTransform> = (0..instance_count)
            .map(|_| FrameInstanceTransform {
                model: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                color_rgba: [1.0; 4],
                material_index: 0,
                texture_index: 0,
                flags: 0,
                _padding: 0,
            })
            .collect();
        let vertices: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        ];
        let indices: Vec<u32> = vec![0, 1, 2];
        let transforms = Box::leak(transforms.into_boxed_slice());
        let vertices = Box::leak(vertices.into_boxed_slice());
        let indices = Box::leak(indices.into_boxed_slice());
        RenderStateSnapshot {
            frame_id: 1,
            camera_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            opaque_instance_count: instance_count as u32,
            transparent_instance_count: 0,
            primitive_ranges: &[],
            transforms,
            materials: &[],
            textures: &[],
            lights: &[],
            instance_bounds: &[],
            vertices,
            indices,
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn upload_state_snapshot_capacity_guards() {
        let mut backend = MetalBackend::new_for_test(MetalBackendConfig {
            frames_in_flight: 1,
            max_instances: 4,
            occlusion_culling_enabled: true,
        })
        .unwrap();

        let too_many = dummy_snapshot_with_instances(8);
        match backend.upload_state_snapshot(0, too_many) {
            Err(MetalBackendError::SnapshotCapacityExceeded { requested, capacity }) => {
                assert_eq!(requested, 8);
                assert_eq!(capacity, 4);
            }
            other => panic!("expected SnapshotCapacityExceeded, got {:?}", other),
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn headless_two_pass_telemetry() {
        let mut backend = MetalBackend::new_for_test(MetalBackendConfig {
            frames_in_flight: 1,
            max_instances: 4,
            occlusion_culling_enabled: true,
        })
        .unwrap();

        let snapshot = dummy_snapshot_with_instances(2);
        let telemetry = backend.render_n_headless(snapshot).expect("headless render failed");
        assert!(telemetry.command_buffer_submitted);
        assert_eq!(telemetry.prepass_draw_calls, 1);
        assert_eq!(telemetry.main_pass_draw_calls, 1);
        assert_eq!(telemetry.presented, false);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn headless_single_pass_telemetry() {
        let mut backend = MetalBackend::new_for_test(MetalBackendConfig {
            frames_in_flight: 1,
            max_instances: 4,
            occlusion_culling_enabled: false,
        })
        .unwrap();

        let snapshot = dummy_snapshot_with_instances(2);
        let telemetry = backend.render_n_headless(snapshot).expect("headless render failed");
        assert!(telemetry.command_buffer_submitted);
        assert_eq!(telemetry.prepass_draw_calls, 0);
        assert_eq!(telemetry.main_pass_draw_calls, 1);
        assert_eq!(telemetry.presented, false);
    }
}
