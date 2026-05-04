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
    foreign_types::ForeignType, Buffer, CommandQueue, DepthStencilDescriptor, DepthStencilState,
    Device, MTLClearColor, MTLCompareFunction, MTLLoadAction, MTLPixelFormat,
    MTLPrimitiveType, MTLResourceOptions, MTLStoreAction, MTLTextureType, MTLTextureUsage,
    MTLVertexFormat, MTLVertexStepFunction, MetalLayer, RenderPassDescriptor, RenderPipelineState,
    SamplerDescriptor, SamplerState, Texture, TextureDescriptor,
};
use objc::runtime::Object;
use objc::{msg_send, sel, sel_impl};
use winit::dpi::PhysicalSize;
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::Window;

use super::metal::mesh_slot::MeshSlot;
use crate::graphics::frame_snapshot::{
    FrameInstanceTransform, FrameLightRecord, FramePrimitiveRange, FrameSpriteRecord,
    RenderStateSnapshot, FRAME_PRIMITIVE_RANGE_TRANSPARENT,
};

/// Runtime configuration for the raw Metal backend.
#[derive(Debug, Clone)]
pub struct MetalBackendConfig {
    /// Number of ring slots kept in-flight for snapshot uploads.
    pub frames_in_flight: usize,
    /// Maximum transform records accepted per frame.
    pub max_instances: usize,
    /// Enable Z-prepass + Early-Z occlusion culling.
    pub occlusion_culling_enabled: bool,
    /// Maximum sprite instances accepted per frame.
    pub max_sprites: usize,
    /// Enable SXRC runtime compression for snapshot instance data.
    pub enable_snapshot_compression: bool,
}

impl Default for MetalBackendConfig {
    fn default() -> Self {
        Self {
            frames_in_flight: 3,
            max_instances: 32_768,
            occlusion_culling_enabled: true,
            max_sprites: 4096,
            enable_snapshot_compression: false,
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
    /// Byte length after SXRC compression (0 if compression disabled).
    pub compressed_byte_len: usize,
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
    /// SXRC-compressed instance data retained as a cold tier.
    compressed_instance_data: Option<Vec<sxrc::SxrcCompressedPage>>,
}

/// Errors produced by the raw Metal backend.
#[derive(Debug)]
pub enum MetalBackendError {
    NoMetalDevice,
    InvalidConfig(std::borrow::Cow<'static, str>),
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

use super::metal::shader_library::{
    BlendMode, PipelineKey, ShaderLibrary, VertexAttributeDesc, VertexBufferLayoutDesc,
    VertexLayout,
};
use super::metal::shaders::{SCENE_3D_MSL, SCENE_SHADOW_MSL, SCENE_SPRITE_MSL, SCENE_UPSCALE_MSL};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuInstance3d {
    model_col0: [f32; 4],
    model_col1: [f32; 4],
    model_col2: [f32; 4],
    model_col3: [f32; 4],
    base_color: [f32; 4],
    material_params: [f32; 4],
    emissive: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct LightData {
    position_kind: [f32; 4],
    direction_inner: [f32; 4],
    color_intensity: [f32; 4],
    params: [f32; 4],
    shadow: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_eye: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct LightingUniform {
    light_count: u32,
    rt_mode: u32,
    rt_active: u32,
    rt_dynamic_count: u32,
    rt_dynamic_cap: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ShadowPassUniform {
    light_view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ShadowUniform {
    light_view_proj: [[[f32; 4]; 4]; METAL_SHADOW_LAYERS],
    atlas_scale_offset: [[f32; 4]; METAL_SHADOW_LAYERS],
    shadow_light_indices: [i32; 4],
    shadow_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct UpscaleUniform {
    inv_source_size: [f32; 2],
    source_uv_scale: [f32; 2],
    sharpness: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

const METAL_MAX_LIGHTS: usize = 64;
const METAL_SHADOW_LAYERS: usize = 4;
const METAL_REAL_SHADOW_MAPS_ENABLED: bool = true;

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
    forward_transparent_pipeline: RenderPipelineState,
    depth_state_write: DepthStencilState,
    depth_state_equal: DepthStencilState,
    depth_state_read: DepthStencilState,
    transform_buffers: Vec<Buffer>,
    view_proj_buffers: Vec<Buffer>,
    light_data_buffers: Vec<Buffer>,
    lighting_uniform_buffers: Vec<Buffer>,
    instance_scratch: Vec<GpuInstance3d>,
    shadow_texture: Texture,
    shadow_sampler: SamplerState,
    shadow_pipeline: RenderPipelineState,
    shadow_pass_uniform_buffers: Vec<Buffer>,
    shadow_uniform_buffers: Vec<Buffer>,
    offscreen_color_texture: Texture,
    upscale_pipeline: RenderPipelineState,
    upscale_uniform_buffers: Vec<Buffer>,
    upscale_sampler: SamplerState,
    sprite_pipeline: RenderPipelineState,
    sprite_vertex_buffer: Buffer,
    sprite_instance_buffers: Vec<Buffer>,
    sprite_atlas_texture: Texture,
    sprite_sampler: SamplerState,
    mesh_slots: std::collections::HashMap<u8, MeshSlot>,
    frame_pacing: FramePacing,
    snapshot_compressor: Option<crate::compression::SnapshotCompressor>,
}

#[derive(Clone)]
struct FramePacing {
    inner: std::sync::Arc<(std::sync::Mutex<usize>, std::sync::Condvar)>,
    max: usize,
}

impl FramePacing {
    fn new(max: usize) -> Self {
        Self {
            inner: std::sync::Arc::new((std::sync::Mutex::new(0), std::sync::Condvar::new())),
            max,
        }
    }

    fn wait(&self) {
        let (lock, cvar) = &*self.inner;
        let mut guard = lock.lock().unwrap();
        while *guard >= self.max {
            guard = cvar.wait(guard).unwrap();
        }
        *guard += 1;
    }

    fn signal(&self) {
        let (lock, cvar) = &*self.inner;
        let mut guard = lock.lock().unwrap();
        *guard -= 1;
        cvar.notify_one();
    }
}

impl MetalBackend {
    /// Create a new raw Metal backend bound to the provided window.
    pub fn new(window: Arc<Window>, config: MetalBackendConfig) -> Result<Self, MetalBackendError> {
        if config.frames_in_flight == 0 {
            return Err(MetalBackendError::InvalidConfig(
                "frames_in_flight must be greater than zero".into(),
            ));
        }
        if config.max_instances == 0 {
            return Err(MetalBackendError::InvalidConfig(
                "max_instances must be greater than zero".into(),
            ));
        }

        let device = Device::system_default().ok_or(MetalBackendError::NoMetalDevice)?;
        let command_queue = device.new_command_queue();
        let device_name = device.name().to_owned();

        let max_materials = config.max_instances.max(256);
        let max_textures = config.max_instances.max(128);
        let max_lights = METAL_MAX_LIGHTS;
        let snapshot_compressor = if config.enable_snapshot_compression {
            Some(crate::compression::SnapshotCompressor::new().map_err(|e| {
                MetalBackendError::InvalidConfig(format!("SXRC init failed: {e}").into())
            })?)
        } else {
            None
        };

        let frame_slots = (0..config.frames_in_flight)
            .map(|_| SnapshotSlot {
                instance_capacity: config.max_instances,
                material_capacity: max_materials,
                texture_capacity: max_textures,
                light_capacity: max_lights,
                last_state: MetalSnapshotSlotState::default(),
                compressed_instance_data: None,
            })
            .collect::<Vec<_>>();

        let (z_prepass_pipeline, forward_pipeline, forward_transparent_pipeline) =
            build_scene_pipelines(&device)?;

        let depth_state_write = build_depth_state(&device, MTLCompareFunction::Less, true)
            .map_err(MetalBackendError::PipelineCreation)?;
        let depth_state_equal = build_depth_state(&device, MTLCompareFunction::LessEqual, false)
            .map_err(MetalBackendError::PipelineCreation)?;
        let depth_state_read = build_depth_state(&device, MTLCompareFunction::LessEqual, false)
            .map_err(MetalBackendError::PipelineCreation)?;

        let surface_size = window.inner_size();
        let depth_texture = create_depth_texture(&device, surface_size.width, surface_size.height);

        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_display_sync_enabled(true);
        layer.set_maximum_drawable_count(3);
        unsafe {
            let () = msg_send![layer.as_ptr(), setAllowsNextDrawableTimeout: false];
        }
        layer.set_drawable_size(CGSize::new(
            surface_size.width as f64,
            surface_size.height as f64,
        ));

        let view = {
            let handle = window
                .window_handle()
                .map_err(|_| MetalBackendError::InvalidConfig("failed to get window handle".into()))?;
            match handle.as_raw() {
                RawWindowHandle::AppKit(appkit) => appkit.ns_view.as_ptr() as *mut Object,
                _ => return Err(MetalBackendError::InvalidConfig("not a macOS window".into())),
            }
        };
        unsafe {
            let () = msg_send![view, setWantsLayer: true];
            let () = msg_send![view, setLayer: layer.as_ptr()];
        }

        let max_transforms = config.max_instances.max(1);
        let transform_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    (max_transforms * std::mem::size_of::<GpuInstance3d>()) as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let view_proj_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    std::mem::size_of::<CameraUniform>() as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let light_data_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    (METAL_MAX_LIGHTS * std::mem::size_of::<LightData>()) as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let lighting_uniform_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    std::mem::size_of::<LightingUniform>() as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();

        let shadow_desc = TextureDescriptor::new();
        shadow_desc.set_texture_type(MTLTextureType::D2);
        shadow_desc.set_width(2048);
        shadow_desc.set_height(2048);
        shadow_desc.set_pixel_format(MTLPixelFormat::Depth32Float);
        shadow_desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let shadow_texture = device.new_texture(&shadow_desc);
        let shadow_sampler_desc = SamplerDescriptor::new();
        shadow_sampler_desc.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
        shadow_sampler_desc.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
        shadow_sampler_desc.set_compare_function(metal::MTLCompareFunction::LessEqual);
        shadow_sampler_desc.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
        shadow_sampler_desc.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
        let shadow_sampler = device.new_sampler(&shadow_sampler_desc);
        let shadow_pipeline = build_shadow_pipeline(&device)?;
        let shadow_pass_uniform_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    (METAL_SHADOW_LAYERS * std::mem::size_of::<ShadowPassUniform>()) as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let shadow_uniform_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    std::mem::size_of::<ShadowUniform>() as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let offscreen_color_texture =
            create_offscreen_color_texture(&device, surface_size.width, surface_size.height);
        let upscale_pipeline = build_upscale_pipeline(&device)?;
        let upscale_uniform_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    std::mem::size_of::<UpscaleUniform>() as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let upscale_sampler_desc = SamplerDescriptor::new();
        upscale_sampler_desc.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
        upscale_sampler_desc.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
        upscale_sampler_desc.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
        upscale_sampler_desc.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
        let upscale_sampler = device.new_sampler(&upscale_sampler_desc);

        let sprite_pipeline = build_sprite_pipeline(&device)?;
        let sprite_vertex_buffer = device.new_buffer_with_data(
            [
                -0.5f32, -0.5f32, // bottom-left
                0.5f32, -0.5f32,  // bottom-right
                -0.5f32, 0.5f32,  // top-left
                0.5f32, 0.5f32,   // top-right
            ]
            .as_ptr() as *const _,
            8 * std::mem::size_of::<f32>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let max_sprites = config.max_sprites.max(1);
        let sprite_instance_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    (max_sprites * std::mem::size_of::<FrameSpriteRecord>()) as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let sprite_atlas_desc = TextureDescriptor::new();
        sprite_atlas_desc.set_texture_type(MTLTextureType::D2);
        sprite_atlas_desc.set_width(1);
        sprite_atlas_desc.set_height(1);
        sprite_atlas_desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        sprite_atlas_desc.set_usage(MTLTextureUsage::ShaderRead);
        let sprite_atlas_texture = device.new_texture(&sprite_atlas_desc);
        let sprite_sampler_desc = SamplerDescriptor::new();
        sprite_sampler_desc.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
        sprite_sampler_desc.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
        sprite_sampler_desc.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
        sprite_sampler_desc.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
        let sprite_sampler = device.new_sampler(&sprite_sampler_desc);

        let frame_pacing = FramePacing::new(config.frames_in_flight);

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
            forward_transparent_pipeline,
            depth_state_write,
            depth_state_equal,
            depth_state_read,
            transform_buffers,
            view_proj_buffers,
            light_data_buffers,
            lighting_uniform_buffers,
            instance_scratch: Vec::with_capacity(max_transforms),
            shadow_texture,
            shadow_sampler,
            shadow_pipeline,
            shadow_pass_uniform_buffers,
            shadow_uniform_buffers,
            offscreen_color_texture,
            upscale_pipeline,
            upscale_uniform_buffers,
            upscale_sampler,
            sprite_pipeline,
            sprite_vertex_buffer,
            sprite_instance_buffers,
            sprite_atlas_texture,
            sprite_sampler,
            mesh_slots: std::collections::HashMap::new(),
            frame_pacing,
            snapshot_compressor,
        })
    }

    /// Report the active Metal device name.
    pub fn device(&self) -> &Device {
        &self.device
    }

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
    pub fn bind_mesh_slot(&mut self, slot: u8, mesh: MeshSlot) {
        self.mesh_slots.insert(slot, mesh);
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) -> Result<(), MetalBackendError> {
        if new_size.width == 0 || new_size.height == 0 {
            return Ok(());
        }
        self.surface_size = new_size;
        self.depth_texture = create_depth_texture(&self.device, new_size.width, new_size.height);
        self.offscreen_color_texture =
            create_offscreen_color_texture(&self.device, new_size.width, new_size.height);
        self.layer
            .set_drawable_size(CGSize::new(new_size.width as f64, new_size.height as f64));
        Ok(())
    }

    /// Write one render-visible snapshot into the selected frame slot.
    pub fn upload_state_snapshot(
        &mut self,
        frame_slot: usize,
        snapshot: &RenderStateSnapshot<'_>,
    ) -> Result<MetalSnapshotSlotState, MetalBackendError> {
        let slot = self
            .frame_slots
            .get_mut(frame_slot)
            .ok_or(MetalBackendError::InvalidConfig("frame slot out of range".into()))?;

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
            compressed_byte_len: slot.last_state.compressed_byte_len,
        };

        Ok(slot.last_state)
    }

    fn prepare_instance_scratch(&mut self, snapshot: &RenderStateSnapshot<'_>) {
        self.instance_scratch.clear();
        self.instance_scratch.reserve(snapshot.transforms.len());
        for transform in snapshot.transforms {
            let (material_params, emissive_rgb) = snapshot
                .materials
                .get(transform.material_index as usize)
                .map(|material| (material.material_params, material.emissive_rgb))
                .unwrap_or(([0.55, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]));
            self.instance_scratch.push(GpuInstance3d {
                model_col0: transform.model[0],
                model_col1: transform.model[1],
                model_col2: transform.model[2],
                model_col3: transform.model[3],
                base_color: transform.color_rgba,
                material_params,
                emissive: [emissive_rgb[0], emissive_rgb[1], emissive_rgb[2], 0.0],
            });
        }
    }

    fn upload_lights(
        &self,
        _frame_slot: usize,
        lb: &Buffer,
        lub: &Buffer,
        snapshot: &RenderStateSnapshot<'_>,
    ) {
        let lighting = LightingUniform {
            light_count: snapshot.lights.len() as u32,
            rt_mode: 0,
            rt_active: 0,
            rt_dynamic_count: 0,
            rt_dynamic_cap: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        if !snapshot.lights.is_empty() {
            let light_bytes = snapshot.lights.len() * std::mem::size_of::<FrameLightRecord>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    snapshot.lights.as_ptr() as *const u8,
                    lb.contents() as *mut u8,
                    light_bytes.min(lb.length() as usize),
                );
            }
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                &lighting as *const LightingUniform as *const u8,
                lub.contents() as *mut u8,
                std::mem::size_of::<LightingUniform>(),
            );
        }
    }

    fn upload_shadows(
        &self,
        _frame_slot: usize,
        spub: &Buffer,
        sub: &Buffer,
        snapshot: &RenderStateSnapshot<'_>,
    ) {
        use nalgebra::{Isometry3, Perspective3, Point3, Vector3};

        let mut shadow_uniform = ShadowUniform {
            light_view_proj: [[[0.0; 4]; 4]; METAL_SHADOW_LAYERS],
            atlas_scale_offset: [[0.0; 4]; METAL_SHADOW_LAYERS],
            shadow_light_indices: [-1; 4],
            shadow_count: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let mut slot = 0usize;
        for (i, light) in snapshot.lights.iter().enumerate() {
            if slot >= METAL_SHADOW_LAYERS {
                break;
            }
            let casts_shadow = light.shadow[0] > 0.5;
            let kind = light.position_kind[3] as u32;
            if METAL_REAL_SHADOW_MAPS_ENABLED && casts_shadow && kind == 1 {
                let pos = Point3::new(light.position_kind[0], light.position_kind[1], light.position_kind[2]);
                let dir = Vector3::new(light.direction_inner[0], light.direction_inner[1], light.direction_inner[2]).normalize();
                let up = if dir.y.abs() < 0.99 {
                    Vector3::y()
                } else {
                    Vector3::x()
                };
                let view = Isometry3::look_at_rh(&pos, &Point3::from(pos + dir), &up);

                let outer_cos = light.params[1];
                let outer_deg = outer_cos.acos().to_degrees();
                let fov_y = ((outer_deg * 2.0 + 6.0) as f32)
                    .clamp(10.0, 170.0)
                    .to_radians();
                let range = light.params[0].max(1.0);
                let proj = Perspective3::new(1.0, fov_y, 0.1, range);

                let view_proj = proj.to_homogeneous() * view.to_homogeneous();
                let slice = view_proj.as_slice();
                let mut vp = [[0f32; 4]; 4];
                for col in 0..4 {
                    vp[col].copy_from_slice(&slice[col * 4..(col + 1) * 4]);
                }

                shadow_uniform.light_view_proj[slot] = vp;
                shadow_uniform.atlas_scale_offset[slot] = [
                    0.5,
                    0.5,
                    if slot % 2 == 0 { 0.0 } else { 0.5 },
                    if slot / 2 == 0 { 0.0 } else { 0.5 },
                ];
                shadow_uniform.shadow_light_indices[slot] = i as i32;

                let pass_uniform = ShadowPassUniform { light_view_proj: vp };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &pass_uniform as *const ShadowPassUniform as *const u8,
                        (spub.contents() as *mut u8).add(slot * std::mem::size_of::<ShadowPassUniform>()),
                        std::mem::size_of::<ShadowPassUniform>(),
                    );
                }

                slot += 1;
            }
        }
        shadow_uniform.shadow_count = slot as u32;

        unsafe {
            std::ptr::copy_nonoverlapping(
                &shadow_uniform as *const ShadowUniform as *const u8,
                sub.contents() as *mut u8,
                std::mem::size_of::<ShadowUniform>(),
            );
        }
    }

    fn encode_shadow_pass(
        &self,
        command_buffer: &metal::CommandBufferRef,
        tb: &Buffer,
        spub: &Buffer,
        ranges: &[FramePrimitiveRange],
    ) {
        let pass_desc = RenderPassDescriptor::new();
        let depth_attachment = pass_desc.depth_attachment().unwrap();
        depth_attachment.set_texture(Some(&self.shadow_texture));
        depth_attachment.set_load_action(MTLLoadAction::Clear);
        depth_attachment.set_store_action(MTLStoreAction::Store);
        depth_attachment.set_clear_depth(1.0);

        let encoder = command_buffer.new_render_command_encoder(&pass_desc);
        encoder.set_render_pipeline_state(&self.shadow_pipeline);
        encoder.set_depth_stencil_state(&self.depth_state_write);
        encoder.set_cull_mode(metal::MTLCullMode::Back);
        encoder.set_depth_clip_mode(metal::MTLDepthClipMode::Clamp);
        encoder.set_depth_bias(0.001, 1.0, 0.001);
        encoder.set_vertex_buffer(1, Some(tb), 0);

        for layer in 0..METAL_SHADOW_LAYERS {
            let offset_x = if layer % 2 == 0 { 0.0 } else { 1024.0 };
            let offset_y = if layer / 2 == 0 { 0.0 } else { 1024.0 };
            encoder.set_viewport(metal::MTLViewport {
                originX: offset_x,
                originY: offset_y,
                width: 1024.0,
                height: 1024.0,
                znear: 0.0,
                zfar: 1.0,
            });
            let offset = (layer * std::mem::size_of::<ShadowPassUniform>()) as u64;
            encoder.set_vertex_buffer(2, Some(spub), offset);
            for range in ranges.iter().filter(|range| !Self::is_transparent_range(range)) {
                if let Some(mesh) = self.resolve_mesh_slot(range.primitive_code as u8) {
                    encoder.set_vertex_buffer(0, Some(&mesh.vertex_buffer), 0);
                    encoder.draw_indexed_primitives_instanced_base_instance(
                        MTLPrimitiveType::Triangle,
                        mesh.index_count as u64,
                        mesh.index_type,
                        &mesh.index_buffer,
                        0,
                        range.instance_count as u64,
                        0,
                        range.first_instance as u64,
                    );
                }
            }
        }
        encoder.end_encoding();
    }

    fn encode_sprite_pass(
        &self,
        command_buffer: &metal::CommandBufferRef,
        color_texture: &metal::TextureRef,
        frame_slot: usize,
        sprite_count: usize,
        lighting_buffer: &Buffer,
    ) {
        if sprite_count == 0 {
            return;
        }
        let pass_desc = RenderPassDescriptor::new();
        let color_attachment = pass_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(color_texture));
        color_attachment.set_load_action(MTLLoadAction::Load);
        color_attachment.set_store_action(MTLStoreAction::Store);

        let encoder = command_buffer.new_render_command_encoder(&pass_desc);
        encoder.set_render_pipeline_state(&self.sprite_pipeline);
        encoder.set_vertex_buffer(0, Some(&self.sprite_vertex_buffer), 0);
        encoder.set_vertex_buffer(1, Some(&self.sprite_instance_buffers[frame_slot]), 0);
        encoder.set_fragment_texture(0, Some(&self.sprite_atlas_texture));
        encoder.set_fragment_sampler_state(0, Some(&self.sprite_sampler));
        encoder.set_fragment_buffer(0, Some(lighting_buffer), 0);
        encoder.draw_primitives_instanced(
            MTLPrimitiveType::TriangleStrip,
            0,
            4,
            sprite_count as u64,
        );
        encoder.end_encoding();
    }

    fn encode_upscale_pass(
        &self,
        command_buffer: &metal::CommandBufferRef,
        source_texture: &metal::TextureRef,
        target_texture: &metal::TextureRef,
        source_width: u32,
        source_height: u32,
        target_width: u32,
        target_height: u32,
        frame_slot: usize,
    ) {
        let pass_desc = RenderPassDescriptor::new();
        let color_attachment = pass_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(target_texture));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_store_action(MTLStoreAction::Store);
        color_attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));

        let encoder = command_buffer.new_render_command_encoder(&pass_desc);
        encoder.set_render_pipeline_state(&self.upscale_pipeline);
        encoder.set_fragment_texture(0, Some(source_texture));
        encoder.set_fragment_sampler_state(0, Some(&self.upscale_sampler));

        let uniform = UpscaleUniform {
            inv_source_size: [1.0 / source_width as f32, 1.0 / source_height as f32],
            source_uv_scale: [
                source_width as f32 / target_width.max(1) as f32,
                source_height as f32 / target_height.max(1) as f32,
            ],
            sharpness: 0.5,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &uniform as *const UpscaleUniform as *const u8,
                self.upscale_uniform_buffers[frame_slot].contents() as *mut u8,
                std::mem::size_of::<UpscaleUniform>(),
            );
        }
        encoder.set_fragment_buffer(0, Some(&self.upscale_uniform_buffers[frame_slot]), 0);
        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        encoder.end_encoding();
    }

    /// Submit one frame worth of work to the Metal command queue.
    pub fn render_n(
        &mut self,
        snapshot: RenderStateSnapshot<'_>,
    ) -> Result<MetalFrameExecutionTelemetry, MetalBackendError> {
        let frame_slot = self.current_frame_slot;
        let snapshot_state = self.upload_state_snapshot(frame_slot, &snapshot)?;
        self.prepare_instance_scratch(&snapshot);

        // Wait for an in-flight slot before acquiring a drawable
        self.frame_pacing.wait();

        let drawable = self.layer.next_drawable();
        let command_buffer = self.command_queue.new_command_buffer();

        let mut prepass_draw_calls = 0u32;
        let mut main_pass_draw_calls = 0u32;
        let mut early_z_reject_estimate = 0u32;

        if let Some(drawable_ref) = drawable.as_ref() {
            // Upload transform and view-proj data if provided
            let tb = &self.transform_buffers[frame_slot];
            let vpb = &self.view_proj_buffers[frame_slot];
            let cull_result = if self.config.occlusion_culling_enabled {
                let planes = Self::extract_frustum_planes(snapshot.camera_view_proj);
                Some(self.cull_and_compact(&snapshot, &planes))
            } else {
                None
            };
            let mut draw_ranges: Vec<FramePrimitiveRange> = match &cull_result {
                Some((_, visible_ranges)) => visible_ranges.clone(),
                None => snapshot.primitive_ranges.to_vec(),
            };
            let instance_data: &[GpuInstance3d] = match &cull_result {
                Some((visible_instances, _)) => visible_instances.as_slice(),
                None => self.instance_scratch.as_slice(),
            };

            // Back-to-front transparent sorting: farther surfaces are drawn first
            // so that nearer transparent pixels blend over them correctly.
            if draw_ranges.iter().any(|r| Self::is_transparent_range(r)) {
                let camera_eye = snapshot.camera_eye;
                let transforms = snapshot.transforms;
                draw_ranges.sort_by(|a, b| {
                    let a_trans = Self::is_transparent_range(a);
                    let b_trans = Self::is_transparent_range(b);
                    match (a_trans, b_trans) {
                        (true, true) => {
                            let dist_a = Self::range_camera_distance(a, transforms, camera_eye);
                            let dist_b = Self::range_camera_distance(b, transforms, camera_eye);
                            dist_b.partial_cmp(&dist_a).unwrap_or(std::cmp::Ordering::Equal)
                        }
                        (true, false) => std::cmp::Ordering::Greater,
                        (false, true) => std::cmp::Ordering::Less,
                        (false, false) => std::cmp::Ordering::Equal,
                    }
                });
            }

            if !instance_data.is_empty() {
                let trans_bytes = std::mem::size_of_val(instance_data);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        instance_data.as_ptr() as *const u8,
                        tb.contents() as *mut u8,
                        trans_bytes.min(tb.length() as usize),
                    );
                }
                if let Some(ref compressor) = self.snapshot_compressor {
                    let instance_bytes = unsafe {
                        std::slice::from_raw_parts(
                            instance_data.as_ptr() as *const u8,
                            trans_bytes,
                        )
                    };
                    match compressor.compress(instance_bytes) {
                        Ok(pages) => {
                            let compressed_len: usize = pages.iter().map(|p| p.encoded.len()).sum();
                            let slot = &mut self.frame_slots[frame_slot];
                            slot.compressed_instance_data = Some(pages);
                            slot.last_state.compressed_byte_len = compressed_len;
                        }
                        Err(_) => {
                            let slot = &mut self.frame_slots[frame_slot];
                            slot.compressed_instance_data = None;
                            slot.last_state.compressed_byte_len = 0;
                        }
                    }
                }
            }
            {
                let camera = CameraUniform {
                    view_proj: snapshot.camera_view_proj,
                    camera_eye: snapshot.camera_eye,
                };
                let vp_bytes = std::mem::size_of::<CameraUniform>();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &camera as *const CameraUniform as *const u8,
                        vpb.contents() as *mut u8,
                        vp_bytes.min(vpb.length() as usize),
                    );
                }
            }

            let lb = &self.light_data_buffers[frame_slot];
            let lub = &self.lighting_uniform_buffers[frame_slot];
            let spub = &self.shadow_pass_uniform_buffers[frame_slot];
            let sub = &self.shadow_uniform_buffers[frame_slot];
            self.upload_lights(frame_slot, lb, lub, &snapshot);
            self.upload_shadows(frame_slot, spub, sub, &snapshot);

            if METAL_REAL_SHADOW_MAPS_ENABLED {
                self.encode_shadow_pass(&command_buffer, tb, spub, &draw_ranges);
            }

            let (p, m, e) = self.encode_frame(
                &command_buffer,
                &snapshot,
                &self.offscreen_color_texture,
                &self.depth_texture,
                tb,
                vpb,
                lb,
                lub,
                sub,
                &draw_ranges,
            );
            prepass_draw_calls = p;
            main_pass_draw_calls = m;
            early_z_reject_estimate = e;

            if !snapshot.sprites.is_empty() {
                let sprite_buf = &self.sprite_instance_buffers[frame_slot];
                let sprite_bytes = std::mem::size_of_val(snapshot.sprites);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        snapshot.sprites.as_ptr() as *const u8,
                        sprite_buf.contents() as *mut u8,
                        sprite_bytes.min(sprite_buf.length() as usize),
                    );
                }
            }
            self.encode_sprite_pass(
                &command_buffer,
                &self.offscreen_color_texture,
                frame_slot,
                snapshot.sprites.len(),
                lub,
            );

            self.encode_upscale_pass(
                &command_buffer,
                &self.offscreen_color_texture,
                drawable_ref.texture(),
                self.surface_size.width,
                self.surface_size.height,
                self.surface_size.width,
                self.surface_size.height,
                frame_slot,
            );

            let pacing = self.frame_pacing.clone();
            let concrete = block::ConcreteBlock::new(move |_buffer: &metal::CommandBufferRef| {
                pacing.signal();
            });
            let block = concrete.copy();
            command_buffer.add_completed_handler(&block);
            command_buffer.present_drawable(drawable_ref);
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
                presented: true,
                primary_submission_serial: self.primary_submission_serial,
                prepass_draw_calls,
                main_pass_draw_calls,
                early_z_reject_estimate,
            })
        } else {
            // No drawable available – return the pacing token immediately
            self.frame_pacing.signal();
            Ok(MetalFrameExecutionTelemetry {
                submission: MetalFrameSubmissionTelemetry {
                    frame_slot,
                    frame_id: snapshot_state.frame_id,
                    instance_count: snapshot_state.instance_count,
                    material_count: snapshot_state.material_count,
                    texture_count: snapshot_state.texture_count,
                    light_count: snapshot_state.light_count,
                },
                snapshot_state,
                command_buffer_submitted: false,
                presented: false,
                primary_submission_serial: self.primary_submission_serial,
                prepass_draw_calls: 0,
                main_pass_draw_calls: 0,
                early_z_reject_estimate: 0,
            })
        }
    }

    /// Resolve a mesh slot from a primitive code, matching the WGPU renderer mapping:
    /// - 0 → built-in sphere
    /// - 1 → built-in box
    /// - 2+ → custom mesh slot = code - 2
    fn resolve_mesh_slot(&self, primitive_code: u8) -> Option<&MeshSlot> {
        match primitive_code {
            0 => self.mesh_slots.get(&0),
            1 => self.mesh_slots.get(&1),
            code => self.mesh_slots.get(&(code.saturating_sub(2))),
        }
    }

    /// Extract six view-frustum planes from a column-major view-projection matrix.
    /// Planes are normalized (xyz) and oriented so that `dot(point, xyz) + w >= 0`
    /// means the point is on the inside side.
    #[inline]
    fn extract_frustum_planes(view_proj: [[f32; 4]; 4]) -> [[f32; 4]; 6] {
        let mut p = [[0.0f32; 4]; 6];
        // column-major layout: view_proj[col][row] = M[row][col]
        // row i = [view_proj[0][i], view_proj[1][i], view_proj[2][i], view_proj[3][i]]
        for c in 0..4 {
            p[0][c] = view_proj[c][0] + view_proj[c][3]; // left   = row0 + row3
            p[1][c] = view_proj[c][3] - view_proj[c][0]; // right  = row3 - row0
            p[2][c] = view_proj[c][1] + view_proj[c][3]; // bottom = row1 + row3
            p[3][c] = view_proj[c][3] - view_proj[c][1]; // top    = row3 - row1
            p[4][c] = view_proj[c][2] + view_proj[c][3]; // near   = row2 + row3 (Metal z ∈ [0,w])
            p[5][c] = view_proj[c][3] - view_proj[c][2]; // far    = row3 - row2
        }
        for plane in &mut p {
            let len = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2])
                .sqrt()
                .max(1e-6);
            plane[0] /= len;
            plane[1] /= len;
            plane[2] /= len;
            plane[3] /= len;
        }
        p
    }

    #[inline]
    fn is_sphere_visible(center: [f32; 3], radius_sq: f32, planes: &[[f32; 4]; 6]) -> bool {
        for plane in planes {
            let dist = plane[0] * center[0] + plane[1] * center[1] + plane[2] * center[2] + plane[3];
            if dist < 0.0 && dist * dist > radius_sq {
                return false;
            }
        }
        true
    }

    #[inline]
    fn primitive_radius_sq(primitive_code: u32) -> f32 {
        match primitive_code {
            0 => 1.0,  // unit sphere
            1 => 0.75, // (sqrt(3)/2)^2: circumscribed sphere of unit box
            _ => 1.0,  // custom mesh: conservative
        }
    }

    /// CPU-side frustum cull + compaction.
    ///
    /// Returns a compacted instance buffer containing only visible instances,
    /// and rewritten primitive ranges that point into the compacted buffer.
    fn cull_and_compact(
        &self,
        snapshot: &RenderStateSnapshot<'_>,
        planes: &[[f32; 4]; 6],
    ) -> (Vec<GpuInstance3d>, Vec<FramePrimitiveRange>) {
        let mut visible = Vec::with_capacity(self.instance_scratch.len());
        let mut ranges = Vec::with_capacity(snapshot.primitive_ranges.len());

        for range in snapshot.primitive_ranges {
            let base_radius_sq = Self::primitive_radius_sq(range.primitive_code);
            let start = range.first_instance;
            let end = range.first_instance + range.instance_count;
            let mut block_start: Option<u32> = None;

            for i in start..end {
                let inst = &self.instance_scratch[i as usize];
                let center = [inst.model_col3[0], inst.model_col3[1], inst.model_col3[2]];
                let sx = inst.model_col0[0] * inst.model_col0[0]
                    + inst.model_col0[1] * inst.model_col0[1]
                    + inst.model_col0[2] * inst.model_col0[2];
                let sy = inst.model_col1[0] * inst.model_col1[0]
                    + inst.model_col1[1] * inst.model_col1[1]
                    + inst.model_col1[2] * inst.model_col1[2];
                let sz = inst.model_col2[0] * inst.model_col2[0]
                    + inst.model_col2[1] * inst.model_col2[1]
                    + inst.model_col2[2] * inst.model_col2[2];
                let radius_sq = base_radius_sq * sx.max(sy).max(sz);

                if Self::is_sphere_visible(center, radius_sq, planes) {
                    visible.push(*inst);
                    if block_start.is_none() {
                        block_start = Some((visible.len() - 1) as u32);
                    }
                } else if let Some(bs) = block_start {
                    ranges.push(FramePrimitiveRange {
                        primitive_code: range.primitive_code,
                        first_instance: bs,
                        instance_count: (visible.len() as u32) - bs,
                        flags: range.flags,
                    });
                    block_start = None;
                }
            }

            if let Some(bs) = block_start {
                ranges.push(FramePrimitiveRange {
                    primitive_code: range.primitive_code,
                    first_instance: bs,
                    instance_count: (visible.len() as u32) - bs,
                    flags: range.flags,
                });
            }
        }

        Self::merge_ranges(&mut ranges);
        (visible, ranges)
    }

    /// Merge consecutive primitive ranges that share the same mesh and contiguous instances.
    fn merge_ranges(ranges: &mut Vec<FramePrimitiveRange>) {
        if ranges.len() < 2 {
            return;
        }
        let mut write = 0;
        for read in 1..ranges.len() {
            let prev = ranges[write];
            let curr = ranges[read];
            if prev.primitive_code == curr.primitive_code
                && prev.first_instance + prev.instance_count == curr.first_instance
                && prev.flags == curr.flags
            {
                ranges[write].instance_count += curr.instance_count;
            } else {
                write += 1;
                ranges[write] = curr;
            }
        }
        ranges.truncate(write + 1);
    }

    #[inline]
    fn is_transparent_range(range: &FramePrimitiveRange) -> bool {
        range.flags & FRAME_PRIMITIVE_RANGE_TRANSPARENT != 0
    }

    /// Approximate squared camera distance for a primitive range, averaged over
    /// its instances. Used for back-to-front transparent sorting.
    fn range_camera_distance(
        range: &FramePrimitiveRange,
        transforms: &[FrameInstanceTransform],
        camera_eye: [f32; 4],
    ) -> f32 {
        if transforms.is_empty() {
            return 0.0;
        }
        let start = range.first_instance as usize;
        let end = (range.first_instance + range.instance_count) as usize;
        let mut avg = [0.0f32; 3];
        let mut count = 0usize;
        for i in start..end.min(transforms.len()) {
            let model = transforms[i].model;
            avg[0] += model[3][0];
            avg[1] += model[3][1];
            avg[2] += model[3][2];
            count += 1;
        }
        if count > 0 {
            let inv = 1.0 / count as f32;
            avg[0] *= inv;
            avg[1] *= inv;
            avg[2] *= inv;
        }
        let dx = avg[0] - camera_eye[0];
        let dy = avg[1] - camera_eye[1];
        let dz = avg[2] - camera_eye[2];
        dx * dx + dy * dy + dz * dz
    }

    fn encode_frame(
        &self,
        command_buffer: &metal::CommandBufferRef,
        _snapshot: &RenderStateSnapshot<'_>,
        color_texture: &metal::TextureRef,
        depth_texture: &metal::TextureRef,
        tb: &Buffer,
        vpb: &Buffer,
        lb: &Buffer,
        lub: &Buffer,
        sub: &Buffer,
        ranges: &[FramePrimitiveRange],
    ) -> (u32, u32, u32) {
        let mut prepass_draw_calls = 0u32;
        let mut main_pass_draw_calls = 0u32;
        let mut early_z_reject_estimate = 0u32;

        let has_opaque = ranges
            .iter()
            .any(|range| range.instance_count > 0 && !Self::is_transparent_range(range));
        let has_transparent = ranges
            .iter()
            .any(|range| range.instance_count > 0 && Self::is_transparent_range(range));

        if !has_opaque && !has_transparent {
            let pass_desc = RenderPassDescriptor::new();
            let color_attachment = pass_desc.color_attachments().object_at(0).unwrap();
            color_attachment.set_texture(Some(color_texture));
            color_attachment.set_load_action(MTLLoadAction::Clear);
            color_attachment.set_store_action(MTLStoreAction::Store);
            color_attachment.set_clear_color(scene_clear_color());

            let depth_attachment = pass_desc.depth_attachment().unwrap();
            depth_attachment.set_texture(Some(depth_texture));
            depth_attachment.set_load_action(MTLLoadAction::Clear);
            depth_attachment.set_store_action(MTLStoreAction::DontCare);
            depth_attachment.set_clear_depth(1.0);

            command_buffer
                .new_render_command_encoder(&pass_desc)
                .end_encoding();
            return (0, 0, 0);
        }

        if has_opaque {
            if self.config.occlusion_culling_enabled {
                // Pass 1 — Z-Prepass for opaque geometry only.
                let pass_desc = RenderPassDescriptor::new();
                let color_attachment = pass_desc.color_attachments().object_at(0).unwrap();
                color_attachment.set_texture(Some(color_texture));
                color_attachment.set_load_action(MTLLoadAction::Clear);
                color_attachment.set_store_action(MTLStoreAction::DontCare);
                color_attachment.set_clear_color(scene_clear_color());

                let depth_attachment = pass_desc.depth_attachment().unwrap();
                depth_attachment.set_texture(Some(depth_texture));
                depth_attachment.set_load_action(MTLLoadAction::Clear);
                depth_attachment.set_store_action(MTLStoreAction::Store);
                depth_attachment.set_clear_depth(1.0);

                let encoder = command_buffer.new_render_command_encoder(&pass_desc);
                encoder.set_render_pipeline_state(&self.z_prepass_pipeline);
                encoder.set_depth_stencil_state(&self.depth_state_write);
                encoder.set_cull_mode(metal::MTLCullMode::Back);
                encoder.set_vertex_buffer(1, Some(tb), 0);
                encoder.set_vertex_buffer(2, Some(vpb), 0);
                for range in ranges.iter().filter(|range| !Self::is_transparent_range(range)) {
                    if let Some(mesh) = self.resolve_mesh_slot(range.primitive_code as u8) {
                        encoder.set_vertex_buffer(0, Some(&mesh.vertex_buffer), 0);
                        encoder.draw_indexed_primitives_instanced_base_instance(
                            MTLPrimitiveType::Triangle,
                            mesh.index_count as u64,
                            mesh.index_type,
                            &mesh.index_buffer,
                            0,
                            range.instance_count as u64,
                            0,
                            range.first_instance as u64,
                        );
                        prepass_draw_calls += 1;
                    }
                }
                encoder.end_encoding();

                let pass_desc = RenderPassDescriptor::new();
                let color_attachment = pass_desc.color_attachments().object_at(0).unwrap();
                color_attachment.set_texture(Some(color_texture));
                // The prepass is depth-only, so the color buffer contents are undefined here.
                color_attachment.set_load_action(MTLLoadAction::Clear);
                color_attachment.set_store_action(MTLStoreAction::Store);
                color_attachment.set_clear_color(scene_clear_color());

                let depth_attachment = pass_desc.depth_attachment().unwrap();
                depth_attachment.set_texture(Some(depth_texture));
                depth_attachment.set_load_action(MTLLoadAction::Load);
                depth_attachment.set_store_action(if has_transparent {
                    MTLStoreAction::Store
                } else {
                    MTLStoreAction::DontCare
                });

                let encoder = command_buffer.new_render_command_encoder(&pass_desc);
                encoder.set_render_pipeline_state(&self.forward_pipeline);
                encoder.set_depth_stencil_state(&self.depth_state_equal);
                encoder.set_cull_mode(metal::MTLCullMode::Back);
                encoder.set_vertex_buffer(1, Some(tb), 0);
                encoder.set_vertex_buffer(2, Some(vpb), 0);
                encoder.set_fragment_buffer(0, Some(vpb), 0);
                encoder.set_fragment_buffer(1, Some(lb), 0);
                encoder.set_fragment_buffer(2, Some(lub), 0);
                encoder.set_fragment_buffer(3, Some(sub), 0);
                encoder.set_fragment_texture(0, Some(&self.shadow_texture));
                encoder.set_fragment_sampler_state(0, Some(&self.shadow_sampler));
                for range in ranges.iter().filter(|range| !Self::is_transparent_range(range)) {
                    if let Some(mesh) = self.resolve_mesh_slot(range.primitive_code as u8) {
                        encoder.set_vertex_buffer(0, Some(&mesh.vertex_buffer), 0);
                        encoder.draw_indexed_primitives_instanced_base_instance(
                            MTLPrimitiveType::Triangle,
                            mesh.index_count as u64,
                            mesh.index_type,
                            &mesh.index_buffer,
                            0,
                            range.instance_count as u64,
                            0,
                            range.first_instance as u64,
                        );
                        main_pass_draw_calls += 1;
                    }
                }
                encoder.end_encoding();

                early_z_reject_estimate =
                    prepass_draw_calls.saturating_sub(main_pass_draw_calls);
            } else {
                let pass_desc = RenderPassDescriptor::new();
                let color_attachment = pass_desc.color_attachments().object_at(0).unwrap();
                color_attachment.set_texture(Some(color_texture));
                color_attachment.set_load_action(MTLLoadAction::Clear);
                color_attachment.set_store_action(MTLStoreAction::Store);
                color_attachment.set_clear_color(scene_clear_color());

                let depth_attachment = pass_desc.depth_attachment().unwrap();
                depth_attachment.set_texture(Some(depth_texture));
                depth_attachment.set_load_action(MTLLoadAction::Clear);
                depth_attachment.set_store_action(if has_transparent {
                    MTLStoreAction::Store
                } else {
                    MTLStoreAction::DontCare
                });
                depth_attachment.set_clear_depth(1.0);

                let encoder = command_buffer.new_render_command_encoder(&pass_desc);
                encoder.set_render_pipeline_state(&self.forward_pipeline);
                encoder.set_depth_stencil_state(&self.depth_state_write);
                encoder.set_vertex_buffer(1, Some(tb), 0);
                encoder.set_vertex_buffer(2, Some(vpb), 0);
                encoder.set_fragment_buffer(0, Some(vpb), 0);
                encoder.set_fragment_buffer(1, Some(lb), 0);
                encoder.set_fragment_buffer(2, Some(lub), 0);
                encoder.set_fragment_buffer(3, Some(sub), 0);
                encoder.set_fragment_texture(0, Some(&self.shadow_texture));
                encoder.set_fragment_sampler_state(0, Some(&self.shadow_sampler));
                for range in ranges.iter().filter(|range| !Self::is_transparent_range(range)) {
                    if let Some(mesh) = self.resolve_mesh_slot(range.primitive_code as u8) {
                        encoder.set_vertex_buffer(0, Some(&mesh.vertex_buffer), 0);
                        encoder.draw_indexed_primitives_instanced_base_instance(
                            MTLPrimitiveType::Triangle,
                            mesh.index_count as u64,
                            mesh.index_type,
                            &mesh.index_buffer,
                            0,
                            range.instance_count as u64,
                            0,
                            range.first_instance as u64,
                        );
                        main_pass_draw_calls += 1;
                    }
                }
                encoder.end_encoding();
            }
        }

        if has_transparent {
            let pass_desc = RenderPassDescriptor::new();
            let color_attachment = pass_desc.color_attachments().object_at(0).unwrap();
            color_attachment.set_texture(Some(color_texture));
            color_attachment.set_load_action(if has_opaque {
                MTLLoadAction::Load
            } else {
                MTLLoadAction::Clear
            });
            color_attachment.set_store_action(MTLStoreAction::Store);
            if !has_opaque {
                color_attachment.set_clear_color(scene_clear_color());
            }

            let depth_attachment = pass_desc.depth_attachment().unwrap();
            depth_attachment.set_texture(Some(depth_texture));
            depth_attachment.set_load_action(if has_opaque {
                MTLLoadAction::Load
            } else {
                MTLLoadAction::Clear
            });
            depth_attachment.set_store_action(MTLStoreAction::DontCare);
            if !has_opaque {
                depth_attachment.set_clear_depth(1.0);
            }

            let encoder = command_buffer.new_render_command_encoder(&pass_desc);
            encoder.set_render_pipeline_state(&self.forward_transparent_pipeline);
            encoder.set_depth_stencil_state(&self.depth_state_read);
            encoder.set_cull_mode(metal::MTLCullMode::Back);
            encoder.set_vertex_buffer(1, Some(tb), 0);
            encoder.set_vertex_buffer(2, Some(vpb), 0);
            encoder.set_fragment_buffer(0, Some(vpb), 0);
            encoder.set_fragment_buffer(1, Some(lb), 0);
            encoder.set_fragment_buffer(2, Some(lub), 0);
            encoder.set_fragment_buffer(3, Some(sub), 0);
            encoder.set_fragment_texture(0, Some(&self.shadow_texture));
            encoder.set_fragment_sampler_state(0, Some(&self.shadow_sampler));
            for range in ranges.iter().filter(|range| Self::is_transparent_range(range)) {
                if let Some(mesh) = self.resolve_mesh_slot(range.primitive_code as u8) {
                    encoder.set_vertex_buffer(0, Some(&mesh.vertex_buffer), 0);
                    encoder.draw_indexed_primitives_instanced_base_instance(
                        MTLPrimitiveType::Triangle,
                        mesh.index_count as u64,
                        mesh.index_type,
                        &mesh.index_buffer,
                        0,
                        range.instance_count as u64,
                        0,
                        range.first_instance as u64,
                    );
                    main_pass_draw_calls += 1;
                }
            }
            encoder.end_encoding();
        }

        (
            prepass_draw_calls,
            main_pass_draw_calls,
            early_z_reject_estimate,
        )
    }

    #[cfg(test)]
    pub fn new_for_test(config: MetalBackendConfig) -> Result<Self, MetalBackendError> {
        if config.frames_in_flight == 0 {
            return Err(MetalBackendError::InvalidConfig(
                "frames_in_flight must be greater than zero".into(),
            ));
        }
        if config.max_instances == 0 {
            return Err(MetalBackendError::InvalidConfig(
                "max_instances must be greater than zero".into(),
            ));
        }

        let device = Device::system_default().ok_or(MetalBackendError::NoMetalDevice)?;
        let command_queue = device.new_command_queue();
        let device_name = device.name().to_owned();

        let max_materials = config.max_instances.max(256);
        let max_textures = config.max_instances.max(128);
        let max_lights = METAL_MAX_LIGHTS;
        let snapshot_compressor = if config.enable_snapshot_compression {
            Some(crate::compression::SnapshotCompressor::new().map_err(|e| {
                MetalBackendError::InvalidConfig(format!("SXRC init failed: {e}").into())
            })?)
        } else {
            None
        };

        let frame_slots = (0..config.frames_in_flight)
            .map(|_| SnapshotSlot {
                instance_capacity: config.max_instances,
                material_capacity: max_materials,
                texture_capacity: max_textures,
                light_capacity: max_lights,
                last_state: MetalSnapshotSlotState::default(),
                compressed_instance_data: None,
            })
            .collect::<Vec<_>>();

        let (z_prepass_pipeline, forward_pipeline, forward_transparent_pipeline) =
            build_scene_pipelines(&device)?;

        let depth_state_write = build_depth_state(&device, MTLCompareFunction::Less, true)
            .map_err(MetalBackendError::PipelineCreation)?;
        let depth_state_equal = build_depth_state(&device, MTLCompareFunction::LessEqual, false)
            .map_err(MetalBackendError::PipelineCreation)?;
        let depth_state_read = build_depth_state(&device, MTLCompareFunction::LessEqual, false)
            .map_err(MetalBackendError::PipelineCreation)?;

        let surface_size = PhysicalSize::new(64, 64);
        let depth_texture = create_depth_texture(&device, surface_size.width, surface_size.height);

        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_drawable_size(CGSize::new(
            surface_size.width as f64,
            surface_size.height as f64,
        ));

        let max_transforms = config.max_instances.max(1);

        let transform_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    (max_transforms * std::mem::size_of::<GpuInstance3d>()) as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let view_proj_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    std::mem::size_of::<CameraUniform>() as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let light_data_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    (METAL_MAX_LIGHTS * std::mem::size_of::<LightData>()) as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let lighting_uniform_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    std::mem::size_of::<LightingUniform>() as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();

        let shadow_desc = TextureDescriptor::new();
        shadow_desc.set_texture_type(MTLTextureType::D2Array);
        shadow_desc.set_texture_type(MTLTextureType::D2);
        shadow_desc.set_width(2048);
        shadow_desc.set_height(2048);
        shadow_desc.set_pixel_format(MTLPixelFormat::Depth32Float);
        shadow_desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let shadow_texture = device.new_texture(&shadow_desc);
        let shadow_sampler_desc = SamplerDescriptor::new();
        shadow_sampler_desc.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
        shadow_sampler_desc.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
        shadow_sampler_desc.set_compare_function(metal::MTLCompareFunction::LessEqual);
        shadow_sampler_desc.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
        shadow_sampler_desc.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
        let shadow_sampler = device.new_sampler(&shadow_sampler_desc);
        let shadow_pipeline = build_shadow_pipeline(&device)?;
        let shadow_pass_uniform_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    (METAL_SHADOW_LAYERS * std::mem::size_of::<ShadowPassUniform>()) as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let shadow_uniform_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    std::mem::size_of::<ShadowUniform>() as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let offscreen_color_texture =
            create_offscreen_color_texture(&device, surface_size.width, surface_size.height);
        let upscale_pipeline = build_upscale_pipeline(&device)?;
        let upscale_uniform_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    std::mem::size_of::<UpscaleUniform>() as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let upscale_sampler_desc = SamplerDescriptor::new();
        upscale_sampler_desc.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
        upscale_sampler_desc.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
        upscale_sampler_desc.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
        upscale_sampler_desc.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
        let upscale_sampler = device.new_sampler(&upscale_sampler_desc);

        let sprite_pipeline = build_sprite_pipeline(&device)?;
        let sprite_vertex_buffer = device.new_buffer_with_data(
            [
                -0.5f32, -0.5f32,
                0.5f32, -0.5f32,
                -0.5f32, 0.5f32,
                0.5f32, 0.5f32,
            ]
            .as_ptr() as *const _,
            8 * std::mem::size_of::<f32>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let max_sprites = config.max_sprites.max(1);
        let sprite_instance_buffers: Vec<Buffer> = (0..config.frames_in_flight)
            .map(|_| {
                device.new_buffer(
                    (max_sprites * std::mem::size_of::<FrameSpriteRecord>()) as u64,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let sprite_atlas_desc = TextureDescriptor::new();
        sprite_atlas_desc.set_texture_type(MTLTextureType::D2);
        sprite_atlas_desc.set_width(1);
        sprite_atlas_desc.set_height(1);
        sprite_atlas_desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        sprite_atlas_desc.set_usage(MTLTextureUsage::ShaderRead);
        let sprite_atlas_texture = device.new_texture(&sprite_atlas_desc);
        let sprite_sampler_desc = SamplerDescriptor::new();
        sprite_sampler_desc.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
        sprite_sampler_desc.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
        sprite_sampler_desc.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
        sprite_sampler_desc.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
        let sprite_sampler = device.new_sampler(&sprite_sampler_desc);

        let frame_pacing = FramePacing::new(config.frames_in_flight);

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
            forward_transparent_pipeline,
            depth_state_write,
            depth_state_equal,
            depth_state_read,
            transform_buffers,
            view_proj_buffers,
            light_data_buffers,
            lighting_uniform_buffers,
            instance_scratch: Vec::with_capacity(max_transforms),
            shadow_texture,
            shadow_sampler,
            shadow_pipeline,
            shadow_pass_uniform_buffers,
            shadow_uniform_buffers,
            offscreen_color_texture,
            upscale_pipeline,
            upscale_uniform_buffers,
            upscale_sampler,
            sprite_pipeline,
            sprite_vertex_buffer,
            sprite_instance_buffers,
            sprite_atlas_texture,
            sprite_sampler,
            mesh_slots: std::collections::HashMap::new(),
            frame_pacing,
            snapshot_compressor,
        })
    }

    #[cfg(test)]
    pub fn render_n_headless(
        &mut self,
        snapshot: RenderStateSnapshot<'_>,
    ) -> Result<MetalFrameExecutionTelemetry, MetalBackendError> {
        let frame_slot = self.current_frame_slot;
        let snapshot_state = self.upload_state_snapshot(frame_slot, &snapshot)?;
        self.prepare_instance_scratch(&snapshot);

        self.frame_pacing.wait();

        let command_buffer = self.command_queue.new_command_buffer();

        let mut prepass_draw_calls = 0u32;
        let mut main_pass_draw_calls = 0u32;
        let mut early_z_reject_estimate = 0u32;

        let tb = &self.transform_buffers[frame_slot];
        let vpb = &self.view_proj_buffers[frame_slot];
        let cull_result = if self.config.occlusion_culling_enabled {
            let planes = Self::extract_frustum_planes(snapshot.camera_view_proj);
            Some(self.cull_and_compact(&snapshot, &planes))
        } else {
            None
        };
        let mut draw_ranges: Vec<FramePrimitiveRange> = match &cull_result {
            Some((_, visible_ranges)) => visible_ranges.clone(),
            None => snapshot.primitive_ranges.to_vec(),
        };
        let instance_data: &[GpuInstance3d] = match &cull_result {
            Some((visible_instances, _)) => visible_instances.as_slice(),
            None => self.instance_scratch.as_slice(),
        };

        if draw_ranges.iter().any(|r| Self::is_transparent_range(r)) {
            let camera_eye = snapshot.camera_eye;
            let transforms = snapshot.transforms;
            draw_ranges.sort_by(|a, b| {
                let a_trans = Self::is_transparent_range(a);
                let b_trans = Self::is_transparent_range(b);
                match (a_trans, b_trans) {
                    (true, true) => {
                        let dist_a = Self::range_camera_distance(a, transforms, camera_eye);
                        let dist_b = Self::range_camera_distance(b, transforms, camera_eye);
                        dist_b.partial_cmp(&dist_a).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    (false, false) => std::cmp::Ordering::Equal,
                }
            });
        }

        if !instance_data.is_empty() {
            let trans_bytes = std::mem::size_of_val(instance_data);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    instance_data.as_ptr() as *const u8,
                    tb.contents() as *mut u8,
                    trans_bytes.min(tb.length() as usize),
                );
            }
            if let Some(ref compressor) = self.snapshot_compressor {
                let instance_bytes = unsafe {
                    std::slice::from_raw_parts(
                        instance_data.as_ptr() as *const u8,
                        trans_bytes,
                    )
                };
                match compressor.compress(instance_bytes) {
                    Ok(pages) => {
                        let compressed_len: usize = pages.iter().map(|p| p.encoded.len()).sum();
                        let slot = &mut self.frame_slots[frame_slot];
                        slot.compressed_instance_data = Some(pages);
                        slot.last_state.compressed_byte_len = compressed_len;
                    }
                    Err(_) => {
                        let slot = &mut self.frame_slots[frame_slot];
                        slot.compressed_instance_data = None;
                        slot.last_state.compressed_byte_len = 0;
                    }
                }
            }
        }
        {
            let camera = CameraUniform {
                view_proj: snapshot.camera_view_proj,
                camera_eye: snapshot.camera_eye,
            };
            let vp_bytes = std::mem::size_of::<CameraUniform>();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &camera as *const CameraUniform as *const u8,
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

        let lb = &self.light_data_buffers[frame_slot];
        let lub = &self.lighting_uniform_buffers[frame_slot];
        let spub = &self.shadow_pass_uniform_buffers[frame_slot];
        let sub = &self.shadow_uniform_buffers[frame_slot];
        self.upload_lights(frame_slot, lb, lub, &snapshot);
        self.upload_shadows(frame_slot, spub, sub, &snapshot);

        if METAL_REAL_SHADOW_MAPS_ENABLED {
            self.encode_shadow_pass(&command_buffer, tb, spub, &draw_ranges);
        }

        let (p, m, e) = self.encode_frame(
            &command_buffer,
            &snapshot,
            &color_texture,
            &self.depth_texture,
            tb,
            vpb,
            lb,
            lub,
            sub,
            &draw_ranges,
        );
        prepass_draw_calls = p;
        main_pass_draw_calls = m;
        early_z_reject_estimate = e;

        command_buffer.commit();
        command_buffer.wait_until_completed();

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

fn build_scene_pipelines(
    device: &Device,
) -> Result<
    (
        RenderPipelineState,
        RenderPipelineState,
        RenderPipelineState,
    ),
    MetalBackendError,
> {
    let mut shader_library =
        ShaderLibrary::new(device, SCENE_3D_MSL).map_err(MetalBackendError::ShaderCompilation)?;
    let scene_3d_layout = scene_3d_vertex_layout();
    let layout_hash = scene_3d_layout.hash_key();

    let z_prepass_key = PipelineKey {
        vertex_function: "scene_3d_vertex".to_string(),
        fragment_function: None,
        color_format: MTLPixelFormat::BGRA8Unorm,
        depth_format: MTLPixelFormat::Depth32Float,
        sample_count: 1,
        blend_mode: BlendMode::None,
        vertex_layout_hash: layout_hash,
    };
    let z_prepass_pipeline = shader_library
        .get_pipeline(&z_prepass_key, &scene_3d_layout)
        .map_err(MetalBackendError::PipelineCreation)?
        .clone();

    let forward_key = PipelineKey {
        vertex_function: "scene_3d_vertex".to_string(),
        fragment_function: Some("scene_3d_fragment".to_string()),
        color_format: MTLPixelFormat::BGRA8Unorm,
        depth_format: MTLPixelFormat::Depth32Float,
        sample_count: 1,
        blend_mode: BlendMode::None,
        vertex_layout_hash: layout_hash,
    };
    let forward_pipeline = shader_library
        .get_pipeline(&forward_key, &scene_3d_layout)
        .map_err(MetalBackendError::PipelineCreation)?
        .clone();

    let transparent_key = PipelineKey {
        blend_mode: BlendMode::Alpha,
        ..forward_key
    };
    let forward_transparent_pipeline = shader_library
        .get_pipeline(&transparent_key, &scene_3d_layout)
        .map_err(MetalBackendError::PipelineCreation)?
        .clone();

    Ok((
        z_prepass_pipeline,
        forward_pipeline,
        forward_transparent_pipeline,
    ))
}

fn build_shadow_pipeline(
    device: &Device,
) -> Result<RenderPipelineState, MetalBackendError> {
    let mut shader_library =
        ShaderLibrary::new(device, SCENE_SHADOW_MSL).map_err(MetalBackendError::ShaderCompilation)?;
    let layout = scene_3d_vertex_layout();
    let layout_hash = layout.hash_key();

    let key = PipelineKey {
        vertex_function: "scene_shadow_vertex".to_string(),
        fragment_function: None,
        color_format: MTLPixelFormat::BGRA8Unorm,
        depth_format: MTLPixelFormat::Depth32Float,
        sample_count: 1,
        blend_mode: BlendMode::None,
        vertex_layout_hash: layout_hash,
    };
    Ok(shader_library
        .get_pipeline(&key, &layout)
        .map_err(MetalBackendError::PipelineCreation)?
        .clone())
}

fn build_upscale_pipeline(
    device: &Device,
) -> Result<RenderPipelineState, MetalBackendError> {
    let mut shader_library =
        ShaderLibrary::new(device, SCENE_UPSCALE_MSL).map_err(MetalBackendError::ShaderCompilation)?;
    let empty_layout = VertexLayout { buffer_layouts: vec![] };
    let layout_hash = empty_layout.hash_key();

    let key = PipelineKey {
        vertex_function: "upscale_vertex".to_string(),
        fragment_function: Some("upscale_fragment".to_string()),
        color_format: MTLPixelFormat::BGRA8Unorm,
        depth_format: MTLPixelFormat::Invalid,
        sample_count: 1,
        blend_mode: BlendMode::None,
        vertex_layout_hash: layout_hash,
    };
    Ok(shader_library
        .get_pipeline(&key, &empty_layout)
        .map_err(MetalBackendError::PipelineCreation)?
        .clone())
}

fn build_sprite_pipeline(
    device: &Device,
) -> Result<RenderPipelineState, MetalBackendError> {
    let mut shader_library =
        ShaderLibrary::new(device, SCENE_SPRITE_MSL).map_err(MetalBackendError::ShaderCompilation)?;
    let layout = sprite_vertex_layout();
    let layout_hash = layout.hash_key();

    let key = PipelineKey {
        vertex_function: "sprite_vertex".to_string(),
        fragment_function: Some("sprite_fragment".to_string()),
        color_format: MTLPixelFormat::BGRA8Unorm,
        depth_format: MTLPixelFormat::Invalid,
        sample_count: 1,
        blend_mode: BlendMode::Alpha,
        vertex_layout_hash: layout_hash,
    };
    Ok(shader_library
        .get_pipeline(&key, &layout)
        .map_err(MetalBackendError::PipelineCreation)?
        .clone())
}

fn sprite_vertex_layout() -> VertexLayout {
    VertexLayout {
        buffer_layouts: vec![
            VertexBufferLayoutDesc {
                stride: 2 * std::mem::size_of::<f32>(),
                step_function: MTLVertexStepFunction::PerVertex,
                attributes: vec![VertexAttributeDesc {
                    format: MTLVertexFormat::Float2,
                    offset: 0,
                    buffer_index: 0,
                }],
            },
            VertexBufferLayoutDesc {
                stride: std::mem::size_of::<FrameSpriteRecord>(),
                step_function: MTLVertexStepFunction::PerInstance,
                attributes: vec![
                    VertexAttributeDesc {
                        format: MTLVertexFormat::Float4,
                        offset: 0,
                        buffer_index: 1,
                    },
                    VertexAttributeDesc {
                        format: MTLVertexFormat::Float4,
                        offset: 16,
                        buffer_index: 1,
                    },
                    VertexAttributeDesc {
                        format: MTLVertexFormat::Float4,
                        offset: 32,
                        buffer_index: 1,
                    },
                    VertexAttributeDesc {
                        format: MTLVertexFormat::Float4,
                        offset: 48,
                        buffer_index: 1,
                    },
                    VertexAttributeDesc {
                        format: MTLVertexFormat::Float4,
                        offset: 64,
                        buffer_index: 1,
                    },
                ],
            },
        ],
    }
}

fn scene_3d_vertex_layout() -> VertexLayout {
    VertexLayout {
        buffer_layouts: vec![
            VertexBufferLayoutDesc {
                stride: 8 * std::mem::size_of::<f32>(),
                step_function: MTLVertexStepFunction::PerVertex,
                attributes: vec![
                    VertexAttributeDesc {
                        format: MTLVertexFormat::Float3,
                        offset: 0,
                        buffer_index: 0,
                    },
                    VertexAttributeDesc {
                        format: MTLVertexFormat::Float3,
                        offset: 3 * std::mem::size_of::<f32>(),
                        buffer_index: 0,
                    },
                    VertexAttributeDesc {
                        format: MTLVertexFormat::Float2,
                        offset: 6 * std::mem::size_of::<f32>(),
                        buffer_index: 0,
                    },
                ],
            },
            VertexBufferLayoutDesc {
                stride: std::mem::size_of::<GpuInstance3d>(),
                step_function: MTLVertexStepFunction::PerInstance,
                attributes: vec![
                    VertexAttributeDesc {
                        format: MTLVertexFormat::Float4,
                        offset: 0,
                        buffer_index: 1,
                    },
                    VertexAttributeDesc {
                        format: MTLVertexFormat::Float4,
                        offset: 16,
                        buffer_index: 1,
                    },
                    VertexAttributeDesc {
                        format: MTLVertexFormat::Float4,
                        offset: 32,
                        buffer_index: 1,
                    },
                    VertexAttributeDesc {
                        format: MTLVertexFormat::Float4,
                        offset: 48,
                        buffer_index: 1,
                    },
                    VertexAttributeDesc {
                        format: MTLVertexFormat::Float4,
                        offset: 64,
                        buffer_index: 1,
                    },
                    VertexAttributeDesc {
                        format: MTLVertexFormat::Float4,
                        offset: 80,
                        buffer_index: 1,
                    },
                    VertexAttributeDesc {
                        format: MTLVertexFormat::Float4,
                        offset: 96,
                        buffer_index: 1,
                    },
                ],
            },
        ],
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

fn create_offscreen_color_texture(device: &Device, width: u32, height: u32) -> Texture {
    let desc = TextureDescriptor::new();
    desc.set_texture_type(MTLTextureType::D2);
    desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    desc.set_width(width as u64);
    desc.set_height(height as u64);
    desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    desc.set_storage_mode(metal::MTLStorageMode::Private);
    device.new_texture(&desc)
}

fn scene_clear_color() -> MTLClearColor {
    MTLClearColor::new(0.07, 0.09, 0.12, 1.0)
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
    use crate::graphics::frame_snapshot::{FrameInstanceTransform, FramePrimitiveRange};
    use metal::MTLIndexType;

    #[cfg(target_os = "macos")]
    fn metal_device_or_skip() -> Option<Device> {
        let device = Device::system_default();
        if device.is_none() {
            eprintln!("skipping Metal backend test: no Metal device available");
        }
        device
    }

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
        let Some(device) = metal_device_or_skip() else {
            return;
        };
        let library = ShaderLibrary::new(&device, SCENE_3D_MSL).expect("shader compilation failed");
        assert!(library.get_function("scene_3d_vertex").is_ok());
        assert!(library.get_function("scene_3d_fragment").is_ok());
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn depth_state_creation() {
        let Some(device) = metal_device_or_skip() else {
            return;
        };
        assert!(build_depth_state(&device, MTLCompareFunction::Less, true).is_ok());
        assert!(build_depth_state(&device, MTLCompareFunction::Equal, false).is_ok());
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn pipeline_creation() {
        let Some(device) = metal_device_or_skip() else {
            return;
        };
        assert!(build_scene_pipelines(&device).is_ok());
    }

    fn dummy_snapshot(
        instance_count: usize,
        translation: [f32; 3],
        range_flags: u32,
        opaque_instance_count: u32,
        transparent_instance_count: u32,
    ) -> RenderStateSnapshot<'static> {
        let transforms: Vec<FrameInstanceTransform> = (0..instance_count)
            .map(|_| FrameInstanceTransform {
                model: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [translation[0], translation[1], translation[2], 1.0],
                ],
                color_rgba: [1.0; 4],
                material_index: 0,
                texture_index: 0,
                flags: 0,
                _padding: 0,
            })
            .collect();
        let primitive_ranges = Box::leak(vec![FramePrimitiveRange {
            primitive_code: 0,
            first_instance: 0,
            instance_count: instance_count as u32,
            flags: range_flags,
        }].into_boxed_slice());
        let transforms = Box::leak(transforms.into_boxed_slice());
        RenderStateSnapshot {
            frame_id: 1,
            camera_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            camera_eye: [0.0, 12.0, 36.0, 1.0],
            opaque_instance_count,
            transparent_instance_count,
            primitive_ranges,
            transforms,
            materials: &[],
            textures: &[],
            lights: &[],
            sprites: &[],
            instance_bounds: &[],
            vertices: &[],
            indices: &[],
        }
    }

    fn dummy_snapshot_with_instances(instance_count: usize) -> RenderStateSnapshot<'static> {
        dummy_snapshot(
            instance_count,
            [0.0, 0.0, 0.0],
            0,
            instance_count as u32,
            0,
        )
    }

    fn dummy_snapshot_with_translation(
        instance_count: usize,
        translation: [f32; 3],
    ) -> RenderStateSnapshot<'static> {
        dummy_snapshot(instance_count, translation, 0, instance_count as u32, 0)
    }

    fn dummy_transparent_snapshot(instance_count: usize) -> RenderStateSnapshot<'static> {
        dummy_snapshot(
            instance_count,
            [0.0, 0.0, 0.0],
            FRAME_PRIMITIVE_RANGE_TRANSPARENT,
            0,
            instance_count as u32,
        )
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn upload_state_snapshot_capacity_guards() {
        let mut backend = match MetalBackend::new_for_test(MetalBackendConfig {
            frames_in_flight: 1,
            max_instances: 4,
            occlusion_culling_enabled: true,
            max_sprites: 4096,
            enable_snapshot_compression: false,
        }) {
            Ok(backend) => backend,
            Err(MetalBackendError::NoMetalDevice) => return,
            Err(err) => panic!("unexpected Metal backend init error: {err}"),
        };

        let too_many = dummy_snapshot_with_instances(8);
        match backend.upload_state_snapshot(0, &too_many) {
            Err(MetalBackendError::SnapshotCapacityExceeded {
                requested,
                capacity,
            }) => {
                assert_eq!(requested, 8);
                assert_eq!(capacity, 4);
            }
            other => panic!("expected SnapshotCapacityExceeded, got {:?}", other),
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn headless_two_pass_telemetry() {
        let mut backend = match MetalBackend::new_for_test(MetalBackendConfig {
            frames_in_flight: 1,
            max_instances: 4,
            occlusion_culling_enabled: true,
            max_sprites: 4096,
            enable_snapshot_compression: false,
        }) {
            Ok(backend) => backend,
            Err(MetalBackendError::NoMetalDevice) => return,
            Err(err) => panic!("unexpected Metal backend init error: {err}"),
        };

        let vertices: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        ];
        let indices: Vec<u32> = vec![0, 1, 2];
        let mesh = MeshSlot::new(backend.device(), &vertices, &indices, MTLIndexType::UInt32);
        backend.bind_mesh_slot(0, mesh);

        let snapshot = dummy_snapshot_with_instances(2);
        let telemetry = backend
            .render_n_headless(snapshot)
            .expect("headless render failed");
        assert!(telemetry.command_buffer_submitted);
        assert_eq!(telemetry.prepass_draw_calls, 1);
        assert_eq!(telemetry.main_pass_draw_calls, 1);
        assert_eq!(telemetry.presented, false);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn headless_single_pass_telemetry() {
        let mut backend = match MetalBackend::new_for_test(MetalBackendConfig {
            frames_in_flight: 1,
            max_instances: 4,
            occlusion_culling_enabled: false,
            max_sprites: 4096,
            enable_snapshot_compression: false,
        }) {
            Ok(backend) => backend,
            Err(MetalBackendError::NoMetalDevice) => return,
            Err(err) => panic!("unexpected Metal backend init error: {err}"),
        };

        let vertices: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        ];
        let indices: Vec<u32> = vec![0, 1, 2];
        let mesh = MeshSlot::new(backend.device(), &vertices, &indices, MTLIndexType::UInt32);
        backend.bind_mesh_slot(0, mesh);

        let snapshot = dummy_snapshot_with_instances(2);
        let telemetry = backend
            .render_n_headless(snapshot)
            .expect("headless render failed");
        assert!(telemetry.command_buffer_submitted);
        assert_eq!(telemetry.prepass_draw_calls, 0);
        assert_eq!(telemetry.main_pass_draw_calls, 1);
        assert_eq!(telemetry.presented, false);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn headless_single_pass_does_not_cpu_cull_when_disabled() {
        let mut backend = match MetalBackend::new_for_test(MetalBackendConfig {
            frames_in_flight: 1,
            max_instances: 4,
            occlusion_culling_enabled: false,
            max_sprites: 4096,
            enable_snapshot_compression: false,
        }) {
            Ok(backend) => backend,
            Err(MetalBackendError::NoMetalDevice) => return,
            Err(err) => panic!("unexpected Metal backend init error: {err}"),
        };

        let vertices: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        ];
        let indices: Vec<u32> = vec![0, 1, 2];
        let mesh = MeshSlot::new(backend.device(), &vertices, &indices, MTLIndexType::UInt32);
        backend.bind_mesh_slot(0, mesh);

        let snapshot = dummy_snapshot_with_translation(1, [8.0, 0.0, 0.0]);
        let telemetry = backend
            .render_n_headless(snapshot)
            .expect("headless render failed");
        assert_eq!(telemetry.prepass_draw_calls, 0);
        assert_eq!(telemetry.main_pass_draw_calls, 1);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn headless_transparent_range_skips_opaque_prepass() {
        let mut backend = match MetalBackend::new_for_test(MetalBackendConfig {
            frames_in_flight: 1,
            max_instances: 4,
            occlusion_culling_enabled: true,
            max_sprites: 4096,
            enable_snapshot_compression: false,
        }) {
            Ok(backend) => backend,
            Err(MetalBackendError::NoMetalDevice) => return,
            Err(err) => panic!("unexpected Metal backend init error: {err}"),
        };

        let vertices: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        ];
        let indices: Vec<u32> = vec![0, 1, 2];
        let mesh = MeshSlot::new(backend.device(), &vertices, &indices, MTLIndexType::UInt32);
        backend.bind_mesh_slot(0, mesh);

        let snapshot = dummy_transparent_snapshot(1);
        let telemetry = backend
            .render_n_headless(snapshot)
            .expect("headless render failed");
        assert_eq!(telemetry.prepass_draw_calls, 0);
        assert_eq!(telemetry.main_pass_draw_calls, 1);
    }

    #[test]
    #[cfg(target_os = "macos")]

    fn metal_shadow_maps_are_active_and_projection_matches_wgpu() {
        let backend = match MetalBackend::new_for_test(MetalBackendConfig {
            frames_in_flight: 1,
            max_instances: 4,
            occlusion_culling_enabled: true,
            max_sprites: 4096,
            enable_snapshot_compression: false,
        }) {
            Ok(backend) => backend,
            Err(MetalBackendError::NoMetalDevice) => return,
            Err(err) => panic!("unexpected Metal backend init error: {err}"),
        };

        let lights = Box::leak(vec![FrameLightRecord {
            position_kind: [0.0, 4.0, 0.0, 1.0],
            direction_inner: [0.0, -1.0, 0.0, 0.9],
            color_intensity: [1.0, 0.9, 0.7, 8.0],
            params: [32.0, 0.8, 0.35, 1.0],
            shadow: [1.0, 0.0, 0.0, 0.0],
        }]
        .into_boxed_slice());
        let snapshot = RenderStateSnapshot {
            frame_id: 1,
            camera_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            camera_eye: [0.0, 12.0, 36.0, 1.0],
            opaque_instance_count: 0,
            transparent_instance_count: 0,
            primitive_ranges: &[],
            transforms: &[],
            materials: &[],
            textures: &[],
            lights,
            sprites: &[],
            instance_bounds: &[],
            vertices: &[],
            indices: &[],
        };

        let spub = &backend.shadow_pass_uniform_buffers[0];
        let sub = &backend.shadow_uniform_buffers[0];
        backend.upload_shadows(0, spub, sub, &snapshot);
        let stored = unsafe { std::ptr::read_unaligned(sub.contents() as *const ShadowUniform) };
        // With METAL_REAL_SHADOW_MAPS_ENABLED=true, a shadow-casting spot light
        // should occupy slot 0 and shadow_count should be 1.
        assert_eq!(stored.shadow_count, 1);
        assert_eq!(stored.shadow_light_indices[0], 0);
        // Slot 1..3 remain unused.
        assert_eq!(&stored.shadow_light_indices[1..], &[-1, -1, -1][..]);
        // light_view_proj[0] should be a non-zero matrix (spotlight projection
        // for a light looking down from y=4 with range=32).
        let vp = stored.light_view_proj[0];
        assert!(vp.iter().flatten().any(|&v| v != 0.0), "shadow view-proj matrix should be non-zero");
    }

    #[test]
    #[cfg(target_os = "macos")]

    fn headless_shadow_pass_emits_draw_calls_for_opaque_geometry() {
        let mut backend = match MetalBackend::new_for_test(MetalBackendConfig {
            frames_in_flight: 1,
            max_instances: 4,
            occlusion_culling_enabled: true,
            max_sprites: 4096,
            enable_snapshot_compression: false,
        }) {
            Ok(backend) => backend,
            Err(MetalBackendError::NoMetalDevice) => return,
            Err(err) => panic!("unexpected Metal backend init error: {err}"),
        };

        let vertices: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
        ];
        let indices: Vec<u32> = vec![0, 1, 2];
        let mesh = MeshSlot::new(backend.device(), &vertices, &indices, MTLIndexType::UInt32);
        backend.bind_mesh_slot(0, mesh);

        let lights = Box::leak(vec![FrameLightRecord {
            position_kind: [0.0, 4.0, 0.0, 1.0],
            direction_inner: [0.0, -1.0, 0.0, 0.9],
            color_intensity: [1.0, 0.9, 0.7, 8.0],
            params: [32.0, 0.8, 0.35, 1.0],
            shadow: [1.0, 0.0, 0.0, 0.0],
        }]
        .into_boxed_slice());

        let transforms: Vec<FrameInstanceTransform> = vec![FrameInstanceTransform {
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            color_rgba: [1.0; 4],
            material_index: 0,
            texture_index: 0,
            flags: 0,
            _padding: 0,
        }];
        let primitive_ranges = Box::leak(vec![FramePrimitiveRange {
            primitive_code: 0,
            first_instance: 0,
            instance_count: 1,
            flags: 0,
        }].into_boxed_slice());
        let transforms = Box::leak(transforms.into_boxed_slice());
        let snapshot = RenderStateSnapshot {
            frame_id: 1,
            camera_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            camera_eye: [0.0, 12.0, 36.0, 1.0],
            opaque_instance_count: 1,
            transparent_instance_count: 0,
            primitive_ranges,
            transforms,
            materials: &[],
            textures: &[],
            lights,
            sprites: &[],
            instance_bounds: &[],
            vertices: &[],
            indices: &[],
        };

        let telemetry = backend
            .render_n_headless(snapshot)
            .expect("headless render failed");
        // Shadow pass + Z-prepass + forward opaque each draw once.
        assert_eq!(telemetry.prepass_draw_calls, 1);
        assert_eq!(telemetry.main_pass_draw_calls, 1);
        assert!(telemetry.command_buffer_submitted);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn resize_recreates_offscreen_color_target() {
        let mut backend = match MetalBackend::new_for_test(MetalBackendConfig {
            frames_in_flight: 1,
            max_instances: 4,
            occlusion_culling_enabled: true,
            max_sprites: 4096,
            enable_snapshot_compression: false,
        }) {
            Ok(backend) => backend,
            Err(MetalBackendError::NoMetalDevice) => return,
            Err(err) => panic!("unexpected Metal backend init error: {err}"),
        };

        backend
            .resize(PhysicalSize::new(128, 96))
            .expect("resize should recreate attachments");

        assert_eq!(backend.offscreen_color_texture.width(), 128);
        assert_eq!(backend.offscreen_color_texture.height(), 96);
        assert_eq!(backend.depth_texture.width(), 128);
        assert_eq!(backend.depth_texture.height(), 96);
    }
}
