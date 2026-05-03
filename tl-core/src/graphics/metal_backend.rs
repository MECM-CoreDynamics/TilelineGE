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
use crate::graphics::frame_snapshot::{FramePrimitiveRange, RenderStateSnapshot};

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
            frames_in_flight: 3,
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

use super::metal::shader_library::{
    BlendMode, PipelineKey, ShaderLibrary, VertexAttributeDesc, VertexBufferLayoutDesc,
    VertexLayout,
};
use super::metal::shaders::SCENE_3D_MSL;

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
struct ShadowUniform {
    light_view_proj: [[[f32; 4]; 4]; METAL_SHADOW_LAYERS],
    shadow_light_indices: [i32; 4],
    shadow_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const METAL_MAX_LIGHTS: usize = 64;
const METAL_SHADOW_LAYERS: usize = 4;

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
    transform_buffer: Option<Buffer>,
    view_proj_buffer: Option<Buffer>,
    instance_scratch: Vec<GpuInstance3d>,
    stub_buffer: Buffer,
    stub_shadow_texture: Texture,
    stub_shadow_sampler: SamplerState,
    mesh_slots: std::collections::HashMap<u8, MeshSlot>,
    frame_pacing: FramePacing,
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
        let max_lights = METAL_MAX_LIGHTS;
        let frame_slots = (0..config.frames_in_flight)
            .map(|_| SnapshotSlot {
                instance_capacity: config.max_instances,
                material_capacity: max_materials,
                texture_capacity: max_textures,
                light_capacity: max_lights,
                last_state: MetalSnapshotSlotState::default(),
            })
            .collect::<Vec<_>>();

        let (z_prepass_pipeline, forward_pipeline) = build_scene_pipelines(&device)?;

        let depth_state_write = build_depth_state(&device, MTLCompareFunction::Less, true)
            .map_err(MetalBackendError::PipelineCreation)?;
        let depth_state_equal = build_depth_state(&device, MTLCompareFunction::Equal, false)
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

        let max_transforms = config.max_instances.max(1);

        let transform_buffer = Some(device.new_buffer(
            (max_transforms * std::mem::size_of::<GpuInstance3d>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        ));
        let view_proj_buffer = Some(device.new_buffer(
            std::mem::size_of::<CameraUniform>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        ));

        let stub_buffer = device.new_buffer(
            metal_stub_fragment_buffer_len(),
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        );
        zero_shared_buffer(&stub_buffer);
        let stub_shadow_desc = TextureDescriptor::new();
        stub_shadow_desc.set_texture_type(MTLTextureType::D2Array);
        stub_shadow_desc.set_width(1);
        stub_shadow_desc.set_height(1);
        stub_shadow_desc.set_array_length(1);
        stub_shadow_desc.set_pixel_format(MTLPixelFormat::Depth32Float);
        stub_shadow_desc.set_usage(MTLTextureUsage::ShaderRead);
        let stub_shadow_texture = device.new_texture(&stub_shadow_desc);
        let stub_shadow_sampler_desc = SamplerDescriptor::new();
        stub_shadow_sampler_desc.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
        stub_shadow_sampler_desc.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
        stub_shadow_sampler_desc.set_compare_function(metal::MTLCompareFunction::LessEqual);
        stub_shadow_sampler_desc.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
        stub_shadow_sampler_desc.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
        let stub_shadow_sampler = device.new_sampler(&stub_shadow_sampler_desc);

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
            depth_state_write,
            depth_state_equal,
            transform_buffer,
            view_proj_buffer,
            instance_scratch: Vec::with_capacity(max_transforms),
            stub_buffer,
            stub_shadow_texture,
            stub_shadow_sampler,
            mesh_slots: std::collections::HashMap::new(),
            frame_pacing,
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
            if let (Some(tb), Some(vpb)) = (
                self.transform_buffer.as_ref(),
                self.view_proj_buffer.as_ref(),
            ) {
                let (visible_instances, visible_ranges) = self.cull_and_compact(&snapshot);

                if !visible_instances.is_empty() {
                    let trans_bytes = std::mem::size_of_val(visible_instances.as_slice());
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            visible_instances.as_ptr() as *const u8,
                            tb.contents() as *mut u8,
                            trans_bytes.min(tb.length() as usize),
                        );
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

                let (p, m, e) = self.encode_frame(
                    &command_buffer,
                    &snapshot,
                    drawable_ref.texture(),
                    &self.depth_texture,
                    tb,
                    vpb,
                    &visible_ranges,
                );
                prepass_draw_calls = p;
                main_pass_draw_calls = m;
                early_z_reject_estimate = e;
            }

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
            p[4][c] = view_proj[c][2] + view_proj[c][3]; // near   = row2 + row3
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
    fn is_sphere_visible(center: [f32; 3], radius: f32, planes: &[[f32; 4]; 6]) -> bool {
        for plane in planes {
            let dist = plane[0] * center[0] + plane[1] * center[1] + plane[2] * center[2] + plane[3];
            if dist < -radius {
                return false;
            }
        }
        true
    }

    #[inline]
    fn primitive_radius(primitive_code: u32) -> f32 {
        match primitive_code {
            0 => 1.0,               // unit sphere
            1 => 0.866_025_4,       // sqrt(3)/2: circumscribed sphere of unit box
            _ => 1.0,               // custom mesh: conservative
        }
    }

    /// CPU-side frustum cull + compaction.
    ///
    /// Returns a compacted instance buffer containing only visible instances,
    /// and rewritten primitive ranges that point into the compacted buffer.
    fn cull_and_compact(
        &self,
        snapshot: &RenderStateSnapshot<'_>,
    ) -> (Vec<GpuInstance3d>, Vec<FramePrimitiveRange>) {
        let planes = Self::extract_frustum_planes(snapshot.camera_view_proj);
        let mut visible = Vec::with_capacity(self.instance_scratch.len());
        let mut ranges = Vec::with_capacity(snapshot.primitive_ranges.len());

        for range in snapshot.primitive_ranges {
            let base_radius = Self::primitive_radius(range.primitive_code);
            let start = range.first_instance;
            let end = range.first_instance + range.instance_count;
            let mut block_start: Option<u32> = None;

            for i in start..end {
                let inst = &self.instance_scratch[i as usize];
                let center = [inst.model_col3[0], inst.model_col3[1], inst.model_col3[2]];
                let sx = (inst.model_col0[0] * inst.model_col0[0]
                    + inst.model_col0[1] * inst.model_col0[1]
                    + inst.model_col0[2] * inst.model_col0[2])
                    .sqrt();
                let sy = (inst.model_col1[0] * inst.model_col1[0]
                    + inst.model_col1[1] * inst.model_col1[1]
                    + inst.model_col1[2] * inst.model_col1[2])
                    .sqrt();
                let sz = (inst.model_col2[0] * inst.model_col2[0]
                    + inst.model_col2[1] * inst.model_col2[1]
                    + inst.model_col2[2] * inst.model_col2[2])
                    .sqrt();
                let radius = base_radius * sx.max(sy).max(sz);

                if Self::is_sphere_visible(center, radius, &planes) {
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

    fn encode_frame(
        &self,
        command_buffer: &metal::CommandBufferRef,
        _snapshot: &RenderStateSnapshot<'_>,
        color_texture: &metal::TextureRef,
        depth_texture: &metal::TextureRef,
        tb: &Buffer,
        vpb: &Buffer,
        ranges: &[FramePrimitiveRange],
    ) -> (u32, u32, u32) {
        let mut prepass_draw_calls = 0u32;
        let mut main_pass_draw_calls = 0u32;
        let mut early_z_reject_estimate = 0u32;

        if self.config.occlusion_culling_enabled && !ranges.is_empty() {
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
            encoder.set_vertex_buffer(1, Some(tb), 0);
            encoder.set_vertex_buffer(2, Some(vpb), 0);
            for range in ranges {
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
            encoder.set_vertex_buffer(1, Some(tb), 0);
            encoder.set_vertex_buffer(2, Some(vpb), 0);
            encoder.set_fragment_buffer(0, Some(vpb), 0);
            encoder.set_fragment_buffer(1, Some(&self.stub_buffer), 0);
            encoder.set_fragment_buffer(2, Some(&self.stub_buffer), 0);
            encoder.set_fragment_buffer(3, Some(&self.stub_buffer), 0);
            encoder.set_fragment_texture(0, Some(&self.stub_shadow_texture));
            encoder.set_fragment_sampler_state(0, Some(&self.stub_shadow_sampler));
            for range in ranges {
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
            encoder.set_vertex_buffer(1, Some(tb), 0);
            encoder.set_vertex_buffer(2, Some(vpb), 0);
            encoder.set_fragment_buffer(0, Some(vpb), 0);
            encoder.set_fragment_buffer(1, Some(&self.stub_buffer), 0);
            encoder.set_fragment_buffer(2, Some(&self.stub_buffer), 0);
            encoder.set_fragment_buffer(3, Some(&self.stub_buffer), 0);
            encoder.set_fragment_texture(0, Some(&self.stub_shadow_texture));
            encoder.set_fragment_sampler_state(0, Some(&self.stub_shadow_sampler));
            for range in ranges {
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
        let max_lights = METAL_MAX_LIGHTS;
        let frame_slots = (0..config.frames_in_flight)
            .map(|_| SnapshotSlot {
                instance_capacity: config.max_instances,
                material_capacity: max_materials,
                texture_capacity: max_textures,
                light_capacity: max_lights,
                last_state: MetalSnapshotSlotState::default(),
            })
            .collect::<Vec<_>>();

        let (z_prepass_pipeline, forward_pipeline) = build_scene_pipelines(&device)?;

        let depth_state_write = build_depth_state(&device, MTLCompareFunction::Less, true)
            .map_err(MetalBackendError::PipelineCreation)?;
        let depth_state_equal = build_depth_state(&device, MTLCompareFunction::Equal, false)
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

        let transform_buffer = Some(device.new_buffer(
            (max_transforms * std::mem::size_of::<GpuInstance3d>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        ));
        let view_proj_buffer = Some(device.new_buffer(
            std::mem::size_of::<CameraUniform>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        ));

        let stub_buffer = device.new_buffer(
            metal_stub_fragment_buffer_len(),
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        );
        zero_shared_buffer(&stub_buffer);
        let stub_shadow_desc = TextureDescriptor::new();
        stub_shadow_desc.set_texture_type(MTLTextureType::D2Array);
        stub_shadow_desc.set_width(1);
        stub_shadow_desc.set_height(1);
        stub_shadow_desc.set_array_length(1);
        stub_shadow_desc.set_pixel_format(MTLPixelFormat::Depth32Float);
        stub_shadow_desc.set_usage(MTLTextureUsage::ShaderRead);
        let stub_shadow_texture = device.new_texture(&stub_shadow_desc);
        let stub_shadow_sampler_desc = SamplerDescriptor::new();
        stub_shadow_sampler_desc.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
        stub_shadow_sampler_desc.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
        stub_shadow_sampler_desc.set_compare_function(metal::MTLCompareFunction::LessEqual);
        stub_shadow_sampler_desc.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
        stub_shadow_sampler_desc.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
        let stub_shadow_sampler = device.new_sampler(&stub_shadow_sampler_desc);

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
            depth_state_write,
            depth_state_equal,
            transform_buffer,
            view_proj_buffer,
            instance_scratch: Vec::with_capacity(max_transforms),
            stub_buffer,
            stub_shadow_texture,
            stub_shadow_sampler,
            mesh_slots: std::collections::HashMap::new(),
            frame_pacing,
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

        if let (Some(tb), Some(vpb)) = (
            self.transform_buffer.as_ref(),
            self.view_proj_buffer.as_ref(),
        ) {
            let (visible_instances, visible_ranges) = self.cull_and_compact(&snapshot);

            if !visible_instances.is_empty() {
                let trans_bytes = std::mem::size_of_val(visible_instances.as_slice());
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        visible_instances.as_ptr() as *const u8,
                        tb.contents() as *mut u8,
                        trans_bytes.min(tb.length() as usize),
                    );
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

            let (p, m, e) = self.encode_frame(
                &command_buffer,
                &snapshot,
                &color_texture,
                &self.depth_texture,
                tb,
                vpb,
                &visible_ranges,
            );
            prepass_draw_calls = p;
            main_pass_draw_calls = m;
            early_z_reject_estimate = e;
        }

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
) -> Result<(RenderPipelineState, RenderPipelineState), MetalBackendError> {
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

    Ok((z_prepass_pipeline, forward_pipeline))
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

fn zero_shared_buffer(buffer: &Buffer) {
    unsafe {
        std::ptr::write_bytes(buffer.contents() as *mut u8, 0, buffer.length() as usize);
    }
}

fn metal_stub_fragment_buffer_len() -> u64 {
    std::mem::size_of::<ShadowUniform>()
        .max(std::mem::size_of::<LightingUniform>())
        .max(std::mem::size_of::<
            crate::graphics::frame_snapshot::FrameLightRecord,
        >())
        .max(512) as u64
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

    fn dummy_snapshot_with_instances(instance_count: usize) -> RenderStateSnapshot<'static> {
        let transforms: Vec<FrameInstanceTransform> = (0..instance_count)
            .map(|_| FrameInstanceTransform {
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
            })
            .collect();
        let primitive_ranges = Box::leak(vec![FramePrimitiveRange {
            primitive_code: 0,
            first_instance: 0,
            instance_count: instance_count as u32,
            flags: 0,
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
            opaque_instance_count: instance_count as u32,
            transparent_instance_count: 0,
            primitive_ranges,
            transforms,
            materials: &[],
            textures: &[],
            lights: &[],
            instance_bounds: &[],
            vertices: &[],
            indices: &[],
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn upload_state_snapshot_capacity_guards() {
        let mut backend = match MetalBackend::new_for_test(MetalBackendConfig {
            frames_in_flight: 1,
            max_instances: 4,
            occlusion_culling_enabled: true,
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
}
