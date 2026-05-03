//! Backend-agnostic per-frame render snapshot records.
//!
//! These payload types are shared by Vulkan and Metal runtime backends.

/// Axis-aligned bounding box per instance for occlusion-culling.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct InstanceBounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// Per-instance transform payload uploaded into backend-visible snapshot buffers.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameInstanceTransform {
    pub model: [[f32; 4]; 4],
    pub color_rgba: [f32; 4],
    pub material_index: u32,
    pub texture_index: u32,
    pub flags: u32,
    pub _padding: u32,
}

/// One compact material record referenced by frame instances.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FrameMaterialRecord {
    pub material_params: [f32; 4],
    pub emissive_rgb: [f32; 3],
    pub shading_code: u32,
    pub texture_index: u32,
    pub flags: u32,
    pub _padding: [u32; 2],
}

/// One compact texture indirection record referenced by materials/instances.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameTextureRecord {
    pub texture_slot: u32,
    pub sampler_code: u32,
    pub flags: u32,
    pub _padding: u32,
}

/// One compact scene light record prepared for renderer consumption.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FrameLightRecord {
    pub position_kind: [f32; 4],
    pub direction_inner: [f32; 4],
    pub color_intensity: [f32; 4],
    pub params: [f32; 4],
    pub shadow: [f32; 4],
}

/// One compact sprite instance record prepared for renderer consumption.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FrameSpriteRecord {
    pub translate_size: [f32; 4],
    pub rot_z: [f32; 4],
    pub color: [f32; 4],
    pub atlas_rect: [f32; 4],
    pub kind_params: [f32; 4],
}

/// Transparent flag for one primitive draw range inside a frame snapshot.
pub const FRAME_PRIMITIVE_RANGE_TRANSPARENT: u32 = 1 << 0;

/// CPU-side primitive run metadata for raw backends that issue multiple mesh draws per snapshot.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FramePrimitiveRange {
    pub primitive_code: u32,
    pub first_instance: u32,
    pub instance_count: u32,
    pub flags: u32,
}

/// Render-visible snapshot prepared by MPS / scene-build and consumed by render backends.
#[derive(Debug, Clone, Copy)]
pub struct RenderStateSnapshot<'a> {
    pub frame_id: u64,
    pub camera_view_proj: [[f32; 4]; 4],
    pub camera_eye: [f32; 4],
    pub opaque_instance_count: u32,
    pub transparent_instance_count: u32,
    pub primitive_ranges: &'a [FramePrimitiveRange],
    pub transforms: &'a [FrameInstanceTransform],
    pub materials: &'a [FrameMaterialRecord],
    pub textures: &'a [FrameTextureRecord],
    pub lights: &'a [FrameLightRecord],
    pub sprites: &'a [FrameSpriteRecord],
    /// Per-instance world-space AABB bounds [min_x, min_y, min_z, max_x, max_y, max_z].
    /// Empty when the backend does not support occlusion culling.
    pub instance_bounds: &'a [[f32; 6]],
    /// Shared interleaved vertex buffer: position (3) + normal (3) + uv (2).
    pub vertices: &'a [f32],
    /// Shared index buffer.
    pub indices: &'a [u32],
}
