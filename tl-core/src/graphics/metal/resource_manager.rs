//! Persistent GPU resource manager: buffers, textures, samplers, mesh slots, and shader library.

#![cfg(target_os = "macos")]

use metal::{
    Buffer, Device, MTLPixelFormat, MTLResourceOptions, SamplerDescriptor, SamplerState, Texture,
    TextureDescriptor,
};
use std::collections::HashMap;

use super::mesh_slot::MeshSlot;
use super::shader_library::ShaderLibrary;

pub struct ResourceManager {
    device: Device,
    shader_library: ShaderLibrary,
    mesh_slots: HashMap<u8, MeshSlot>,
    buffers: HashMap<String, Buffer>,
    textures: HashMap<String, Texture>,
    samplers: HashMap<String, SamplerState>,
}

impl ResourceManager {
    pub fn new(device: &Device, shader_source: &str) -> Result<Self, String> {
        let shader_library = ShaderLibrary::new(device, shader_source)?;
        Ok(Self {
            device: device.clone(),
            shader_library,
            mesh_slots: HashMap::new(),
            buffers: HashMap::new(),
            textures: HashMap::new(),
            samplers: HashMap::new(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn shader_library(&mut self) -> &mut ShaderLibrary {
        &mut self.shader_library
    }

    // ------------------------------------------------------------------
    // Mesh slots
    // ------------------------------------------------------------------

    pub fn bind_mesh_slot(&mut self, slot: u8, mesh: MeshSlot) {
        self.mesh_slots.insert(slot, mesh);
    }

    pub fn mesh_slot(&self, slot: u8) -> Option<&MeshSlot> {
        self.mesh_slots.get(&slot)
    }

    // ------------------------------------------------------------------
    // Buffers (power-of-two growth, shared storage)
    // ------------------------------------------------------------------

    /// Return an existing buffer if it is large enough, otherwise recreate it.
    pub fn ensure_buffer(&mut self, name: &str, min_size: usize) -> &Buffer {
        let needs_create = match self.buffers.get(name) {
            Some(buf) => (buf.length() as usize) < min_size,
            None => true,
        };
        if needs_create {
            let size = min_size.next_power_of_two().max(1024);
            let buf = self.device.new_buffer(
                size as u64,
                MTLResourceOptions::CPUCacheModeDefaultCache
                    | MTLResourceOptions::StorageModeShared,
            );
            self.buffers.insert(name.to_string(), buf);
        }
        self.buffers.get(name).unwrap()
    }

    pub fn buffer(&self, name: &str) -> Option<&Buffer> {
        self.buffers.get(name)
    }

    // ------------------------------------------------------------------
    // Textures
    // ------------------------------------------------------------------

    pub fn create_texture(&mut self, name: &str, desc: &TextureDescriptor) -> &Texture {
        let tex = self.device.new_texture(desc);
        self.textures.insert(name.to_string(), tex);
        self.textures.get(name).unwrap()
    }

    pub fn texture(&self, name: &str) -> Option<&Texture> {
        self.textures.get(name)
    }

    /// Convenience helper for a 2D RGBA8Unorm texture.
    pub fn create_texture_2d_rgba8(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        usage: metal::MTLTextureUsage,
    ) -> &Texture {
        let desc = TextureDescriptor::new();
        desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        desc.set_width(width as u64);
        desc.set_height(height as u64);
        desc.set_usage(usage);
        desc.set_storage_mode(metal::MTLStorageMode::Private);
        self.create_texture(name, &desc)
    }

    /// Convenience helper for a 2D depth texture array.
    pub fn create_depth_array_texture(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        array_length: u64,
    ) -> &Texture {
        let desc = TextureDescriptor::new();
        desc.set_texture_type(metal::MTLTextureType::D2Array);
        desc.set_pixel_format(MTLPixelFormat::Depth32Float);
        desc.set_width(width as u64);
        desc.set_height(height as u64);
        desc.set_array_length(array_length);
        desc.set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        desc.set_storage_mode(metal::MTLStorageMode::Private);
        self.create_texture(name, &desc)
    }

    // ------------------------------------------------------------------
    // Samplers
    // ------------------------------------------------------------------

    pub fn create_sampler(&mut self, name: &str, desc: &SamplerDescriptor) -> &SamplerState {
        let sampler = self.device.new_sampler(desc);
        self.samplers.insert(name.to_string(), sampler);
        self.samplers.get(name).unwrap()
    }

    pub fn sampler(&self, name: &str) -> Option<&SamplerState> {
        self.samplers.get(name)
    }

    /// Linear + clamp sampler (default for most textures).
    pub fn create_sampler_linear_clamp(&mut self, name: &str) -> &SamplerState {
        let desc = SamplerDescriptor::new();
        desc.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
        desc.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
        desc.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
        desc.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
        self.create_sampler(name, &desc)
    }

    /// Nearest + clamp sampler (crisp HUD / overlay).
    pub fn create_sampler_nearest_clamp(&mut self, name: &str) -> &SamplerState {
        let desc = SamplerDescriptor::new();
        desc.set_min_filter(metal::MTLSamplerMinMagFilter::Nearest);
        desc.set_mag_filter(metal::MTLSamplerMinMagFilter::Nearest);
        desc.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
        desc.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
        self.create_sampler(name, &desc)
    }

    /// Comparison sampler for PCF shadow map reads.
    pub fn create_sampler_shadow_pcf(&mut self, name: &str) -> &SamplerState {
        let desc = SamplerDescriptor::new();
        desc.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
        desc.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
        desc.set_compare_function(metal::MTLCompareFunction::LessEqual);
        desc.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
        desc.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
        self.create_sampler(name, &desc)
    }
}
