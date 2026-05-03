//! Persistent per-primitive mesh slot (vertex + index buffer pair).

#![cfg(target_os = "macos")]

use metal::{Buffer, MTLIndexType, MTLResourceOptions};

/// A single mesh uploaded to the GPU.
///
/// Primitive code mapping (matches WGPU renderer):
/// - `0` → built-in sphere
/// - `1` → built-in box
/// - `2+` → custom mesh slot = `code - 2`
#[derive(Debug)]
pub struct MeshSlot {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
    pub index_type: MTLIndexType,
    pub vertex_count: u32,
}

impl MeshSlot {
    /// Upload interleaved vertex data and index data into new shared buffers.
    pub fn new<V, I>(
        device: &metal::Device,
        vertices: &[V],
        indices: &[I],
        index_type: MTLIndexType,
    ) -> Self
    where
        V: Copy,
        I: Copy,
    {
        let vert_bytes = std::mem::size_of_val(vertices);
        let idx_bytes = std::mem::size_of_val(indices);

        let vb = device.new_buffer(
            vert_bytes as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        );
        let ib = device.new_buffer(
            idx_bytes as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        );

        unsafe {
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                vb.contents() as *mut u8,
                vert_bytes,
            );
            std::ptr::copy_nonoverlapping(
                indices.as_ptr() as *const u8,
                ib.contents() as *mut u8,
                idx_bytes,
            );
        }

        Self {
            vertex_buffer: vb,
            index_buffer: ib,
            index_count: indices.len() as u32,
            index_type,
            vertex_count: vertices.len() as u32,
        }
    }
}
