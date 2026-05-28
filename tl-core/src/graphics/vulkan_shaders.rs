#![cfg(target_os = "linux")]

use ash::{vk, Device};

use crate::graphics::vulkan_backend::VulkanBackendError;

#[derive(Debug, Clone)]
struct SpirvShaderArtifact {
    words: Vec<u32>,
}

pub(crate) struct SceneShaderModules {
    pub(crate) vertex: vk::ShaderModule,
    pub(crate) fragment: vk::ShaderModule,
}

pub(crate) unsafe fn create_scene_shader_modules(
    device: &Device,
) -> Result<SceneShaderModules, VulkanBackendError> {
    let [vertex_artifact, fragment_artifact] = build_scene_shader_spirv_artifacts()?;
    let vertex = create_shader_module(device, &vertex_artifact.words)?;
    let fragment = match create_shader_module(device, &fragment_artifact.words) {
        Ok(module) => module,
        Err(err) => {
            device.destroy_shader_module(vertex, None);
            return Err(err);
        }
    };
    Ok(SceneShaderModules { vertex, fragment })
}

pub(crate) unsafe fn destroy_scene_shader_modules(device: &Device, modules: SceneShaderModules) {
    device.destroy_shader_module(modules.vertex, None);
    device.destroy_shader_module(modules.fragment, None);
}

fn build_scene_shader_spirv_artifacts() -> Result<[SpirvShaderArtifact; 2], VulkanBackendError> {
    Ok([
        SpirvShaderArtifact {
            words: spirv_bytes_to_words(include_bytes!("../../assets/shaders/spv/scene.vert.spv"))?,
        },
        SpirvShaderArtifact {
            words: spirv_bytes_to_words(include_bytes!("../../assets/shaders/spv/scene.frag.spv"))?,
        },
    ])
}

fn spirv_bytes_to_words(bytes: &[u8]) -> Result<Vec<u32>, VulkanBackendError> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(VulkanBackendError::InvalidConfig(
            "embedded SPIR-V artifact length is not aligned to 4 bytes",
        ));
    }
    Ok(chunks
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

unsafe fn create_shader_module(
    device: &Device,
    spirv: &[u32],
) -> Result<vk::ShaderModule, VulkanBackendError> {
    let info = vk::ShaderModuleCreateInfo::default().code(spirv);
    Ok(device.create_shader_module(&info, None)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_scene_shader_artifacts_are_non_empty() {
        let [vertex, fragment] = build_scene_shader_spirv_artifacts().expect("embedded SPIR-V");
        assert!(!vertex.words.is_empty());
        assert!(!fragment.words.is_empty());
    }

    #[test]
    fn spirv_bytes_to_words_reads_little_endian_words() {
        let words = spirv_bytes_to_words(&[0x03, 0x02, 0x23, 0x07, 0x11, 0x22, 0x33, 0x44])
            .expect("valid SPIR-V word bytes");
        assert_eq!(words, vec![0x0723_0203, 0x4433_2211]);
    }

    #[test]
    fn spirv_bytes_to_words_rejects_unaligned_bytes() {
        let err = spirv_bytes_to_words(&[0x03, 0x02, 0x23])
            .expect_err("unaligned bytes should be rejected");
        assert!(matches!(err, VulkanBackendError::InvalidConfig(_)));
    }
}
