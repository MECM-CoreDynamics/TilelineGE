#![cfg(target_os = "linux")]

use std::ffi::CString;
use std::mem::{offset_of, size_of};

use ash::{vk, Device};

use crate::graphics::frame_snapshot::FrameInstanceTransform;
#[cfg(test)]
use crate::graphics::vulkan_backend::DrawPushConstants;
use crate::graphics::vulkan_backend::VulkanBackendError;
use crate::graphics::vulkan_geometry::SceneVertex;
use crate::graphics::vulkan_shaders;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScenePipelineMode {
    Opaque,
    Transparent,
    Overlay,
}

pub(crate) unsafe fn create_scene_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
    pipeline_layout: vk::PipelineLayout,
    mode: ScenePipelineMode,
    pipeline_cache: vk::PipelineCache,
) -> Result<vk::Pipeline, VulkanBackendError> {
    let shader_modules = vulkan_shaders::create_scene_shader_modules(device)?;

    let entry_name = CString::new("main")
        .map_err(|_| VulkanBackendError::InvalidConfig("invalid Vulkan shader entrypoint"))?;
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .module(shader_modules.vertex)
            .name(&entry_name)
            .stage(vk::ShaderStageFlags::VERTEX),
        vk::PipelineShaderStageCreateInfo::default()
            .module(shader_modules.fragment)
            .name(&entry_name)
            .stage(vk::ShaderStageFlags::FRAGMENT),
    ];

    let vertex_binding_descriptions = [
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<SceneVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        },
        vk::VertexInputBindingDescription {
            binding: 1,
            stride: size_of::<FrameInstanceTransform>() as u32,
            input_rate: vk::VertexInputRate::INSTANCE,
        },
    ];
    let instance_model_offset = offset_of!(FrameInstanceTransform, model) as u32;
    let instance_color_offset = offset_of!(FrameInstanceTransform, color_rgba) as u32;
    let instance_material_index_offset = offset_of!(FrameInstanceTransform, material_index) as u32;
    let instance_flags_offset = offset_of!(FrameInstanceTransform, flags) as u32;
    let vertex_attribute_descriptions = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(SceneVertex, position) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 1,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: instance_model_offset,
        },
        vk::VertexInputAttributeDescription {
            location: 2,
            binding: 1,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: instance_model_offset + 16,
        },
        vk::VertexInputAttributeDescription {
            location: 3,
            binding: 1,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: instance_model_offset + 32,
        },
        vk::VertexInputAttributeDescription {
            location: 4,
            binding: 1,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: instance_model_offset + 48,
        },
        vk::VertexInputAttributeDescription {
            location: 5,
            binding: 1,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: instance_color_offset,
        },
        vk::VertexInputAttributeDescription {
            location: 6,
            binding: 1,
            format: vk::Format::R32_UINT,
            offset: instance_material_index_offset,
        },
        vk::VertexInputAttributeDescription {
            location: 7,
            binding: 1,
            format: vk::Format::R32_UINT,
            offset: instance_flags_offset,
        },
        vk::VertexInputAttributeDescription {
            location: 8,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: offset_of!(SceneVertex, uv) as u32,
        },
    ];
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_binding_descriptions)
        .vertex_attribute_descriptions(&vertex_attribute_descriptions);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);
    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D::default().extent(extent)];
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(if mode != ScenePipelineMode::Opaque {
            vk::CullModeFlags::NONE
        } else {
            vk::CullModeFlags::BACK
        })
        // The Vulkan path applies an explicit clip-space Y flip, which inverts winding.
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = match mode {
        ScenePipelineMode::Opaque => vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false),
        ScenePipelineMode::Transparent => vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false),
        ScenePipelineMode::Overlay => vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::ALWAYS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false),
    };
    let color_blend_attachment = if mode != ScenePipelineMode::Opaque {
        [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)]
    } else {
        [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)]
    };
    let color_blending =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachment);
    let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blending)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)];
    let pipeline = device
        .create_graphics_pipelines(pipeline_cache, &pipeline_info, None)
        .map_err(|(_, err)| VulkanBackendError::Vk(err))?[0];

    vulkan_shaders::destroy_scene_shader_modules(device, shader_modules);
    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_modes_have_distinct_depth_semantics() {
        assert!(ScenePipelineMode::Opaque != ScenePipelineMode::Transparent);
        assert!(ScenePipelineMode::Transparent != ScenePipelineMode::Overlay);
    }

    #[test]
    fn push_constant_block_size_matches_shader_contract() {
        assert_eq!(size_of::<DrawPushConstants>(), 16);
    }
}
