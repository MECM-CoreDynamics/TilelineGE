#version 450
layout(location = 0) in vec4 v_color;
layout(location = 1) in vec3 v_world_position;
layout(location = 2) in vec3 v_world_normal;
layout(location = 3) flat in uint v_material_index;
layout(location = 4) flat in uint v_flags;
layout(location = 5) in vec3 v_local_position;

layout(location = 0) out vec4 out_color;

struct MaterialRecord {
    vec4 material_params;
    vec3 emissive_rgb;
    uint shading_code;
    uint texture_index;
    uint flags;
    uint _padding0;
    uint _padding1;
};

struct LightRecord {
    vec4 position_kind;
    vec4 direction_inner;
    vec4 color_intensity;
    vec4 params;
    vec4 shadow;
};

struct TextureRecord {
    uint texture_slot;
    uint sampler_code;
    uint flags;
    uint _padding0;
};

layout(set = 0, binding = 1, std430) readonly buffer MaterialBuffer {
    MaterialRecord materials[];
} u_materials;

layout(set = 0, binding = 2, std430) readonly buffer LightBuffer {
    LightRecord lights[];
} u_lights;

layout(set = 0, binding = 3, std430) readonly buffer TextureBuffer {
    TextureRecord textures[];
} u_textures;

layout(set = 0, binding = 4) uniform sampler2DArray u_texture_array;

layout(push_constant) uniform DrawPushConstants {
    uint material_count;
    uint light_count;
    uint texture_count;
    uint _padding0;
} u_draw;

const uint MATERIAL_FLAG_UNLIT = 1u << 0;
const uint INSTANCE_FLAG_TRANSPARENT = 1u << 0;
const uint INSTANCE_FLAG_MESH = 1u << 1;
const uint INSTANCE_FLAG_BOX = 1u << 2;

vec3 accumulate_light(vec3 world_position, vec3 world_normal, LightRecord light) {
    vec3 to_light = light.position_kind.xyz - world_position;
    float distance_to_light = length(to_light);
    if (distance_to_light <= 0.0001) {
        return vec3(0.0);
    }

    float light_range = max(light.params.x, 0.001);
    float attenuation = clamp(1.0 - (distance_to_light / light_range), 0.0, 1.0);
    if (attenuation <= 0.0) {
        return vec3(0.0);
    }

    vec3 light_dir = normalize(to_light);
    if (light.position_kind.w > 0.5) {
        vec3 spot_axis = normalize(-light.direction_inner.xyz);
        float spot_cos = dot(light_dir, spot_axis);
        float inner_cos = light.direction_inner.w;
        float outer_cos = light.params.y;
        float softness = clamp(light.params.z, 0.0001, 1.0);
        float cone_t = smoothstep(outer_cos, max(inner_cos, outer_cos + softness * 0.02), spot_cos);
        attenuation *= cone_t;
    }

    float diffuse = max(dot(normalize(world_normal), light_dir), 0.12);
    return light.color_intensity.rgb * light.color_intensity.w * attenuation * diffuse;
}

vec2 scene_texture_uv(vec3 world_position) {
    return fract(abs(world_position.xz) * 0.17 + vec2(0.125, 0.375));
}

void main() {
    vec3 base_color = v_color.rgb;
    float alpha = v_color.a;
    vec3 normal = normalize(v_world_normal);
    vec3 lit_color = base_color * 0.04;

    if (u_draw.material_count > 0u) {
        uint material_index = min(v_material_index, u_draw.material_count - 1u);
        MaterialRecord material = u_materials.materials[material_index];
        if (u_draw.texture_count > 0u) {
            uint texture_index = min(material.texture_index, u_draw.texture_count - 1u);
            TextureRecord texture_record = u_textures.textures[texture_index];
            float layer = float(texture_record.texture_slot);
            vec3 sampled_rgb = texture(u_texture_array, vec3(scene_texture_uv(v_world_position), layer)).rgb;
            base_color *= mix(vec3(1.0), sampled_rgb, 0.35);
        }
        float emissive_strength = material.material_params.z;
        lit_color += material.emissive_rgb * emissive_strength;
        if ((material.flags & MATERIAL_FLAG_UNLIT) != 0u || (v_flags & MATERIAL_FLAG_UNLIT) != 0u) {
            out_color = vec4(base_color + material.emissive_rgb * emissive_strength, alpha);
            return;
        }
    }

    uint light_count = min(u_draw.light_count, 32u);
    if (light_count == 0u) {
        // Fallback "fake sun" so scenes with no configured lights remain visible.
        // Mirrors the wgpu scene_3d.wgsl behaviour: 24% floor + directional contribution.
        vec3 fallback_dir = normalize(vec3(0.42, 0.74, 0.52));
        float fallback = max(dot(normal, fallback_dir), 0.0) * 0.76 + 0.24;
        lit_color += base_color * fallback;
    } else {
        for (uint i = 0u; i < light_count; i++) {
            lit_color += base_color * accumulate_light(v_world_position, normal, u_lights.lights[i]);
        }
    }

    // Box primitives (e.g. bounce-tank container) get an edge highlight so the
    // silhouette stays clear from any view distance.
    if ((v_flags & INSTANCE_FLAG_BOX) != 0u) {
        float edge = max(max(abs(v_local_position.x), abs(v_local_position.y)), abs(v_local_position.z));
        float edge_boost = smoothstep(0.38, 0.50, edge);
        lit_color += vec3(0.08, 0.12, 0.18) * edge_boost;
        alpha = clamp(alpha + edge_boost * 0.32, 0.0, 1.0);
    }

    out_color = vec4(lit_color, alpha);
}
