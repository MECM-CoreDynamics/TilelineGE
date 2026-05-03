//! Inline MSL shader sources for the Metal render graph.

#![cfg(target_os = "macos")]

pub const SCENE_3D_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ── Uniforms ─────────────────────────────────────────────────────────────────
struct Camera {
    float4x4 view_proj;
    float4   camera_eye;
};

struct LightData {
    float4 position_kind;
    float4 direction_inner;
    float4 color_intensity;
    float4 params;
    float4 shadow;
};

struct Lighting {
    uint light_count;
    uint rt_mode;
    uint rt_active;
    uint rt_dynamic_count;
    uint rt_dynamic_cap;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct ShadowUniform {
    float4x4 light_view_proj[4];
    int4     shadow_light_indices;
    uint     shadow_count;
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
};

// ── Vertex Input ─────────────────────────────────────────────────────────────
struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
};

struct InstanceIn {
    float4 model_col0 [[attribute(3)]];
    float4 model_col1 [[attribute(4)]];
    float4 model_col2 [[attribute(5)]];
    float4 model_col3 [[attribute(6)]];
    float4 base_color [[attribute(7)]];
    float4 material_params [[attribute(8)]];
    float4 emissive   [[attribute(9)]];
};

struct VSOut {
    float4 position      [[position]];
    float4 color;
    float3 emissive;
    float3 world_pos;
    float3 local_pos;
    float  primitive_code;
    float  roughness;
    float  metallic;
};

vertex VSOut scene_3d_vertex(
    VertexIn   vin  [[stage_in]],
    InstanceIn inst [[stage_in]],
    constant Camera &camera [[buffer(0)]]
) {
    float4x4 model = float4x4(inst.model_col0, inst.model_col1,
                              inst.model_col2, inst.model_col3);
    float4 world_pos = model * float4(vin.position, 1.0);

    VSOut out;
    out.position       = camera.view_proj * world_pos;
    out.color          = inst.base_color;
    out.emissive       = inst.emissive.xyz;
    out.world_pos      = world_pos.xyz;
    out.local_pos      = vin.position;
    out.primitive_code = inst.material_params.w;
    out.roughness      = inst.material_params.x;
    out.metallic       = inst.material_params.y;
    return out;
}

// ── Fragment ─────────────────────────────────────────────────────────────────
constant float3 ENV_SKY_COLOR    = float3(0.55, 0.75, 1.20);
constant float3 ENV_GROUND_COLOR = float3(0.30, 0.22, 0.14);

float3 schlick_fresnel(float3 f0, float cos_theta) {
    return f0 + (float3(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

float d_ggx(float ndoth, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom + 1e-7);
}

float3 env_sample(float3 r) {
    float t = r.y * 0.5 + 0.5;
    return mix(ENV_GROUND_COLOR, ENV_SKY_COLOR, t);
}

float sample_shadow(
    int slot,
    float3 world_pos,
    constant ShadowUniform &u_shadow,
    depth2d_array<float> shadow_map
) {
    float4 light_space = u_shadow.light_view_proj[slot] * float4(world_pos, 1.0);
    if (light_space.w <= 0.0) return 1.0;
    float3 proj = light_space.xyz / light_space.w;
    if (proj.z < -1.0 || proj.z > 1.0) return 1.0;

    float wgpu_z = proj.z * 0.5 + 0.5;
    float depth_test = wgpu_z - 0.0001;
    float2 uv = proj.xy * float2(0.5, -0.5) + float2(0.5, 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1.0;

    int map_size = 1024;
    float shadow_sum = 0.0;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int2 px = int2(
                clamp(int(uv.x * float(map_size)) + dx, 0, map_size - 1),
                clamp(int(uv.y * float(map_size)) + dy, 0, map_size - 1)
            );
            float stored = shadow_map.read(uint2(px), uint(slot));
            shadow_sum += select(0.0, 1.0, stored >= depth_test);
        }
    }
    return shadow_sum / 9.0;
}

float3 evaluate_light(
    LightData light,
    float3    normal,
    float3    world_pos,
    float3    base_color,
    float3    view_dir,
    float     roughness,
    float     metallic,
    constant ShadowUniform &u_shadow,
    depth2d_array<float>   shadow_map
) {
    float3 to_light  = light.position_kind.xyz - world_pos;
    float  distance  = max(length(to_light), 1e-4);
    float3 light_dir = to_light / distance;
    float  range     = max(light.params.x, 1e-3);
    if (distance > range) return float3(0.0);

    float attenuation = 1.0 - smoothstep(range * 0.65, range, distance);
    attenuation *= attenuation;
    attenuation *= 1.0 / (1.0 + distance * 0.10);

    if (light.position_kind.w > 0.5) {
        float3 spot_axis   = normalize(-light.direction_inner.xyz);
        float  cone        = dot(spot_axis, light_dir);
        float  inner       = light.direction_inner.w;
        float  outer       = light.params.y;
        float  cone_factor = smoothstep(outer, inner, cone);
        attenuation       *= cone_factor;
    }

    float ndotl = max(dot(normal, light_dir), 0.0);
    if (ndotl <= 0.0 || attenuation <= 1e-5) return float3(0.0);

    float3 f0       = mix(float3(0.04), base_color, metallic);
    float3 half_dir = normalize(light_dir + view_dir);
    float  ndoth    = max(dot(normal, half_dir), 0.0);
    float  ldoth    = max(dot(light_dir, half_dir), 0.0);
    float  D        = d_ggx(ndoth, max(roughness, 0.05));
    float3 F        = schlick_fresnel(f0, ldoth);
    float3 specular = D * F * 0.25 * light.params.w;

    float3 kd      = (float3(1.0) - F) * (1.0 - metallic);
    float3 diffuse = kd * base_color * ndotl;

    float shadow_term = 1.0;
    int   shadow_slot = int(round(light.shadow.y));
    if (shadow_slot >= 0 && uint(shadow_slot) < u_shadow.shadow_count) {
        shadow_term = sample_shadow(shadow_slot, world_pos, u_shadow, shadow_map);
    } else if (light.shadow.x > 0.5) {
        float penumbra_floor = mix(0.35, 0.80, 1.0 - light.params.z);
        shadow_term = mix(penumbra_floor, 1.0, ndotl);
    }

    return (diffuse + specular) * light.color_intensity.rgb
           * light.color_intensity.w * attenuation * shadow_term;
}

fragment float4 scene_3d_fragment(
    VSOut                   in            [[stage_in]],
    constant Camera        &camera        [[buffer(0)]],
    constant LightData     *u_lights      [[buffer(1)]],
    constant Lighting      &u_lighting    [[buffer(2)]],
    constant ShadowUniform &u_shadow       [[buffer(3)]],
    depth2d_array<float>   shadow_map     [[texture(0)]],
    bool                    is_front       [[front_facing]]
) {
    float3 dpx = dfdx(in.world_pos);
    float3 dpy = dfdy(in.world_pos);
    float3 normal = normalize(cross(dpy, dpx));
    if (!is_front) normal = -normal;

    float3 view_dir = normalize(camera.camera_eye.xyz - in.world_pos);
    float3 lit = in.color.rgb * 0.04;

    uint light_count = min(u_lighting.light_count, 32u);
    if (light_count == 0u) {
        float3 fallback_dir = normalize(float3(0.42, 0.74, 0.52));
        float  fallback = max(dot(normal, fallback_dir), 0.0) * 0.76 + 0.24;
        lit += in.color.rgb * fallback;
    } else {
        for (uint i = 0; i < light_count; ++i) {
            lit += evaluate_light(
                u_lights[i], normal, in.world_pos, in.color.rgb,
                view_dir, in.roughness, in.metallic,
                u_shadow, shadow_map
            );
        }
    }

    float3 f0_env = mix(float3(0.04), in.color.rgb, in.metallic);
    float3 F_env  = schlick_fresnel(f0_env, max(dot(normal, view_dir), 0.0));
    float3 refl   = reflect(-view_dir, normal);
    float  rough_sq = in.roughness * in.roughness;
    float3 env_spec = F_env * env_sample(refl) * (1.0 - rough_sq) * (1.0 - rough_sq);
    float3 env_diff = env_sample(normal) * in.color.rgb * (1.0 - in.metallic) * 0.12;
    lit += env_spec + env_diff;

    lit += in.emissive;
    float alpha = in.color.a;

    if (in.primitive_code > 0.5) {
        float edge = max(max(abs(in.local_pos.x), abs(in.local_pos.y)), abs(in.local_pos.z));
        float edge_boost = smoothstep(0.38, 0.50, edge);
        lit += float3(0.08, 0.12, 0.18) * edge_boost;
        alpha = clamp(alpha + edge_boost * 0.32, 0.0, 1.0);
    }
    return float4(lit, alpha);
}
"#;

pub const SCENE_SHADOW_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ShadowPassUniform {
    float4x4 light_view_proj;
};

struct VertexIn {
    float3 position [[attribute(0)]];
};

struct InstanceIn {
    float4 model_col0 [[attribute(3)]];
    float4 model_col1 [[attribute(4)]];
    float4 model_col2 [[attribute(5)]];
    float4 model_col3 [[attribute(6)]];
};

vertex float4 scene_shadow_vertex(
    VertexIn   vin  [[stage_in]],
    InstanceIn inst [[stage_in]],
    constant ShadowPassUniform &u_shadow [[buffer(0)]]
) {
    float4x4 model = float4x4(inst.model_col0, inst.model_col1,
                              inst.model_col2, inst.model_col3);
    float4 world_pos = model * float4(vin.position, 1.0);
    return u_shadow.light_view_proj * world_pos;
}
"#;
