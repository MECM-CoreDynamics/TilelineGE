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
    float4   atlas_scale_offset[4];
    int4     shadow_light_indices;
    uint     shadow_count;
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
};

// ── Vertex Input ─────────────────────────────────────────────────────────────
struct VSIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
    float4 model_col0 [[attribute(3)]];
    float4 model_col1 [[attribute(4)]];
    float4 model_col2 [[attribute(5)]];
    float4 model_col3 [[attribute(6)]];
    float4 base_color [[attribute(7)]];
    float4 material_params [[attribute(8)]];
    float4 emissive   [[attribute(9)]];
    uint   texture_slot [[attribute(10)]];
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
    VSIn in [[stage_in]],
    constant Camera &camera [[buffer(2)]]
) {
    float4x4 model = float4x4(in.model_col0, in.model_col1,
                              in.model_col2, in.model_col3);
    float4 world_pos = model * float4(in.position, 1.0);

    float4 pos = camera.view_proj * world_pos;
    pos.z = pos.z * 0.5 + pos.w * 0.5;

    VSOut out;
    out.position       = pos;
    out.color          = in.base_color;
    out.emissive       = in.emissive.xyz;
    out.world_pos      = world_pos.xyz;
    out.local_pos      = in.position;
    out.primitive_code = in.material_params.w;
    out.roughness      = in.material_params.x;
    out.metallic       = in.material_params.y;
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
    float3 light_pos,
    float light_range,
    constant ShadowUniform &u_shadow,
    depth2d<float>       shadow_map
) {
    float3 to_light = light_pos - world_pos;
    float light_dist = length(to_light);
    if (light_dist > light_range) return 1.0;

    float4 light_space = u_shadow.light_view_proj[slot] * float4(world_pos, 1.0);
    if (light_space.w <= 0.0) return 1.0;
    float3 proj = light_space.xyz / light_space.w;
    if (proj.z < -1.0 || proj.z > 1.0) return 1.0;

    float wgpu_z = proj.z * 0.5 + 0.5;
    float bias = 0.000001 + light_dist * 0.000000001;
    float depth_test = wgpu_z - bias;
    float2 uv = proj.xy * float2(0.5, -0.5) + float2(0.5, 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1.0;

    int map_size = 1024;
    float shadow_sum = 0.0;
    for (int dx = 0; dx <= 1; ++dx) {
        for (int dy = 0; dy <= 1; ++dy) {
            int2 px = int2(
                clamp(int(uv.x * float(map_size)) + dx, 0, map_size - 1),
                clamp(int(uv.y * float(map_size)) + dy, 0, map_size - 1)
            );
            float2 atlas_scale = u_shadow.atlas_scale_offset[slot].xy;
            float2 atlas_offset = u_shadow.atlas_scale_offset[slot].zw;
            float2 atlas_uv = uv * atlas_scale + atlas_offset;
            int2 atlas_px = int2(
                clamp(int(atlas_uv.x * float(map_size)), 0, map_size - 1),
                clamp(int(atlas_uv.y * float(map_size)), 0, map_size - 1)
            );
            float stored = shadow_map.read(uint2(atlas_px));
            shadow_sum += select(0.0, 1.0, stored >= depth_test);
        }
    }
    float shadow_val = shadow_sum / 4.0;
    float fade = 1.0 - smoothstep(light_range * 0.8, light_range, light_dist);
    return mix(1.0, shadow_val, fade);
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
    depth2d<float>         shadow_map
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
        shadow_term = sample_shadow(shadow_slot, world_pos, light.position_kind.xyz, light.params.x, u_shadow, shadow_map);
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
    depth2d<float>         shadow_map     [[texture(0)]],
    bool                    is_front       [[front_facing]]
) {
    float3 base_color = in.color.rgb;
    float  alpha      = in.color.a;

    float3 dpx = dfdx(in.world_pos);
    float3 dpy = dfdy(in.world_pos);
    float3 normal = normalize(cross(dpy, dpx));
    if (!is_front) normal = -normal;

    float3 view_dir = normalize(camera.camera_eye.xyz - in.world_pos);
    float3 lit = base_color * 0.04;

    uint light_count = min(u_lighting.light_count, 32u);
    if (light_count == 0u) {
        float3 fallback_dir = normalize(float3(0.42, 0.74, 0.52));
        float  fallback = max(dot(normal, fallback_dir), 0.0) * 0.76 + 0.24;
        lit += base_color * fallback;
    } else {
        for (uint i = 0; i < light_count; ++i) {
            lit += evaluate_light(
                u_lights[i], normal, in.world_pos, base_color,
                view_dir, in.roughness, in.metallic,
                u_shadow, shadow_map
            );
        }
    }

    float3 f0_env = mix(float3(0.04), base_color, in.metallic);
    float3 F_env  = schlick_fresnel(f0_env, max(dot(normal, view_dir), 0.0));
    float3 refl   = reflect(-view_dir, normal);
    float  rough_sq = in.roughness * in.roughness;
    float3 env_spec = F_env * env_sample(refl) * (1.0 - rough_sq) * (1.0 - rough_sq);
    float3 env_diff = env_sample(normal) * base_color * (1.0 - in.metallic) * 0.12;
    lit += env_spec + env_diff;

    lit += in.emissive;

    if (in.primitive_code > 0.5) {
        float edge = max(max(abs(in.local_pos.x), abs(in.local_pos.y)), abs(in.local_pos.z));
        float edge_factor = smoothstep(0.38, 0.50, edge);
        lit += float3(0.08, 0.12, 0.18) * edge_factor;
        alpha = clamp(alpha + edge_factor * 0.32, 0.0, 1.0);
    }
    return float4(lit, alpha);
}
"#;

pub const SCENE_SPRITE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

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

struct SpriteVertexIn {
    float2 local_pos      [[attribute(0)]];
    float4 translate_size [[attribute(1)]];
    float4 rot_z          [[attribute(2)]];
    float4 color          [[attribute(3)]];
    float4 atlas_rect     [[attribute(4)]];
    float4 kind_params    [[attribute(5)]];
};

struct SpriteVSOut {
    float4 position [[position]];
    float4 color    [[user(locn0)]];
    float2 uv       [[user(locn1)]];
    float2 atlas_uv [[user(locn2)]];
    float2 kind     [[user(locn3)]];
};

vertex SpriteVSOut sprite_vertex(SpriteVertexIn in [[stage_in]]) {
    float c = cos(in.rot_z.x);
    float s = sin(in.rot_z.x);
    float2 scaled = float2(in.local_pos.x * in.translate_size.z,
                           in.local_pos.y * in.translate_size.w);
    float2 rotated = float2(scaled.x * c - scaled.y * s,
                            scaled.x * s + scaled.y * c);
    float2 pos = rotated + in.translate_size.xy;
    float2 uv = in.local_pos + float2(0.5, 0.5);
    float2 atlas_uv = mix(in.atlas_rect.xy, in.atlas_rect.zw, uv);

    SpriteVSOut out;
    out.position = float4(pos, in.rot_z.y, 1.0);
    out.color    = in.color;
    out.uv       = uv;
    out.atlas_uv = atlas_uv;
    out.kind     = in.kind_params.xy;
    return out;
}

float3 terrain_style(float3 base_color, float2 uv, float2 atlas_uv, float slot) {
    float stripe = 0.5 + 0.5 * sin((atlas_uv.x * 64.0) + slot * 0.37);
    float3 top = float3(base_color.r * 1.06, base_color.g * 1.03, base_color.b * 0.90);
    float3 bottom = float3(base_color.r * 0.85, base_color.g * 0.92, base_color.b * 0.78);
    float3 grad = mix(bottom, top, uv.y);
    return mix(grad, grad * float3(0.72, 0.88, 0.72), stripe * 0.35);
}

float4 light_glow_style(float3 base_color, float2 uv) {
    float2 centered = uv - float2(0.5, 0.5);
    float dist = length(centered);
    float core = 1.0 - smoothstep(0.0, 0.08, dist);
    float halo = (1.0 - smoothstep(0.04, 0.38, dist)) * 0.55;
    float scatter = (1.0 - smoothstep(0.18, 0.50, dist)) * 0.18;
    float brightness = core + halo + scatter;
    float red_extra = (1.0 - smoothstep(0.06, 0.22, dist)) * 0.18;
    float3 color = float3(base_color.r * brightness + red_extra,
                          base_color.g * brightness,
                          base_color.b * brightness);
    return float4(color, brightness);
}

float3 camera_style(float3 base_color, float2 uv, float2 atlas_uv, float slot) {
    float2 centered = uv - float2(0.5, 0.5);
    float dist = length(centered);
    float ring = 1.0 - smoothstep(0.26, 0.43, abs(dist - 0.31));
    float lens = 1.0 - smoothstep(0.07, 0.31, dist);
    float scan = 0.5 + 0.5 * sin((atlas_uv.y * 48.0) + slot * 0.21);
    float3 ring_color = float3(0.95, 0.98, 1.0);
    float3 lens_color = mix(base_color * float3(0.36, 0.52, 0.78), base_color, scan * 0.6);
    return lens_color * (0.55 + lens * 0.45) + ring_color * ring * 0.55;
}

fragment float4 sprite_fragment(
    SpriteVSOut in          [[stage_in]],
    texture2d<float> sprite_tex [[texture(0)]],
    sampler sprite_smp      [[sampler(0)]],
    constant Lighting &u_lighting [[buffer(0)]]
) {
    int kind = int(in.kind.x + 0.5);
    float slot = in.kind.y;
    float4 sampled = sprite_tex.sample(sprite_smp, in.atlas_uv);
    float3 color = in.color.rgb * sampled.rgb;
    float alpha = in.color.a * sampled.a;

    if (kind == 4) {
        return light_glow_style(in.color.rgb, in.uv);
    }

    if (kind == 2) {
        color = camera_style(color, in.uv, in.atlas_uv, slot);
    } else if (kind == 3) {
        color = terrain_style(color, in.uv, in.atlas_uv, slot);
    } else if (kind == 1) {
        float pulse = 0.93 + 0.07 * sin(in.atlas_uv.x * 28.0 + slot * 0.15);
        color = color * pulse;
    } else {
        float grain = 0.98 + 0.02 * sin((in.atlas_uv.x + in.atlas_uv.y) * 32.0 + slot * 0.11);
        color = color * grain;
    }

    float2 centered = in.uv - float2(0.5, 0.5);
    float radial = 1.0 - smoothstep(0.22, 0.66, length(centered));
    float light_gain = 1.0 + min(float(u_lighting.light_count) / 16.0, 0.40);
    if (kind == 0 || kind == 1) {
        color += float3(0.06, 0.09, 0.14) * radial * light_gain;
    }

    return float4(color, alpha);
}
"#;

pub const SCENE_UPSCALE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct UpscaleUniform {
    float2 inv_source_size;
    float2 source_uv_scale;
    float sharpness;
    float _pad0;
    float _pad1;
    float _pad2;
};

struct UpscaleVSOut {
    float4 position [[position]];
    float2 uv       [[user(locn0)]];
};

vertex UpscaleVSOut upscale_vertex(uint vid [[vertex_id]]) {
    const float2 positions[3] = {
        float2(-1.0, -3.0),
        float2(-1.0,  1.0),
        float2( 3.0,  1.0)
    };
    float2 pos = positions[vid];
    UpscaleVSOut out;
    out.position = float4(pos, 0.0, 1.0);
    out.uv = pos * float2(0.5, -0.5) + float2(0.5, 0.5);
    return out;
}

fragment float4 upscale_fragment(
    UpscaleVSOut in          [[stage_in]],
    texture2d<float> src_tex [[texture(0)]],
    sampler src_smp          [[sampler(0)]],
    constant UpscaleUniform &u_upscale [[buffer(0)]]
) {
    float2 uv = clamp(in.uv * u_upscale.source_uv_scale, float2(0.0), float2(1.0));
    float4 center = src_tex.sample(src_smp, uv);

    float2 texel = u_upscale.inv_source_size;
    float4 s0 = src_tex.sample(src_smp, uv + float2( texel.x, 0.0));
    float4 s1 = src_tex.sample(src_smp, uv + float2(-texel.x, 0.0));
    float4 s2 = src_tex.sample(src_smp, uv + float2(0.0,  texel.y));
    float4 s3 = src_tex.sample(src_smp, uv + float2(0.0, -texel.y));
    float4 neighborhood = (s0 + s1 + s2 + s3) * 0.25;
    float3 sharpened = center.rgb + (center.rgb - neighborhood.rgb) * (u_upscale.sharpness * 1.65);
    return float4(max(sharpened, float3(0.0)), center.a);
}
"#;

pub const SCENE_SHADOW_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ShadowPassUniform {
    float4x4 light_view_proj;
};

struct ShadowVSIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
    float4 model_col0 [[attribute(3)]];
    float4 model_col1 [[attribute(4)]];
    float4 model_col2 [[attribute(5)]];
    float4 model_col3 [[attribute(6)]];
    float4 base_color [[attribute(7)]];
    float4 material_params [[attribute(8)]];
    float4 emissive   [[attribute(9)]];
};

vertex float4 scene_shadow_vertex(
    ShadowVSIn in [[stage_in]],
    constant ShadowPassUniform &u_shadow [[buffer(2)]]
) {
    float4x4 model = float4x4(in.model_col0, in.model_col1,
                              in.model_col2, in.model_col3);
    float4 world_pos = model * float4(in.position, 1.0);
    float4 pos = u_shadow.light_view_proj * world_pos;
    pos.z = pos.z * 0.5 + pos.w * 0.5;
    return pos;
}
"#;
