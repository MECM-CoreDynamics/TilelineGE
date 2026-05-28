enable wgpu_ray_query;

struct Camera {
    view_proj: mat4x4<f32>,
    camera_eye: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u_camera: Camera;

struct LightData {
    position_kind: vec4<f32>,
    direction_inner: vec4<f32>,
    color_intensity: vec4<f32>,
    params: vec4<f32>,
    shadow: vec4<f32>,
};

struct Lighting {
    light_count: u32,
    rt_mode: u32,
    rt_active: u32,
    rt_dynamic_count: u32,
    rt_dynamic_cap: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(1)
var<storage, read> u_lights: array<LightData, 32>;

@group(0) @binding(2)
var<uniform> u_lighting: Lighting;

struct ShadowUniform {
    light_view_proj: array<mat4x4<f32>, 4>,
    shadow_light_indices: vec4<i32>,
    shadow_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(3)
var shadow_map: texture_depth_2d_array;
@group(0) @binding(4)
var shadow_smp: sampler_comparison;
@group(0) @binding(5)
var<uniform> u_shadow: ShadowUniform;

@group(1) @binding(0)
var scene_tlas: acceleration_structure;

struct VSIn {
    @location(0) position: vec3<f32>,
    @location(1) model_col0: vec4<f32>,
    @location(2) model_col1: vec4<f32>,
    @location(3) model_col2: vec4<f32>,
    @location(4) model_col3: vec4<f32>,
    @location(5) base_color: vec4<f32>,
    @location(6) material_params: vec4<f32>,
    @location(7) emissive: vec4<f32>,
};

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) local_pos: vec3<f32>,
    @location(4) shading_code: f32,
    @location(5) roughness: f32,
    @location(6) metallic: f32,
};

struct FSOut {
    @location(0) accum: vec4<f32>,
    @location(1) reveal: f32,
};

@vertex
fn vs_main(input: VSIn) -> VSOut {
    let model = mat4x4<f32>(
        input.model_col0,
        input.model_col1,
        input.model_col2,
        input.model_col3
    );
    let world_pos = model * vec4<f32>(input.position, 1.0);

    var out: VSOut;
    out.position = u_camera.view_proj * world_pos;
    out.color = input.base_color;
    out.emissive = input.emissive.xyz;
    out.world_pos = world_pos.xyz;
    out.local_pos = input.position;
    out.shading_code = input.emissive.w;
    out.roughness = input.material_params.x;
    out.metallic = input.material_params.y;
    return out;
}

fn sample_shadow_rt(world_pos: vec3<f32>, light_pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    let to_light = light_pos - world_pos;
    let light_dist = length(to_light);
    if light_dist < 0.001 {
        return 1.0;
    }
    let light_dir = to_light / light_dist;
    let origin = world_pos + normal * 0.008 + light_dir * 0.001;

    var rq: ray_query;
    rayQueryInitialize(&rq, scene_tlas, RayDesc(4u, 0xFFu, 0.0, light_dist - 0.02, origin, light_dir));
    rayQueryProceed(&rq);
    let intersection = rayQueryGetCommittedIntersection(&rq);
    return select(0.0, 1.0, intersection.kind == RAY_QUERY_INTERSECTION_NONE);
}

fn evaluate_light(
    light: LightData,
    normal: vec3<f32>,
    world_pos: vec3<f32>,
    base_color: vec3<f32>,
) -> vec3<f32> {
    let to_light = light.position_kind.xyz - world_pos;
    let distance = max(length(to_light), 1e-4);
    let light_dir = to_light / distance;
    let range = max(light.params.x, 1e-3);
    if distance > range {
        return vec3<f32>(0.0);
    }

    var attenuation = 1.0 - smoothstep(range * 0.65, range, distance);
    attenuation *= attenuation;
    attenuation *= 1.0 / (1.0 + distance * 0.10);

    if light.position_kind.w > 0.5 {
        let spot_axis = normalize(-light.direction_inner.xyz);
        let cone = dot(spot_axis, light_dir);
        let inner = light.direction_inner.w;
        let outer = light.params.y;
        attenuation *= smoothstep(outer, inner, cone);
    }

    let ndotl = max(dot(normal, light_dir), 0.0);
    if ndotl <= 0.0 || attenuation <= 1e-5 {
        return vec3<f32>(0.0);
    }

    let view_dir = normalize(u_camera.camera_eye.xyz - world_pos);
    let half_dir = normalize(light_dir + view_dir);
    let spec_power = 24.0 + light.params.w * 24.0;
    let specular = pow(max(dot(normal, half_dir), 0.0), spec_power) * light.params.w;

    let diffuse = base_color * ndotl;
    let spec = vec3<f32>(specular);
    return (diffuse + spec) * light.color_intensity.rgb * light.color_intensity.w * attenuation;
}

@fragment
fn fs_main(input: VSOut, @builtin(front_facing) is_front: bool) -> FSOut {
    let dpx = dpdx(input.world_pos);
    let dpy = dpdy(input.world_pos);
    var normal = normalize(cross(dpy, dpx));
    if !is_front {
        normal = -normal;
    }

    let is_unlit = input.shading_code > 0.5;
    var lit = input.color.rgb * 0.30;
    if is_unlit {
        lit = max(input.color.rgb * 0.68 + input.emissive, input.color.rgb * 0.82);
    } else {
        let light_count = min(u_lighting.light_count, 32u);
        if light_count == 0u {
            let fallback_dir = normalize(vec3<f32>(0.42, 0.74, 0.52));
            let fallback = max(dot(normal, fallback_dir), 0.0) * 0.68 + 0.32;
            lit += input.color.rgb * fallback;
        } else {
            for (var i: u32 = 0u; i < light_count; i = i + 1u) {
                lit += evaluate_light(u_lights[i], normal, input.world_pos, input.color.rgb);
            }
        }
        let glass_floor = input.color.rgb * 0.30;
        lit = max(lit, glass_floor);
        lit += input.emissive;
    }

    let view_dir = normalize(u_camera.camera_eye.xyz - input.world_pos);
    let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 3.0);
    let edge = max(max(abs(input.local_pos.x), abs(input.local_pos.y)), abs(input.local_pos.z));
    let edge_boost = smoothstep(0.42, 0.50, edge);
    lit += vec3<f32>(0.24, 0.30, 0.38) * rim;
    lit += vec3<f32>(0.11, 0.16, 0.24) * edge_boost;

    var alpha = 0.0;
    if is_unlit {
        alpha = clamp(max(input.color.a + 0.02, 0.24) + rim * 0.06 + edge_boost * 0.04, 0.0, 0.52);
    } else {
        alpha = clamp(max(input.color.a + 0.04, 0.34) + rim * 0.06 + edge_boost * 0.04, 0.0, 0.60);
    }
    let premultiplied = lit * alpha;

    var out: FSOut;
    out.accum = vec4<f32>(premultiplied, alpha);
    out.reveal = alpha;
    return out;
}
