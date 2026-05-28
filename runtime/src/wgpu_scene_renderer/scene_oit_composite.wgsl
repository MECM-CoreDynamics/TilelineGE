struct VSOut {
    @builtin(position) position: vec4<f32>,
};

@group(0) @binding(0)
var opaque_tex: texture_2d<f32>;
@group(0) @binding(1)
var accum_tex: texture_2d<f32>;
@group(0) @binding(2)
var reveal_tex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VSOut {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );

    var out: VSOut;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = textureDimensions(opaque_tex);
    let coord = vec2<i32>(
        clamp(i32(frag_pos.x), 0, i32(dims.x) - 1),
        clamp(i32(frag_pos.y), 0, i32(dims.y) - 1),
    );

    let opaque = textureLoad(opaque_tex, coord, 0);
    let accum = textureLoad(accum_tex, coord, 0);
    let reveal = clamp(textureLoad(reveal_tex, coord, 0).x, 0.0, 1.0);
    let transparent_rgb = accum.rgb / max(accum.a, 1e-4);
    let transparent_alpha = clamp(1.0 - reveal, 0.0, 1.0);
    let final_rgb = opaque.rgb * reveal + transparent_rgb * transparent_alpha;
    return vec4<f32>(final_rgb, 1.0);
}
