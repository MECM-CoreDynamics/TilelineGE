#version 450
layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 view_proj;
} u_camera;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_model_col0;
layout(location = 2) in vec4 in_model_col1;
layout(location = 3) in vec4 in_model_col2;
layout(location = 4) in vec4 in_model_col3;
layout(location = 5) in vec4 in_color;
layout(location = 6) in uint in_material_index;
layout(location = 7) in uint in_flags;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec3 v_world_position;
layout(location = 2) out vec3 v_world_normal;
layout(location = 3) flat out uint v_material_index;
layout(location = 4) flat out uint v_flags;
layout(location = 5) out vec3 v_local_position;

void main() {
    mat4 model = mat4(in_model_col0, in_model_col1, in_model_col2, in_model_col3);
    vec4 world_position = model * vec4(in_position, 1.0);
    mat3 normal_matrix = mat3(model);
    vec3 local_normal = normalize(in_position);
    gl_Position = u_camera.view_proj * world_position;
    v_color = in_color;
    v_world_position = world_position.xyz;
    v_world_normal = normalize(normal_matrix * local_normal);
    v_material_index = in_material_index;
    v_flags = in_flags;
    v_local_position = in_position;
}
