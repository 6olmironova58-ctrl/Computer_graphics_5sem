#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(std140, row_major, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    uint point_light_count;
    vec3 directional_light_direction;
    uint spot_light_count;
    vec3 directional_light_color;
    float pad0;
    vec3 ambient_light;
    float pad1;
} scene_uniforms;

layout(std140, row_major, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec3 albedo;
    float pad0;
    vec3 specular;
    float alpha;
} model_uniforms;

layout(location = 0) out vec3 world_position;
layout(location = 1) out vec3 world_normal;

void main() {
    vec4 world = model_uniforms.model * vec4(in_position, 1.0);
    world_position = world.xyz;
    world_normal = transpose(inverse(mat3(model_uniforms.model))) * in_normal;
    gl_Position = scene_uniforms.view_projection * world;
}
