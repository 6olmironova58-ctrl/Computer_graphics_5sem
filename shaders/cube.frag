#version 450

layout(location = 0) in vec3 world_position;
layout(location = 1) in vec3 world_normal;

layout(location = 0) out vec4 out_color;

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

struct PointLight {
    vec3 position;
    float pad0;
    vec3 color;
    float pad1;
};

struct SpotLight {
    vec3 position;
    float inner_cos;
    vec3 direction;
    float outer_cos;
    vec3 color;
    float pad0;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

vec3 blinnPhong(vec3 n, vec3 v, vec3 l, vec3 light_color) {
    float n_dot_l = max(dot(n, l), 0.0);
    if (n_dot_l <= 0.0) {
        return vec3(0.0);
    }
    vec3 h = normalize(v + l);
    float n_dot_h = max(dot(n, h), 0.0);
    vec3 diffuse = model_uniforms.albedo * n_dot_l;
    vec3 specular = model_uniforms.specular * pow(n_dot_h, model_uniforms.alpha);
    return (diffuse + specular) * light_color;
}

void main() {
    vec3 n = normalize(world_normal);
    vec3 v = normalize(scene_uniforms.camera_position - world_position);

    vec3 color = scene_uniforms.ambient_light * model_uniforms.albedo;

    vec3 directional_l = -normalize(scene_uniforms.directional_light_direction);
    color += blinnPhong(n, v, directional_l, scene_uniforms.directional_light_color);

    for (uint i = 0; i < scene_uniforms.point_light_count; ++i) {
        vec3 to_light = point_lights[i].position - world_position;
        float dist2 = max(dot(to_light, to_light), 1e-4);
        vec3 l = to_light * inversesqrt(dist2);
        color += blinnPhong(n, v, l, point_lights[i].color) / dist2;
    }

    for (uint i = 0; i < scene_uniforms.spot_light_count; ++i) {
        vec3 to_light = spot_lights[i].position - world_position;
        float dist2 = max(dot(to_light, to_light), 1e-4);
        vec3 l = to_light * inversesqrt(dist2);
        float cos_angle = dot(-l, normalize(spot_lights[i].direction));
        float edge = max(spot_lights[i].inner_cos - spot_lights[i].outer_cos, 1e-4);
        float cone = clamp((cos_angle - spot_lights[i].outer_cos) / edge, 0.0, 1.0);
        color += blinnPhong(n, v, l, spot_lights[i].color) * cone / dist2;
    }

    out_color = vec4(color, 1.0);
}
