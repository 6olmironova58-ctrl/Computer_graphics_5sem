#include "application.hpp"
#include "math.hpp"

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace application {

struct Vertex {
    float pos[3];
    float normal[3];
};

struct SceneUniforms {
    math::Mat4 viewProjection;
    float cameraPosition[3];
    uint32_t pointLightCount;
    float directionalLightDirection[3];
    uint32_t spotLightCount;
    float directionalLightColor[3];
    float pad0;
    float ambientLight[3];
    float pad1;
};

struct ModelUniform {
    math::Mat4 model;
    float albedo[3];
    float pad0;
    float specular[3];
    float alpha;
};

struct PointLightGpu {
    float position[3];
    float pad0;
    float color[3];
    float pad1;
};

struct SpotLightGpu {
    float position[3];
    float innerCos;
    float direction[3];
    float outerCos;
    float color[3];
    float pad0;
};

static_assert(sizeof(SceneUniforms) == 128, "SceneUniforms must match std140 layout");
static_assert(sizeof(ModelUniform) == 96, "ModelUniform must match std140 layout");
static_assert(sizeof(PointLightGpu) == 32, "PointLightGpu must match std430 layout");
static_assert(sizeof(SpotLightGpu) == 48, "SpotLightGpu must match std430 layout");

struct PointLightState {
    float position[3] = { 0.0f, 2.0f, 0.0f };
    float color[3] = { 1.0f, 1.0f, 1.0f };
    float intensity = 5.0f;
};

struct SpotLightState {
    float position[3] = { 0.0f, 3.0f, 0.0f };
    float direction[3] = { 0.0f, -1.0f, 0.0f };
    float color[3] = { 1.0f, 1.0f, 1.0f };
    float intensity = 20.0f;
    float innerAngle = 15.0f;
    float outerAngle = 25.0f;
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    void* mapped = nullptr;
};

constexpr uint32_t kMaxPointLights = 64;
constexpr uint32_t kMaxSpotLights = 64;

namespace {
    std::vector<Vertex> cubeVertices;
    std::vector<uint16_t> cubeIndices;

    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer sceneBuffer;
    Buffer modelBuffer;
    Buffer pointLightBuffer;
    Buffer spotLightBuffer;

    VkDescriptorSetLayout sceneSetLayout;
    VkDescriptorSetLayout modelSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet sceneDescriptorSet;
    VkDescriptorSet modelDescriptorSet;

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkShaderModule vertShaderModule;
    VkShaderModule fragShaderModule;

    // 1.1
    int projectionMode = 1;

    // 1.2
    float cubePosition[3] = { 0.0f, 0.0f, 0.0f };
    float cubeRotation[3] = { 0.0f, 0.0f, 0.0f };
    float cubeScale[3]    = { 1.0f, 1.0f, 1.0f };

    // 1.3
    bool is_animation_paused = false;
    float anim_accumulated_time = 0.0f;
    float animation_speed = 1.0f;
    float trajectory_radius = 1.0f;
    int trajectory_type = 0;

    // 2.1
    float cameraPosition[3] = { 0.0f, 2.0f, 6.0f };
    float cameraRotation[3] = { -18.0f, 0.0f, 0.0f };
    float cameraFov = 45.0f;

    // 2.2
    float ambientColor[3] = { 1.0f, 1.0f, 1.0f };
    float ambientIntensity = 0.1f;
    float directionalDirection[3] = { -0.4f, -1.0f, -0.6f };
    float directionalColor[3] = { 1.0f, 1.0f, 1.0f };
    float directionalIntensity = 1.0f;

    // 2.3
    float materialAlbedo[3] = { 0.8f, 0.2f, 0.2f };
    float materialSpecular[3] = { 1.0f, 1.0f, 1.0f };
    float materialShininess = 32.0f;

    // 2.4
    std::vector<PointLightState> pointLights;

    // 2.5
    std::vector<SpotLightState> spotLights;
}

void generateProceduralCube(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
    const float normals[6][3] = {
        { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, -1.0f },
        { 1.0f, 0.0f, 0.0f }, { -1.0f, 0.0f, 0.0f },
        { 0.0f, 1.0f, 0.0f }, { 0.0f, -1.0f, 0.0f }
    };
    const float corners[4][2] = { { -1.0f, -1.0f }, { 1.0f, -1.0f }, { 1.0f, 1.0f }, { -1.0f, 1.0f } };

    for (int f = 0; f < 6; ++f) {
        const float* n = normals[f];
        bool vertical = std::fabs(n[1]) > 0.5f;
        const float ref[3] = { 0.0f, vertical ? 0.0f : 1.0f, vertical ? 1.0f : 0.0f };
        const float u[3] = {
            ref[1] * n[2] - ref[2] * n[1],
            ref[2] * n[0] - ref[0] * n[2],
            ref[0] * n[1] - ref[1] * n[0]
        };
        const float w[3] = {
            n[1] * u[2] - n[2] * u[1],
            n[2] * u[0] - n[0] * u[2],
            n[0] * u[1] - n[1] * u[0]
        };

        uint16_t base = static_cast<uint16_t>(vertices.size());
        for (int c = 0; c < 4; ++c) {
            Vertex v;
            for (int k = 0; k < 3; ++k) {
                v.pos[k] = 0.5f * (n[k] + corners[c][0] * u[k] + corners[c][1] * w[k]);
                v.normal[k] = n[k];
            }
            vertices.push_back(v);
        }
        const uint16_t faceIndices[6] = { 0, 1, 2, 2, 3, 0 };
        for (uint16_t i : faceIndices) {
            indices.push_back(base + i);
        }
    }
}

void normalizeDirection(const float in[3], float out[3]) {
    float len = std::sqrt(in[0] * in[0] + in[1] * in[1] + in[2] * in[2]);
    if (len < 1e-6f) {
        out[0] = 0.0f; out[1] = -1.0f; out[2] = 0.0f;
        return;
    }
    for (int k = 0; k < 3; ++k) {
        out[k] = in[k] / len;
    }
}

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл шейдера: " + filename);
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

VkShaderModule createShaderModule(const std::vector<char>& code, VkDevice device) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Не удалось создать модуль шейдера!");
    }
    return shaderModule;
}

Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    auto& ctx = graphics::internal::context;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT;

    Buffer result;
    VmaAllocationInfo allocInfo{};
    vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocCreateInfo,
                    &result.buffer, &result.allocation, &allocInfo);
    result.mapped = allocInfo.pMappedData;
    return result;
}

void uploadBuffer(const Buffer& buffer, const void* data, size_t size) {
    auto& ctx = graphics::internal::context;
    std::memcpy(buffer.mapped, data, size);
    vmaFlushAllocation(ctx.allocator, buffer.allocation, 0, size);
}

void destroyBuffer(Buffer& buffer) {
    auto& ctx = graphics::internal::context;
    vmaDestroyBuffer(ctx.allocator, buffer.buffer, buffer.allocation);
    buffer = {};
}

VkDescriptorSetLayoutBinding makeLayoutBinding(uint32_t binding, VkDescriptorType type, VkShaderStageFlags stages) {
    VkDescriptorSetLayoutBinding result{};
    result.binding = binding;
    result.descriptorType = type;
    result.descriptorCount = 1;
    result.stageFlags = stages;
    return result;
}

VkWriteDescriptorSet makeWrite(VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                               const VkDescriptorBufferInfo* info) {
    VkWriteDescriptorSet result{};
    result.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    result.dstSet = set;
    result.dstBinding = binding;
    result.descriptorCount = 1;
    result.descriptorType = type;
    result.pBufferInfo = info;
    return result;
}

bool initialize() {
    auto& ctx = graphics::internal::context;

    generateProceduralCube(cubeVertices, cubeIndices);

    auto vertCode = readFile("shaders/cube.vert.spv");
    auto fragCode = readFile("shaders/cube.frag.spv");
    vertShaderModule = createShaderModule(vertCode, ctx.device);
    fragShaderModule = createShaderModule(fragCode, ctx.device);

    size_t vertexBytes = sizeof(Vertex) * cubeVertices.size();
    vertexBuffer = createBuffer(vertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    uploadBuffer(vertexBuffer, cubeVertices.data(), vertexBytes);

    size_t indexBytes = sizeof(uint16_t) * cubeIndices.size();
    indexBuffer = createBuffer(indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    uploadBuffer(indexBuffer, cubeIndices.data(), indexBytes);

    sceneBuffer = createBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    modelBuffer = createBuffer(sizeof(ModelUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    pointLightBuffer = createBuffer(sizeof(PointLightGpu) * kMaxPointLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    spotLightBuffer = createBuffer(sizeof(SpotLightGpu) * kMaxSpotLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // Набор дескрипторов 0: данные сцены
    VkDescriptorSetLayoutBinding sceneBindings[3] = {
        makeLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT),
        makeLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT),
        makeLayoutBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT)
    };
    VkDescriptorSetLayoutCreateInfo sceneLayoutInfo{};
    sceneLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sceneLayoutInfo.bindingCount = 3;
    sceneLayoutInfo.pBindings = sceneBindings;
    vkCreateDescriptorSetLayout(ctx.device, &sceneLayoutInfo, nullptr, &sceneSetLayout);

    // Набор дескрипторов 1: данные модели
    VkDescriptorSetLayoutBinding modelBinding = makeLayoutBinding(
        0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    VkDescriptorSetLayoutCreateInfo modelLayoutInfo{};
    modelLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    modelLayoutInfo.bindingCount = 1;
    modelLayoutInfo.pBindings = &modelBinding;
    vkCreateDescriptorSetLayout(ctx.device, &modelLayoutInfo, nullptr, &modelSetLayout);

    VkDescriptorSetLayout setLayouts[2] = { sceneSetLayout, modelSetLayout };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = setLayouts;
    vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

    VkPipelineShaderStageCreateInfo vertStageInfo{};
    vertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStageInfo.module = vertShaderModule;
    vertStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragStageInfo{};
    fragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStageInfo.module = fragShaderModule;
    fragStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertStageInfo, fragStageInfo };

    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(Vertex);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrDesc[2]{};
    attrDesc[0].binding = 0; attrDesc[0].location = 0;
    attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrDesc[0].offset = offsetof(Vertex, pos);
    attrDesc[1].binding = 0; attrDesc[1].location = 1;
    attrDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrDesc[1].offset = offsetof(Vertex, normal);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = 2;
    vertexInputInfo.pVertexAttributeDescriptions = attrDesc;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.lineWidth = 1.0f;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = ctx.render_pass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        std::cerr << "[ERROR] Failed to create graphics pipeline!\n";
        return false;
    }

    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 2;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = 2;
    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool);

    VkDescriptorSet sets[2];
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 2;
    allocInfo.pSetLayouts = setLayouts;
    vkAllocateDescriptorSets(ctx.device, &allocInfo, sets);
    sceneDescriptorSet = sets[0];
    modelDescriptorSet = sets[1];

    VkDescriptorBufferInfo sceneInfo{ sceneBuffer.buffer, 0, sizeof(SceneUniforms) };
    VkDescriptorBufferInfo pointInfo{ pointLightBuffer.buffer, 0, sizeof(PointLightGpu) * kMaxPointLights };
    VkDescriptorBufferInfo spotInfo{ spotLightBuffer.buffer, 0, sizeof(SpotLightGpu) * kMaxSpotLights };
    VkDescriptorBufferInfo modelInfo{ modelBuffer.buffer, 0, sizeof(ModelUniform) };

    VkWriteDescriptorSet writes[4] = {
        makeWrite(sceneDescriptorSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &sceneInfo),
        makeWrite(sceneDescriptorSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &pointInfo),
        makeWrite(sceneDescriptorSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &spotInfo),
        makeWrite(modelDescriptorSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &modelInfo)
    };
    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, nullptr);

    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);

    vkDestroyPipeline(ctx.device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
    vkDestroyShaderModule(ctx.device, fragShaderModule, nullptr);
    vkDestroyShaderModule(ctx.device, vertShaderModule, nullptr);

    destroyBuffer(spotLightBuffer);
    destroyBuffer(pointLightBuffer);
    destroyBuffer(modelBuffer);
    destroyBuffer(sceneBuffer);
    destroyBuffer(indexBuffer);
    destroyBuffer(vertexBuffer);

    vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, modelSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, sceneSetLayout, nullptr);
}

void update([[maybe_unused]] double time) {
    // ИНТЕРФЕЙС 1.1
    ImGui::Begin("Projection Settings");
    ImGui::Text("Choose Projection Type:");
    ImGui::RadioButton("Perspective", &projectionMode, 1);
    ImGui::RadioButton("Orthographic", &projectionMode, 0);
    ImGui::End();

    // ИНТЕРФЕЙС 1.2
    ImGui::Begin("Transform Settings");
    ImGui::SliderFloat3("Position", cubePosition, -5.0f, 5.0f);
    ImGui::SliderFloat3("Rotation", cubeRotation, -3.14159f, 3.14159f);
    ImGui::SliderFloat3("Scale", cubeScale, 0.1f, 3.0f);
    ImGui::End();

    // ИНТЕРФЕЙС 1.3
    ImGui::Begin("Animation Settings");
    ImGui::Checkbox("Pause Animation", &is_animation_paused);
    ImGui::SliderFloat("Speed", &animation_speed, 0.0f, 3.0f);
    ImGui::SliderFloat("Radius", &trajectory_radius, 0.5f, 2.0f);

    const char* traj_types[] = { "Circle", "Figure-8" };
    ImGui::Combo("Trajectory Type", &trajectory_type, traj_types, IM_ARRAYSIZE(traj_types));
    ImGui::End();

    // ИНТЕРФЕЙС 2.1
    ImGui::Begin("Camera Settings");
    ImGui::DragFloat3("Position", cameraPosition, 0.05f);
    ImGui::SliderFloat("Pitch", &cameraRotation[0], -89.0f, 89.0f);
    ImGui::SliderFloat("Yaw", &cameraRotation[1], -180.0f, 180.0f);
    ImGui::SliderFloat("Roll", &cameraRotation[2], -180.0f, 180.0f);
    ImGui::SliderFloat("Field of View", &cameraFov, 10.0f, 120.0f);
    ImGui::End();

    // ИНТЕРФЕЙС 2.2
    ImGui::Begin("Scene Lighting");
    ImGui::Text("Ambient light");
    ImGui::ColorEdit3("Ambient Color", ambientColor);
    ImGui::SliderFloat("Ambient Intensity", &ambientIntensity, 0.0f, 1.0f);
    ImGui::Separator();
    ImGui::Text("Directional light");
    ImGui::DragFloat3("Direction", directionalDirection, 0.01f, -1.0f, 1.0f);
    ImGui::ColorEdit3("Color", directionalColor);
    ImGui::SliderFloat("Intensity", &directionalIntensity, 0.0f, 5.0f);
    ImGui::End();

    // ИНТЕРФЕЙС 2.3
    ImGui::Begin("Material Settings");
    ImGui::ColorEdit3("Albedo", materialAlbedo);
    ImGui::ColorEdit3("Specular Color", materialSpecular);
    ImGui::SliderFloat("Shininess", &materialShininess, 1.0f, 512.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
    ImGui::End();

    // ИНТЕРФЕЙС 2.4
    ImGui::Begin("Point Lights");
    ImGui::BeginDisabled(pointLights.size() >= kMaxPointLights);
    if (ImGui::Button("Add point light")) {
        pointLights.push_back(PointLightState{});
    }
    ImGui::EndDisabled();
    int removePoint = -1;
    for (int i = 0; i < (int)pointLights.size(); ++i) {
        PointLightState& light = pointLights[i];
        ImGui::PushID(i);
        std::string title = "Point light " + std::to_string(i);
        if (ImGui::CollapsingHeader(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Position", light.position, 0.05f);
            ImGui::ColorEdit3("Color", light.color);
            ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 100.0f);
            if (ImGui::Button("Remove")) {
                removePoint = i;
            }
        }
        ImGui::PopID();
    }
    if (removePoint >= 0) {
        pointLights.erase(pointLights.begin() + removePoint);
    }
    ImGui::End();

    // ИНТЕРФЕЙС 2.5
    ImGui::Begin("Spot Lights");
    ImGui::BeginDisabled(spotLights.size() >= kMaxSpotLights);
    if (ImGui::Button("Add spot light")) {
        spotLights.push_back(SpotLightState{});
    }
    ImGui::EndDisabled();
    int removeSpot = -1;
    for (int i = 0; i < (int)spotLights.size(); ++i) {
        SpotLightState& light = spotLights[i];
        ImGui::PushID(i);
        std::string title = "Spot light " + std::to_string(i);
        if (ImGui::CollapsingHeader(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Position", light.position, 0.05f);
            ImGui::DragFloat3("Direction", light.direction, 0.01f, -1.0f, 1.0f);
            ImGui::ColorEdit3("Color", light.color);
            ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 200.0f);
            ImGui::SliderFloat("Outer Angle", &light.outerAngle, 1.0f, 89.0f);
            light.innerAngle = std::min(light.innerAngle, light.outerAngle);
            ImGui::SliderFloat("Inner Angle", &light.innerAngle, 0.0f, light.outerAngle);
            if (ImGui::Button("Remove")) {
                removeSpot = i;
            }
        }
        ImGui::PopID();
    }
    if (removeSpot >= 0) {
        spotLights.erase(spotLights.begin() + removeSpot);
    }
    ImGui::End();

    auto& ctx = graphics::internal::context;

    if (!is_animation_paused) {
        anim_accumulated_time += (float)ImGui::GetIO().DeltaTime * animation_speed;
    }
    float t = anim_accumulated_time;

    float anim_x = 0.0f, anim_z = 0.0f;
    if (trajectory_type == 0) {
        anim_x = trajectory_radius * std::cos(t);
        anim_z = trajectory_radius * std::sin(t);
    } else {
        anim_x = trajectory_radius * std::sin(t);
        anim_z = trajectory_radius * std::sin(t) * std::cos(t);
    }

    math::Mat4 scaleMat = math::mat4Scale(cubeScale[0], cubeScale[1], cubeScale[2]);
    math::Mat4 rotX = math::mat4RotateX(cubeRotation[0]);
    math::Mat4 rotY = math::mat4RotateY(cubeRotation[1]);
    math::Mat4 rotZ = math::mat4RotateZ(cubeRotation[2]);
    math::Mat4 rotationMat = math::mat4Multiply(rotZ, math::mat4Multiply(rotY, rotX));
    math::Mat4 manualTranslate = math::mat4Translate(cubePosition[0], cubePosition[1], cubePosition[2]);
    math::Mat4 animTranslate = math::mat4Translate(anim_x, 0.0f, anim_z);

    math::Mat4 model = math::mat4Multiply(rotationMat, scaleMat);
    model = math::mat4Multiply(animTranslate, model);
    model = math::mat4Multiply(manualTranslate, model);

    math::Mat4 view = math::mat4View(cameraPosition,
                                     math::radians(cameraRotation[0]),
                                     math::radians(cameraRotation[1]),
                                     math::radians(cameraRotation[2]));

    float aspect = (float)ctx.swapchain_extent.width / (float)ctx.swapchain_extent.height;
    math::Mat4 proj;
    if (projectionMode == 1) {
        proj = math::mat4Perspective(math::radians(cameraFov), aspect, 0.1f, 100.0f);
    } else {
        float size = 2.0f;
        proj = math::mat4Orthographic(-size * aspect, size * aspect, -size, size, 0.1f, 100.0f);
    }
    for (int j = 0; j < 4; ++j) {
        proj[1][j] *= -1.0f;
    }

    SceneUniforms scene{};
    scene.viewProjection = math::mat4Multiply(proj, view);
    std::memcpy(scene.cameraPosition, cameraPosition, sizeof(cameraPosition));
    normalizeDirection(directionalDirection, scene.directionalLightDirection);
    for (int k = 0; k < 3; ++k) {
        scene.directionalLightColor[k] = directionalColor[k] * directionalIntensity;
        scene.ambientLight[k] = ambientColor[k] * ambientIntensity;
    }
    scene.pointLightCount = (uint32_t)pointLights.size();
    scene.spotLightCount = (uint32_t)spotLights.size();
    uploadBuffer(sceneBuffer, &scene, sizeof(scene));

    ModelUniform modelUniform{};
    modelUniform.model = model;
    std::memcpy(modelUniform.albedo, materialAlbedo, sizeof(materialAlbedo));
    std::memcpy(modelUniform.specular, materialSpecular, sizeof(materialSpecular));
    modelUniform.alpha = materialShininess;
    uploadBuffer(modelBuffer, &modelUniform, sizeof(modelUniform));

    if (!pointLights.empty()) {
        std::vector<PointLightGpu> gpu(pointLights.size());
        for (size_t i = 0; i < pointLights.size(); ++i) {
            const PointLightState& src = pointLights[i];
            std::memcpy(gpu[i].position, src.position, sizeof(src.position));
            for (int k = 0; k < 3; ++k) {
                gpu[i].color[k] = src.color[k] * src.intensity;
            }
        }
        uploadBuffer(pointLightBuffer, gpu.data(), gpu.size() * sizeof(PointLightGpu));
    }

    if (!spotLights.empty()) {
        std::vector<SpotLightGpu> gpu(spotLights.size());
        for (size_t i = 0; i < spotLights.size(); ++i) {
            const SpotLightState& src = spotLights[i];
            std::memcpy(gpu[i].position, src.position, sizeof(src.position));
            normalizeDirection(src.direction, gpu[i].direction);
            for (int k = 0; k < 3; ++k) {
                gpu[i].color[k] = src.color[k] * src.intensity;
            }
            gpu[i].innerCos = std::cos(math::radians(src.innerAngle));
            gpu[i].outerCos = std::cos(math::radians(src.outerAngle));
        }
        uploadBuffer(spotLightBuffer, gpu.data(), gpu.size() * sizeof(SpotLightGpu));
    }
}

void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;

    vkResetCommandBuffer(fd.command_buffer, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(fd.command_buffer, &beginInfo);

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = ctx.render_pass;
    renderPassInfo.framebuffer = fd.framebuffer;
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = ctx.swapchain_extent;

    VkClearValue clearValues[2];
    clearValues[0].color = {{ 0.1f, 0.1f, 0.1f, 1.0f }};
    clearValues[1].depthStencil = { 1.0f, 0 };
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(fd.command_buffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkViewport viewport{};
    viewport.width = (float)ctx.swapchain_extent.width;
    viewport.height = (float)ctx.swapchain_extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = ctx.swapchain_extent;
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    VkDescriptorSet descriptorSets[2] = { sceneDescriptorSet, modelDescriptorSet };
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                            0, 2, descriptorSets, 0, nullptr);

    VkBuffer vertexBuffers[] = { vertexBuffer.buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(fd.command_buffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);

    vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(cubeIndices.size()), 1, 0, 0, 0);

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
