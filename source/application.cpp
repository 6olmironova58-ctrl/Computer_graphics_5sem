#include "application.hpp"
#include "math.hpp"

#include <imgui.h>
#include <vector>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <GLFW/glfw3.h>

namespace application {

struct Vertex {
    float pos[3];   // x, y, z
    float color[3]; // r, g, b
};

std::vector<Vertex> generateProceduralCubeVertices() {
    std::vector<Vertex> vertices;
    
    float coords[8][3] = {
        {-0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f, -0.5f},
        { 0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f},
        {-0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f,  0.5f},
        { 0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f}
    };

    for (int i = 0; i < 8; ++i) {
        Vertex v;
        v.pos[0] = coords[i][0];
        v.pos[1] = coords[i][1];
        v.pos[2] = coords[i][2];

        v.color[0] = v.pos[0] + 0.5f; 
        v.color[1] = v.pos[1] + 0.5f; 
        v.color[2] = v.pos[2] + 0.5f; 

        vertices.push_back(v);
    }
    return vertices;
}

const std::vector<Vertex> cubeVertices = generateProceduralCubeVertices();

const std::vector<uint16_t> cubeIndices = {
    // Задняя грань
    0, 1, 2,  2, 3, 0,
    // Передняя грань
    4, 6, 5,  6, 4, 7,
    // Левая грань
    0, 3, 7,  7, 4, 0,
    // Правая грань
    1, 5, 6,  6, 2, 1,
    // Верхняя грань
    3, 2, 6,  6, 7, 3,
    // Нижняя грань
    0, 4, 5,  5, 1, 0
};

struct UniformBufferObject {
    math::Mat4 model;
    math::Mat4 view;
    math::Mat4 proj;
    float baseColor[4];
};

namespace {
    VkBuffer vertexBuffer;
    VmaAllocation vertexBufferAllocation;

    VkBuffer indexBuffer;
    VmaAllocation indexBufferAllocation;

    VkBuffer uniformBuffer;
    VmaAllocation uniformBufferAllocation;

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

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

    // 1.4
    float figureColor[4] = {1.0f, 1.0f, 1.0f, 1.0f}; 

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

bool initialize() {
	auto& ctx = graphics::internal::context;

	auto vertCode = readFile("shaders/cube.vert.spv");
	auto fragCode = readFile("shaders/cube.frag.spv");
	vertShaderModule = createShaderModule(vertCode, ctx.device);
	fragShaderModule = createShaderModule(fragCode, ctx.device);

	VkBufferCreateInfo vertexBufferInfo{};
	vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vertexBufferInfo.size = sizeof(cubeVertices[0]) * cubeVertices.size();
	vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	vertexBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo vertexAllocInfo{};
	vertexAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	vmaCreateBuffer(ctx.allocator, &vertexBufferInfo, &vertexAllocInfo, 
					&vertexBuffer, &vertexBufferAllocation, nullptr);

	void* vertexData;
	vmaMapMemory(ctx.allocator, vertexBufferAllocation, &vertexData);
	memcpy(vertexData, cubeVertices.data(), vertexBufferInfo.size);
	vmaUnmapMemory(ctx.allocator, vertexBufferAllocation);

	VkBufferCreateInfo indexBufferInfo{};
	indexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	indexBufferInfo.size = sizeof(cubeIndices[0]) * cubeIndices.size();
	indexBufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	indexBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo indexAllocInfo{};
	indexAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	vmaCreateBuffer(ctx.allocator, &indexBufferInfo, &indexAllocInfo, 
					&indexBuffer, &indexBufferAllocation, nullptr);

	void* indexData;
	vmaMapMemory(ctx.allocator, indexBufferAllocation, &indexData);
	memcpy(indexData, cubeIndices.data(), indexBufferInfo.size);
	vmaUnmapMemory(ctx.allocator, indexBufferAllocation);

	VkBufferCreateInfo uBufInfo{};
	uBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	uBufInfo.size = sizeof(UniformBufferObject);
	uBufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	uBufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo uAllocInfo{};
	uAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
	vmaCreateBuffer(ctx.allocator, &uBufInfo, &uAllocInfo, &uniformBuffer, &uniformBufferAllocation, nullptr);
	
	VkDescriptorSetLayoutBinding uboLayoutBinding{};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 1;
	layoutInfo.pBindings = &uboLayoutBinding;
	vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr, &descriptorSetLayout);

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
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

    VkVertexInputAttributeDescription attrDesc[2];
    attrDesc[0].binding = 0; attrDesc[0].location = 0; attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrDesc[0].offset = 0;
    attrDesc[1].binding = 0; attrDesc[1].location = 1; attrDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrDesc[1].offset = 12;

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = 2;
    vertexInputInfo.pVertexAttributeDescriptions = attrDesc;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.x = 0.0f; viewport.y = 0.0f;
    viewport.width = (float)ctx.swapchain_extent.width;
    viewport.height = (float)ctx.swapchain_extent.height;
    viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0}; scissor.extent = ctx.swapchain_extent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1; viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1; viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.lineWidth = 1.0f;
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
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
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

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
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = ctx.render_pass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
    std::cerr << "[ERROR] Failed to create graphics pipeline!\n";
    return false;
	}	
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;
    vkAllocateDescriptorSets(ctx.device, &allocInfo, &descriptorSet);

    VkDescriptorBufferInfo bufferInfoDesc{};
    bufferInfoDesc.buffer = uniformBuffer;
    bufferInfoDesc.offset = 0;
    bufferInfoDesc.range = sizeof(UniformBufferObject);

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = descriptorSet;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &bufferInfoDesc;
    
    vkUpdateDescriptorSets(ctx.device, 1, &descriptorWrite, 0, nullptr);
    
    return true;
}	

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);

    vkDestroyPipeline(ctx.device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
    vkDestroyShaderModule(ctx.device, fragShaderModule, nullptr);
    vkDestroyShaderModule(ctx.device, vertShaderModule, nullptr);
    
    vmaDestroyBuffer(ctx.allocator, uniformBuffer, uniformBufferAllocation);
    vmaDestroyBuffer(ctx.allocator, indexBuffer, indexBufferAllocation);
    vmaDestroyBuffer(ctx.allocator, vertexBuffer, vertexBufferAllocation);
    
    vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, descriptorSetLayout, nullptr);
}

void update([[maybe_unused]] double time) {
	ImGui::ShowDemoWindow();

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

    // ИНТЕРФЕЙС 1.4
    ImGui::Begin("Color Settings");
    ImGui::ColorEdit4("Figure Color", figureColor); 
    ImGui::End();

	auto& ctx = graphics::internal::context;

	UniformBufferObject ubo{};

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

    math::Mat4 temp = math::mat4Multiply(rotationMat, scaleMat);
    temp = math::mat4Multiply(animTranslate, temp);
    ubo.model = math::mat4Multiply(manualTranslate, temp);

	ubo.view = math::mat4LookAt();

    float aspect = (float)ctx.swapchain_extent.width / ctx.swapchain_extent.height;
	if (projectionMode == 1) {
        ubo.proj = math::mat4Perspective(3.14159f / 4.0f, aspect, 0.1f, 100.0f);
    } else {
        float size = 2.0f; 
        ubo.proj = math::mat4Orthographic(-size * aspect, size * aspect, -size, size, 0.1f, 100.0f);
    }

	ubo.baseColor[0] = figureColor[0];
    ubo.baseColor[1] = figureColor[1];
    ubo.baseColor[2] = figureColor[2];
    ubo.baseColor[3] = figureColor[3];

    ubo.proj[1][1] *= -1.0f; 

    void* data;
    vmaMapMemory(ctx.allocator, uniformBufferAllocation, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vmaUnmapMemory(ctx.allocator, uniformBufferAllocation);
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
	renderPassInfo.renderArea.offset = {0, 0};
	renderPassInfo.renderArea.extent = ctx.swapchain_extent;

	VkClearValue clearValues[2];
	clearValues[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}};
	clearValues[1].depthStencil = {1.0f, 0};
	renderPassInfo.clearValueCount = 2;
	renderPassInfo.pClearValues = clearValues;

	vkCmdBeginRenderPass(fd.command_buffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

	VkBuffer vertexBuffers[] = { vertexBuffer };
	VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vertexBuffers, offsets);
	vkCmdBindIndexBuffer(fd.command_buffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

	vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(cubeIndices.size()), 1, 0, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application