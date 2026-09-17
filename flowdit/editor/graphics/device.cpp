module;
#include <cuda_runtime_api.h>
module flowdit.editor.graphics.device;
import std;
import vulkan;
namespace flowdit::editor::graphics {
    Instance::Instance(const std::span<const char* const> extensions) {
        const vk::ApplicationInfo application{"FlowDiT", 1, "FlowDiT", 1, vk::ApiVersion14};
        instance = vk::raii::Instance{context, vk::InstanceCreateInfo{{}, &application, 0, nullptr, static_cast<std::uint32_t>(extensions.size()), extensions.data()}};
    }
    Device::Device(Instance& instance, const int cuda_device) {
        cudaDeviceProp cuda_properties{};
        if (const auto result = cudaGetDeviceProperties(&cuda_properties, cuda_device); result != cudaSuccess) throw std::runtime_error{cudaGetErrorString(result)};
        for (const auto& candidate : instance.instance.enumeratePhysicalDevices()) {
            const auto properties = candidate.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties>();
            if (std::memcmp(properties.get<vk::PhysicalDeviceIDProperties>().deviceUUID.data(), cuda_properties.uuid.bytes, sizeof(cuda_properties.uuid.bytes)) == 0) {
                physical = candidate;
                break;
            }
        }
        if (!*physical) throw std::runtime_error{"The selected CUDA device has no matching Vulkan device"};
        memory              = physical.getMemoryProperties();
        const auto extended = physical.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();
        heap_properties     = extended.get<vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();
        const auto families = physical.getQueueFamilyProperties();
        for (std::uint32_t i = 0; i < families.size(); ++i)
            if ((families[i].queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)) == (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)) {
                family = i;
                break;
            }
        std::vector<const char*> extensions{vk::EXTDescriptorHeapExtensionName, vk::KHRShaderUntypedPointersExtensionName, vk::EXTShaderObjectExtensionName, vk::KHRExternalMemoryWin32ExtensionName, vk::KHRExternalSemaphoreWin32ExtensionName, vk::KHRSwapchainExtensionName};
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceDescriptorHeapFeaturesEXT, vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR, vk::PhysicalDeviceShaderObjectFeaturesEXT> features;
        features.get<vk::PhysicalDeviceFeatures2>().features.shaderInt64        = true;
        features.get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters = true;
        auto& v12                                                               = features.get<vk::PhysicalDeviceVulkan12Features>();
        v12.bufferDeviceAddress = v12.scalarBlockLayout = v12.timelineSemaphore = true;
        auto& v13                                                               = features.get<vk::PhysicalDeviceVulkan13Features>();
        v13.synchronization2 = v13.dynamicRendering                                              = true;
        features.get<vk::PhysicalDeviceDescriptorHeapFeaturesEXT>().descriptorHeap               = true;
        features.get<vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR>().shaderUntypedPointers = true;
        features.get<vk::PhysicalDeviceShaderObjectFeaturesEXT>().shaderObject                   = true;
        const std::array priorities{1.0F};
        const vk::DeviceQueueCreateInfo queues{{}, family, 1, priorities.data()};
        logical  = vk::raii::Device{physical, vk::DeviceCreateInfo{{}, 1, &queues, 0, nullptr, static_cast<std::uint32_t>(extensions.size()), extensions.data(), nullptr, &features.get<vk::PhysicalDeviceFeatures2>()}};
        graphics = logical.getQueue(family, 0);
    }
    Device::~Device() {
        if (*logical) logical.waitIdle();
    }
} // namespace flowdit::editor::graphics
