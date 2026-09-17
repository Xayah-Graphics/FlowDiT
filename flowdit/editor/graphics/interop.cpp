module;
#include <Windows.h>
#include <GLFW/glfw3.h>
#include <flowdit/cuda.h>
module flowdit.editor.graphics.interop;
import std;
import vulkan;
namespace flowdit::editor {
    Interop::Interop(graphics::Device& gpu, const int cuda_device) : device{gpu} {
        if (const auto result = cudaSetDevice(cuda_device); result != cudaSuccess) throw std::runtime_error{cudaGetErrorString(result)};
        const vk::SemaphoreTypeCreateInfo type{vk::SemaphoreType::eTimeline};
        const vk::ExportSemaphoreCreateInfo exported{vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32, &type};
        for (auto& slot : slots) {
            slot.timeline = vk::raii::Semaphore{device.logical, vk::SemaphoreCreateInfo{{}, &exported}};
            const std::unique_ptr<void, decltype(&CloseHandle)> semaphore_handle{device.logical.getSemaphoreWin32HandleKHR(vk::SemaphoreGetWin32HandleInfoKHR{*slot.timeline, vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32}), CloseHandle};
            cudaExternalSemaphoreHandleDesc semaphore{};
            semaphore.type                = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
            semaphore.handle.win32.handle = semaphore_handle.get();
            if (const auto result = cudaImportExternalSemaphore(std::out_ptr(slot.semaphore), &semaphore); result != cudaSuccess) throw std::runtime_error{cudaGetErrorString(result)};
        }
    }
    Interop::~Interop() {
        device.logical.waitIdle();
    }
    void Interop::publish(const FrameInfo& info, const std::uint8_t* pixels, const std::uint32_t width, const std::uint32_t height, const ::cuda::stream_ref stream) {
        const auto index     = next_slot;
        next_slot            = (next_slot + 1) % slots.size();
        auto& slot           = slots[index];
        const auto semaphore = slot.semaphore.get();
        if (slot.value) {
            cudaExternalSemaphoreWaitParams wait{};
            wait.params.fence.value = slot.value + 1;
            if (const auto result = cudaWaitExternalSemaphoresAsync(&semaphore, &wait, 1, stream.get()); result != cudaSuccess) throw std::runtime_error{cudaGetErrorString(result)};
        }
        const std::size_t bytes = static_cast<std::size_t>(width) * height * 4uz;
        if (slot.buffer.size != bytes) {
            stream.sync();
            slot.pixels.reset();
            slot.memory.reset();
            slot.buffer = graphics::Buffer{device, bytes, false, {}, true};
            const std::unique_ptr<void, decltype(&CloseHandle)> memory_handle{device.logical.getMemoryWin32HandleKHR(vk::MemoryGetWin32HandleInfoKHR{*slot.buffer.memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32}), CloseHandle};
            cudaExternalMemoryHandleDesc memory{};
            memory.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
            memory.handle.win32.handle = memory_handle.get();
            memory.size                = slot.buffer.allocation_size;
            memory.flags               = cudaExternalMemoryDedicated;
            if (const auto result = cudaImportExternalMemory(std::out_ptr(slot.memory), &memory); result != cudaSuccess) throw std::runtime_error{cudaGetErrorString(result)};
            const cudaExternalMemoryBufferDesc mapping{0, slot.buffer.allocation_size, 0};
            if (const auto result = cudaExternalMemoryGetMappedBuffer(std::out_ptr<void*>(slot.pixels), slot.memory.get(), &mapping); result != cudaSuccess) throw std::runtime_error{cudaGetErrorString(result)};
        }
        if (const auto result = cudaMemcpyAsync(slot.pixels.get(), pixels, bytes, cudaMemcpyDeviceToDevice, stream.get()); result != cudaSuccess) throw std::runtime_error{cudaGetErrorString(result)};
        slot.value = slot.value ? slot.value + 2 : 1;
        cudaExternalSemaphoreSignalParams signal{};
        signal.params.fence.value = slot.value;
        if (const auto result = cudaSignalExternalSemaphoresAsync(&semaphore, &signal, 1, stream.get()); result != cudaSuccess) throw std::runtime_error{cudaGetErrorString(result)};
        // Publish only after CUDA has acquired and filled the slot, so one Vulkan submission never waits on two generations of it.
        stream.sync();
        {
            const std::lock_guard lock{mutex};
            frames.push_back({info, index, slot.value});
        }
        glfwPostEmptyEvent();
    }
    std::vector<Interop::Frame> Interop::receive() {
        const std::lock_guard lock{mutex};
        return std::exchange(frames, {});
    }
} // namespace flowdit::editor
