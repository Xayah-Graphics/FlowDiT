module;
#include <flowdit/cuda.h>
export module flowdit.editor.graphics.interop;
import flowdit.editor.graphics.device;
import flowdit.editor.graphics.resources;
export import flowdit.runtime.session;
import std;
import vulkan;
export namespace flowdit::editor {
    struct Interop final {
        struct Slot final {
            graphics::Buffer buffer;
            vk::raii::Semaphore timeline{nullptr};
            std::unique_ptr<std::remove_pointer_t<cudaExternalMemory_t>, decltype(&cudaDestroyExternalMemory)> memory{nullptr, cudaDestroyExternalMemory};
            std::unique_ptr<std::remove_pointer_t<cudaExternalSemaphore_t>, decltype(&cudaDestroyExternalSemaphore)> semaphore{nullptr, cudaDestroyExternalSemaphore};
            std::unique_ptr<std::uint8_t, decltype(&cudaFree)> pixels{nullptr, cudaFree};
            std::uint64_t value{};
        };
        struct Frame final {
            FrameInfo info;
            std::size_t slot{};
            std::uint64_t ready{};
        };
        graphics::Device& device;
        std::array<Slot, 2> slots;
        Interop(graphics::Device& device, int cuda_device);
        ~Interop();
        void publish(const FrameInfo& info, const std::uint8_t* pixels, ::cuda::stream_ref stream);
        std::vector<Frame> receive();

    private:
        std::size_t next_slot{};
        std::mutex mutex;
        std::vector<Frame> frames;
    };
} // namespace flowdit::editor
