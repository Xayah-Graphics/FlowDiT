module;
#include <GLFW/glfw3.h>
#include <flowdit/cuda.h>
#include <imgui.h>
#include <implot.h>
module flowdit.editor;
import flowdit.editor.platform.window;
import flowdit.editor.graphics.renderer;
import flowdit.editor.graphics.interop;
import flowdit.editor.panels.dataset;
import flowdit.editor.panels.training;
import flowdit.editor.panels.sampling;
import std;
namespace flowdit::editor {
    namespace {
        struct Workspace final {
            Renderer& renderer;
            Interop interop;
            Session session;
            DatasetPanel dataset;
            TrainingPanel training;
            SamplingPanel sampling;
            SessionStatus status;
            std::vector<std::string> messages;
            int page{};
            bool closing{};
            Workspace(Renderer& renderer, int device, DatasetKind kind, std::string dataset, std::string output, std::string checkpoint);
            void receive();
            void draw();
        };
        Workspace::Workspace(Renderer& source, const int device, const DatasetKind kind, std::string dataset_path, std::string output_path, std::string checkpoint_path) : renderer{source}, interop{renderer.device, device}, session{SessionObserver{.notify = [] { glfwPostEmptyEvent(); }, .image = [this](const FrameInfo& info, const std::uint8_t* pixels, const std::uint32_t width, const std::uint32_t height, const ::cuda::stream_ref stream) { interop.publish(info, pixels, width, height, stream); }}} {
            dataset.kind        = kind;
            dataset.directory   = std::move(dataset_path);
            training.directory  = output_path;
            training.device     = device;
            sampling.directory  = output_path + "/inference";
            sampling.checkpoint = std::move(checkpoint_path);
            sampling.device     = device;
            if (!sampling.checkpoint.empty()) sampling.open_checkpoint();
            dataset.open();
        }
        void Workspace::receive() {
            auto update = session.receive();
            status      = update.status;
            for (const auto& frame : interop.receive()) {
                auto& slot = interop.slots[frame.slot];
                if (closing || (frame.info.training_step && !training.follow)) {
                    renderer.discard(*slot.timeline, frame.ready);
                    continue;
                }
                const auto texture = renderer.texture({frame.width, frame.height});
                renderer.copy(texture, slot.buffer, *slot.timeline, frame.ready);
                if (frame.info.training_step) training.preview(renderer, frame.info, texture);
                else sampling.preview(frame.info, texture);
            }
            messages.insert(messages.end(), update.messages.begin(), update.messages.end());
            training.accept(renderer, update);
            sampling.accept(renderer, update);
            if (sampling.checkpoint.empty() && !update.checkpoints.empty()) {
                sampling.checkpoint = update.checkpoints.back().string();
                sampling.open_checkpoint();
            }
            dataset.receive(renderer);
        }
        void Workspace::draw() {
            const auto* viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(viewport->WorkPos);
            ImGui::SetNextWindowSize(viewport->WorkSize);
            ImGui::Begin("FlowDiT", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{0.34F, 0.87F, 0.79F, 1});
            ImGui::TextUnformatted("FLOWDIT");
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::TextDisabled("Image generation workspace");
            ImGui::SameLine(ImGui::GetWindowWidth() - 420);
            const double elapsed = status.started == std::chrono::steady_clock::time_point{} ? 0.0 : std::chrono::duration<double>{(status.busy ? std::chrono::steady_clock::now() : status.ended) - status.started}.count();
            ImGui::Text("%s | %.1fs elapsed", closing ? "Closing after current step..." : stage_names[static_cast<std::size_t>(status.stage)].data(), elapsed);
            ImGui::Separator();
            const std::array labels{"Dataset", "Training", "Inference"};
            for (int i = 0; i < 3; ++i) {
                if (i) ImGui::SameLine();
                const bool selected = page == i;
                if (selected) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.12F, 0.36F, 0.34F, 1});
                if (ImGui::Button(labels[i], {125, 30})) page = i;
                if (selected) ImGui::PopStyleColor();
            }
            ImGui::Separator();
            if (!status.error.empty()) ImGui::TextColored({1, 0.4F, 0.4F, 1}, "%s", status.error.c_str());
            ImGui::BeginDisabled(closing);
            ImGui::BeginChild("workspace", {0, -64});
            ImGui::PushID(page);
            if (page == 0) dataset.draw(renderer, status.busy);
            else if (page == 1) training.draw(renderer, session, status, dataset.dataset, dataset.loaded_directory, dataset.loaded_kind);
            else sampling.draw(renderer, session, status);
            ImGui::PopID();
            ImGui::EndChild();
            ImGui::EndDisabled();
            ImGui::Separator();
            ImGui::BeginChild("log");
            if (!messages.empty()) ImGui::TextWrapped("%s", messages.back().c_str());
            ImGui::TextDisabled("CUDA compute / Vulkan 1.4 | Training and inference run exclusively");
            ImGui::EndChild();
            ImGui::End();
        }
    } // namespace
    int run(const std::span<const std::string_view> arguments) {
        const std::filesystem::path root{FLOWDIT_SOURCE_DIRECTORY};
        std::string dataset;
        DatasetKind kind{DatasetKind::cifar10};
        const auto stamp   = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::string output = (root / "data" / "runs" / std::format("run-{}", stamp)).string();
        std::string checkpoint;
        int device{};
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            const auto option = arguments[i];
            const auto value  = arguments[++i];
            if (option == "--dataset-type") kind = dataset_kind(value);
            else if (option == "--dataset") dataset = value;
            else if (option == "--output") output = value;
            else if (option == "--checkpoint") checkpoint = value;
            else if (option == "--device") std::from_chars(value.data(), value.data() + value.size(), device);
            else throw std::runtime_error{"Unknown Editor option: " + std::string{option}};
        }
        if (dataset.empty()) dataset = (root / "data" / (kind == DatasetKind::cifar10 ? "cifar-10-batches-bin" : "mnist")).string();
        WindowPlatform window;
        Renderer renderer{window, device};
        ImPlot::CreateContext();
        auto& style = ImGui::GetStyle();
        ImGui::StyleColorsDark();
        style.WindowPadding                  = {16, 12};
        style.FramePadding                   = {7, 5};
        style.ItemSpacing                    = {9, 7};
        style.FrameRounding                  = 4;
        style.ChildRounding                  = 5;
        style.Colors[ImGuiCol_WindowBg]      = {0.055F, 0.066F, 0.085F, 1};
        style.Colors[ImGuiCol_ChildBg]       = {0.055F, 0.066F, 0.085F, 1};
        style.Colors[ImGuiCol_FrameBg]       = {0.10F, 0.125F, 0.15F, 1};
        style.Colors[ImGuiCol_Button]        = {0.14F, 0.20F, 0.25F, 1};
        style.Colors[ImGuiCol_ButtonHovered] = {0.18F, 0.36F, 0.36F, 1};
        style.Colors[ImGuiCol_CheckMark]     = {0.35F, 0.87F, 0.78F, 1};
        style.Colors[ImGuiCol_SliderGrab]    = {0.35F, 0.87F, 0.78F, 1};
        {
            Workspace workspace{renderer, device, kind, std::move(dataset), std::move(output), std::move(checkpoint)};
            for (;;) {
                glfwPollEvents();
                if (glfwWindowShouldClose(window.window) && !workspace.closing) {
                    workspace.closing = true;
                    workspace.session.shutdown();
                }
                if (!renderer.begin()) continue;
                workspace.receive();
                workspace.draw();
                renderer.present();
                if (workspace.closing && workspace.status.finished) break;
                glfwWaitEventsTimeout(1.0 / 60.0);
            }
        }
        ImPlot::DestroyContext();
        return 0;
    }
} // namespace flowdit::editor
