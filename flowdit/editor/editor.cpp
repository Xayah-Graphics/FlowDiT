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
        enum class View { dataset, training, sampling };
        struct Workspace final {
            Renderer& renderer;
            Interop interop;
            Session session;
            DatasetPanel dataset;
            TrainingPanel training;
            SamplingPanel sampling;
            SessionStatus status;
            View view{View::dataset};
            bool left{true}, right{true}, closing{};
            Workspace(Renderer& renderer, int device, DatasetKind kind, std::string dataset, std::string output, std::string checkpoint);
            void receive();
            void draw();
        };
        Workspace::Workspace(Renderer& source, const int device, const DatasetKind kind, std::string dataset_path, std::string output_path, std::string checkpoint_path) : renderer{source}, interop{renderer.device, device}, session{SessionObserver{.notify = [] { glfwPostEmptyEvent(); }, .image = [this](const FrameInfo& info, const std::uint8_t* pixels, const std::uint32_t width, const std::uint32_t height, const ::cuda::stream_ref stream) {
            if (!info.training_step) interop.publish(info, pixels, width, height, stream);
        }}} {
            dataset.kind = kind;
            dataset.directory = std::move(dataset_path);
            training.directory = output_path;
            training.device = device;
            if (kind == DatasetKind::mnist) training.configuration.end_step = 20'000;
            sampling.directory = output_path + "/inference";
            sampling.checkpoint = std::move(checkpoint_path);
            sampling.device = device;
            if (!sampling.checkpoint.empty()) sampling.open_checkpoint();
            dataset.open();
        }
        void Workspace::receive() {
            auto update = session.receive();
            status = update.status;
            for (const auto& frame : interop.receive()) {
                auto& slot = interop.slots[frame.slot];
                if (closing) {
                    renderer.discard(*slot.timeline, frame.ready);
                    continue;
                }
                const auto texture = renderer.texture({frame.width, frame.height});
                renderer.copy(texture, slot.buffer, *slot.timeline, frame.ready);
                sampling.preview(renderer, frame.info, texture);
            }
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
            const float dpi = renderer.dpi;
            const float left_width = left ? 244 * dpi : 0;
            const float right_width = right ? 220 * dpi : 0;
            const float center_width = viewport->WorkSize.x - left_width - right_width;
            ImGui::SetNextWindowPos(viewport->WorkPos);
            ImGui::SetNextWindowSize(viewport->WorkSize);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
            ImGui::Begin("FlowDiT", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
            ImGui::PopStyleVar(2);
            ImGui::BeginDisabled(closing);
            ImGui::PushStyleColor(ImGuiCol_ChildBg, {0.098F, 0.110F, 0.118F, 1});
            if (left) {
                ImGui::SetCursorPos({0, 0});
                ImGui::BeginChild("left-panel", {left_width, 0}, ImGuiChildFlags_AlwaysUseWindowPadding);
                ImGui::TextUnformatted("Dataset");
                ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 22 * dpi);
                if (ImGui::SmallButton("<")) left = false;
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Hide dataset and training");
                ImGui::Spacing();
                ImGui::PushID("dataset");
                if (dataset.draw(renderer, status.busy)) view = View::dataset;
                ImGui::PopID();
                ImGui::Dummy({0, 14 * dpi});
                ImGui::Separator();
                ImGui::Dummy({0, 10 * dpi});
                ImGui::PushID("training");
                if (training.draw(renderer, session, status, dataset.dataset, dataset.loaded_directory, dataset.loaded_kind, dataset.loading.valid())) view = View::training;
                ImGui::PopID();
                ImGui::EndChild();
            }
            if (right) {
                ImGui::SetCursorPos({viewport->WorkSize.x - right_width, 0});
                ImGui::BeginChild("right-panel", {right_width, 0}, ImGuiChildFlags_AlwaysUseWindowPadding);
                ImGui::TextUnformatted("Inference");
                ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 22 * dpi);
                if (ImGui::SmallButton(">")) right = false;
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Hide inference");
                ImGui::Spacing();
                ImGui::PushID("sampling");
                if (sampling.draw(renderer, session, status)) view = View::sampling;
                ImGui::PopID();
                ImGui::EndChild();
            }
            ImGui::PopStyleColor();
            ImGui::SetCursorPos({left_width, 0});
            ImGui::BeginChild("canvas", {center_width, 0}, ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            if (!left) {
                if (ImGui::SmallButton(">")) left = true;
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show dataset and training");
            }
            if (!right) {
                ImGui::SetCursorPos({center_width - 40 * dpi, ImGui::GetStyle().WindowPadding.y});
                if (ImGui::SmallButton("<")) right = true;
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show inference");
            }
            const bool popup = ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
            renderer.window.drag_region = popup ? std::array<float, 4>{} : std::array<float, 4>{left_width + 46 * dpi, 0, left_width + center_width - 46 * dpi, 34 * dpi};
            ImGui::SetCursorPosY(38 * dpi);
            if (closing) ImGui::TextDisabled(status.mode == Mode::training ? "Stopping and saving before closing..." : "Stopping before closing...");
            if (view == View::dataset) dataset.draw_images(renderer);
            else if (view == View::training) training.draw_images();
            else sampling.draw_images();
            ImGui::EndChild();
            ImGui::EndDisabled();
            ImGui::End();
        }
    } // namespace
    int run(const std::span<const std::string_view> arguments) {
        const std::filesystem::path root{FLOWDIT_SOURCE_DIRECTORY};
        std::string dataset;
        DatasetKind kind{DatasetKind::cifar10};
        const auto stamp = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::string output = (root / "data" / "runs" / std::format("run-{}", stamp)).string();
        std::string checkpoint;
        int device{};
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            const auto option = arguments[i];
            const auto value = arguments[++i];
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
        ImGui::StyleColorsDark();
        auto& style = ImGui::GetStyle();
        style.WindowPadding = {18, 18};
        style.FramePadding = {6, 4};
        style.ItemSpacing = {8, 5};
        style.ItemInnerSpacing = {6, 4};
        style.WindowRounding = 0;
        style.ChildRounding = 0;
        style.FrameRounding = 3;
        style.PopupRounding = 4;
        style.ChildBorderSize = 0;
        style.FrameBorderSize = 0;
        style.ScrollbarSize = 6;
        style.Colors[ImGuiCol_Text] = {0.84F, 0.86F, 0.87F, 1};
        style.Colors[ImGuiCol_TextDisabled] = {0.47F, 0.51F, 0.53F, 1};
        style.Colors[ImGuiCol_WindowBg] = {0.078F, 0.086F, 0.094F, 1};
        style.Colors[ImGuiCol_ChildBg] = {0, 0, 0, 0};
        style.Colors[ImGuiCol_PopupBg] = {0.11F, 0.12F, 0.13F, 1};
        style.Colors[ImGuiCol_Border] = {0.24F, 0.27F, 0.28F, 0.35F};
        style.Colors[ImGuiCol_FrameBg] = {0.13F, 0.145F, 0.15F, 1};
        style.Colors[ImGuiCol_FrameBgHovered] = {0.17F, 0.19F, 0.20F, 1};
        style.Colors[ImGuiCol_FrameBgActive] = {0.20F, 0.23F, 0.24F, 1};
        style.Colors[ImGuiCol_Button] = {0, 0, 0, 0};
        style.Colors[ImGuiCol_ButtonHovered] = {0.23F, 0.28F, 0.27F, 0.45F};
        style.Colors[ImGuiCol_ButtonActive] = {0.26F, 0.33F, 0.30F, 0.65F};
        style.Colors[ImGuiCol_Header] = {0.23F, 0.30F, 0.27F, 0.5F};
        style.Colors[ImGuiCol_HeaderHovered] = {0.25F, 0.32F, 0.29F, 0.65F};
        style.Colors[ImGuiCol_HeaderActive] = {0.28F, 0.36F, 0.32F, 0.8F};
        style.Colors[ImGuiCol_Separator] = {0.25F, 0.28F, 0.29F, 0.45F};
        style.Colors[ImGuiCol_CheckMark] = {0.55F, 0.70F, 0.63F, 1};
        style.Colors[ImGuiCol_PlotHistogram] = {0.55F, 0.70F, 0.63F, 1};
        style.Colors[ImGuiCol_NavCursor] = {0.55F, 0.70F, 0.63F, 0.65F};
        auto& plot = ImPlot::GetStyle();
        plot.PlotPadding = {4, 4};
        plot.PlotBorderSize = 0;
        plot.Colors[ImPlotCol_PlotBg] = {0, 0, 0, 0};
        plot.Colors[ImPlotCol_AxisGrid] = {0.3F, 0.34F, 0.35F, 0.22F};
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
                if (renderer.visible) workspace.draw();
                renderer.present();
                if (workspace.closing && workspace.status.finished) break;
                glfwWaitEventsTimeout(1.0 / 60.0);
            }
        }
        ImPlot::DestroyContext();
        return 0;
    }
} // namespace flowdit::editor
