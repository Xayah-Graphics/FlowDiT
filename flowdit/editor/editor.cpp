module;
#include <GLFW/glfw3.h>
#include <flowdit/cuda.h>
#include <imgui.h>
module flowdit.editor;
import flowdit.editor.platform.window;
import flowdit.editor.graphics.renderer;
import flowdit.editor.graphics.interop;
import flowdit.editor.panels.dataset;
import flowdit.editor.panels.training;
import flowdit.editor.panels.sampling;
import flowdit.editor.widgets.controls;
import flowdit.runtime.catalog;
import std;
namespace flowdit::editor {
    namespace {
        enum class View { dataset, training, sampling };
        enum class Tool { none, training, sampling };
        struct Workspace final {
            Renderer& renderer;
            Interop interop;
            Session session;
            Catalog catalog;
            DatasetPanel dataset;
            TrainingPanel training;
            SamplingPanel sampling;
            SessionStatus status;
            std::string selected, error;
            View view{View::dataset};
            Tool tool{Tool::none};
            bool left{}, closing{};
            float left_amount{};
            explicit Workspace(Renderer& renderer);
            void select_dataset();
            void receive();
            void draw_application();
            void draw();
        };
        Workspace::Workspace(Renderer& source) : renderer{source}, interop{renderer.device, 0}, session{SessionObserver{.notify = [] { glfwPostEmptyEvent(); }, .image = [this](const FrameInfo& info, const std::uint8_t* pixels, const ::cuda::stream_ref stream) { interop.publish(info, pixels, stream); }}} {
            try {
                catalog.scan();
            } catch (const std::exception& failure) {
                error = failure.what();
            }
        }
        void Workspace::select_dataset() {
            const auto& entry = catalog.datasets.at(selected);
            dataset.open(renderer, entry);
            training.select(renderer, catalog, entry);
            if (sampling.picture.texture) renderer.retire(sampling.picture.texture);
            sampling = {};
            sampling.select(entry);
            view = View::dataset;
            tool = Tool::none;
        }
        void Workspace::receive() {
            auto update = session.receive();
            status      = update.status;
            for (const auto& frame : interop.receive()) {
                auto& slot = interop.slots[frame.slot];
                if (closing) {
                    renderer.discard(*slot.timeline, frame.ready);
                    continue;
                }
                const auto texture = sampling.preview(renderer, frame.info);
                renderer.copy(texture, slot.buffer, *slot.timeline, frame.ready, frame.info.image.width, frame.info.image.height, static_cast<std::uint32_t>(frame.info.labels.size()));
            }
            training.accept(renderer, update);
            sampling.accept(update);
            if (!selected.empty() && !update.checkpoints.empty()) {
                try {
                    auto& entry = catalog.datasets.at(selected);
                    for (const auto& checkpoint : update.checkpoints) catalog.refresh(entry, checkpoint);
                    if (sampling.checkpoint.path.empty()) sampling.select(entry);
                } catch (const std::exception& failure) {
                    error = failure.what();
                }
            }
        }
        void Workspace::draw_application() {
            const float dpi    = renderer.dpi;
            const float height = 48 * dpi;
            ImGui::BeginChild("application", {0, height}, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            const auto origin = ImGui::GetWindowPos();
            const float width = ImGui::GetWindowWidth();
            auto* draw        = ImGui::GetWindowDrawList();
            const auto accent = ImGui::GetColorU32(ImGuiCol_CheckMark);
            const float x = origin.x + 20 * dpi, y = origin.y + 15 * dpi;
            draw->AddQuadFilled({x, y + 22 * dpi}, {x + 4 * dpi, y + 22 * dpi}, {x + 9 * dpi, y}, {x + 5 * dpi, y}, accent);
            draw->AddQuadFilled({x + 5 * dpi, y}, {x + 19 * dpi, y}, {x + 18 * dpi, y + 3 * dpi}, {x + 4 * dpi, y + 3 * dpi}, accent);
            draw->AddQuadFilled({x + 3 * dpi, y + 9 * dpi}, {x + 15 * dpi, y + 9 * dpi}, {x + 14 * dpi, y + 12 * dpi}, {x + 2 * dpi, y + 12 * dpi}, accent);
            ImGui::SetCursorPos({52 * dpi, (height - ImGui::GetTextLineHeight()) * 0.5F});
            ImGui::TextDisabled("FlowDiT");
            const auto now           = std::chrono::steady_clock::now();
            const bool current       = status.mode == Mode::training ? training.attached : sampling.attached;
            const bool completed     = current && !status.busy && status.stage == Stage::complete && now - status.ended < std::chrono::seconds{1};
            const bool indeterminate = dataset.loading.valid() || (status.busy && status.stage == Stage::loading);
            std::string text         = dataset.loading.valid() ? "Loading dataset…" : "Ready";
            std::string elapsed;
            float progress{};
            if (status.busy) {
                const auto seconds = static_cast<std::int64_t>(status.mode == Mode::training ? training.elapsed_base : 0) + std::chrono::duration_cast<std::chrono::seconds>(now - status.started).count();
                elapsed            = std::format("{}:{:02}", seconds / 60, seconds % 60);
                text               = stage_names[static_cast<std::size_t>(status.stage)];
                if (status.mode == Mode::training) {
                    progress = static_cast<float>(status.training.step) / static_cast<float>(training.configuration.end_step);
                    if (status.stage == Stage::preparing) {
                        progress = status.preparation_count ? static_cast<float>(status.prepared) / status.preparation_count : 0;
                        text     = std::format("Preparing latents · {} / {}", status.prepared, status.preparation_count);
                    }
                    if (status.stage == Stage::optimizing) text = std::format("Training · {} / {}", status.training.step, training.configuration.end_step);
                    else if (status.stage == Stage::preview_parameters || status.stage == Stage::preview_ema) text += std::format(" · Step {}", status.training.step);
                    else if (status.stage == Stage::saving) text = "Saving checkpoint…";
                    else if (status.stage == Stage::loading) text = "Initializing training…";
                    if (training.stopping || closing) text = "Stopping and saving…";
                } else {
                    if (status.sampling.step_count) progress = static_cast<float>(status.sampling.step) / status.sampling.step_count;
                    if (status.stage == Stage::sampling) text = std::format("Sampling · {} / {}", status.sampling.step, status.sampling.step_count);
                    else if (status.stage == Stage::loading) text = "Loading checkpoint…";
                    else if (status.stage == Stage::saving) text = "Saving images…";
                    if (sampling.stopping || closing) text = "Stopping…";
                }
            } else if (current && status.stage == Stage::failed) text = "Failed";
            else if (completed) {
                text     = status.mode == Mode::training ? "Target reached" : "Complete";
                progress = 1;
            }
            if (!error.empty() || !dataset.error.empty()) text = "Failed";
            if (closing && !status.busy) text = "Closing…";
            const float text_width    = ImGui::CalcTextSize(text.c_str()).x;
            const float elapsed_width = elapsed.empty() ? 0 : ImGui::CalcTextSize(elapsed.c_str()).x + 14 * dpi;
            const float status_x      = width - 20 * dpi - text_width - elapsed_width;
            ImGui::SetCursorPos({status_x, (height - ImGui::GetTextLineHeight()) * 0.5F});
            draw->AddCircleFilled({origin.x + status_x - 10 * dpi, origin.y + height * 0.5F}, 2 * dpi, ImGui::GetColorU32(status.busy ? ImGuiCol_CheckMark : ImGuiCol_TextDisabled));
            ImGui::TextDisabled("%s", text.c_str());
            if (ImGui::IsItemHovered() && current && !status.error.empty()) ImGui::SetTooltip("%s", status.error.c_str());
            if (!elapsed.empty()) {
                ImGui::SameLine(0, 14 * dpi);
                ImGui::TextDisabled("%s", elapsed.c_str());
            }
            if (indeterminate) {
                const float phase = static_cast<float>(std::fmod(ImGui::GetTime() * 0.6, 1.2)) - 0.2F;
                draw->AddRectFilled({origin.x + width * std::max(0.0F, phase), origin.y}, {origin.x + width * std::min(1.0F, phase + 0.2F), origin.y + 2 * dpi}, accent);
            } else if (status.busy || completed) {
                auto color = ImGui::GetStyleColorVec4(ImGuiCol_CheckMark);
                if (completed) color.w *= 1 - std::chrono::duration<float>(now - status.ended).count();
                draw->AddRectFilled(origin, {origin.x + width * progress, origin.y + 2 * dpi}, ImGui::GetColorU32(color));
            }
            renderer.window.drag_region = ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) ? std::array<float, 4>{} : std::array<float, 4>{112 * dpi, 0, status_x - 22 * dpi, height};
            ImGui::EndChild();
        }
        void Workspace::draw() {
            const bool popup   = ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
            const bool editing = ImGui::GetIO().WantTextInput;
            if (!closing) {
                if (ImGui::Shortcut(ImGuiKey_F11, ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_RouteOverActive) || (!editing && !popup && ImGui::Shortcut(ImGuiKey_F, ImGuiInputFlags_RouteGlobal))) renderer.window.toggle_fullscreen();
                if (!editing && !popup) {
                    if (ImGui::Shortcut(ImGuiKey_GraveAccent, ImGuiInputFlags_RouteGlobal)) left = !left;
                }
            }
            const auto* viewport   = ImGui::GetMainViewport();
            const float dpi        = renderer.dpi;
            const float top        = 48 * dpi;
            const float left_width = std::min(800 * dpi, viewport->WorkSize.x * 0.40F);
            const float height     = viewport->WorkSize.y - top;
            ImGui::SetNextWindowPos(viewport->WorkPos);
            ImGui::SetNextWindowSize(viewport->WorkSize);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
            ImGui::Begin("FlowDiT", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBackground);
            ImGui::PopStyleVar(2);
            ImGui::BeginDisabled(closing);
            draw_application();
            const float target = static_cast<float>(left);
            left_amount        = std::lerp(left_amount, target, std::min(ImGui::GetIO().DeltaTime, 1.0F / 60) / 0.045F);
            if (std::abs(left_amount - target) < 0.01F) left_amount = target;
            const float visible_width = left_width * left_amount;
            if (left_amount > 0) {
                const auto input = left ? ImGuiWindowFlags_None : ImGuiWindowFlags_NoInputs;
                ImGui::SetCursorPos({visible_width - left_width, top + 24 * dpi});
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * left_amount);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{16 * dpi, 0});
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{8 * dpi, 5 * dpi});
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{8 * dpi, 6 * dpi});
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4 * dpi);
                ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0);
                ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 5 * dpi);
                ImGui::PushStyleColor(ImGuiCol_Header, {0.32F, 0.60F, 0.65F, 0.18F});
                ImGui::PushStyleColor(ImGuiCol_HeaderHovered, {0.55F, 0.65F, 0.69F, 0.12F});
                ImGui::PushStyleColor(ImGuiCol_HeaderActive, {0.32F, 0.60F, 0.65F, 0.25F});
                ImGui::PushStyleColor(ImGuiCol_Button, {0.18F, 0.21F, 0.24F, 0.65F});
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.24F, 0.30F, 0.33F, 0.75F});
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0.28F, 0.37F, 0.40F, 0.85F});
                ImGui::PushStyleColor(ImGuiCol_PlotLines, {0.44F, 0.80F, 0.87F, 0.85F});
                ImGui::PushStyleColor(ImGuiCol_Separator, {0.70F, 0.72F, 0.85F, 0.12F});
                ImGui::BeginChild("sidebar", {left_width, height - 104 * dpi}, ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | input);
                ImGui::PushID(selected.c_str());
                ImGui::PushFont(nullptr, 12);
                const float content_y = ImGui::GetCursorPosY() + ImGui::GetFontSize() + 12 * dpi;
                ImGui::TextDisabled("DATASETS");
                ImGui::PopFont();
                ImGui::SetCursorPosY(content_y);
                ImGui::TextUnformatted(selected.empty() ? "No dataset" : catalog.datasets.at(selected).info->specification.name.c_str());
                if (!selected.empty()) {
                    const auto& entry = catalog.datasets.at(selected);
                    if (dataset.dataset) {
                        const auto& image = dataset.dataset->specification;
                        ImGui::TextDisabled("%zu images · %u × %u · %s", dataset.dataset->labels.size(), image.width, image.height, image.channels == 1 ? "Grayscale" : "RGB");
                    } else ImGui::TextDisabled(dataset.loading.valid() ? "Loading images…" : "Dataset could not be loaded");
                    {
                        std::size_t runs{}, checkpoints{};
                        std::optional<std::uint64_t> latest;
                        for (const auto& history : entry.runs | std::views::values) {
                            if (!history.error.empty()) continue;
                            ++runs;
                            for (const auto& checkpoint : history.checkpoints) {
                                if (!checkpoint.error.empty()) continue;
                                ++checkpoints;
                                if (!latest) latest = checkpoint.step;
                            }
                        }
                        const auto name    = "USiT";
                        const auto summary = std::format("{} · {} run{} · {} checkpoint{}", name, runs, runs == 1 ? "" : "s", checkpoints, checkpoints == 1 ? "" : "s");
                        const auto detail  = latest ? std::format(" · latest step {}", *latest) : runs ? " · No checkpoint yet" : " · Not trained";
                        ImGui::TextColored(latest ? ImVec4{0.44F, 0.80F, 0.87F, 1} : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled), "%s%s", summary.c_str(), detail.c_str());
                    }
                    if (!dataset.error.empty()) ImGui::TextWrapped("%s", dataset.error.c_str());
                    ImGui::Spacing();
                    if (dataset.draw_browse()) view = View::dataset;
                    ImGui::Spacing();
                    const float button_width = (ImGui::GetContentRegionAvail().x - 4 * dpi) * 0.5F;
                    if (tool_button("Train", tool == Tool::training, button_width)) tool = tool == Tool::training ? Tool::none : Tool::training;
                    ImGui::SameLine(0, 4 * dpi);
                    if (tool_button("Inference", tool == Tool::sampling, button_width)) tool = tool == Tool::sampling ? Tool::none : Tool::sampling;
                    if (tool != Tool::none) {
                        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{4 * dpi, 6 * dpi});
                        ImGui::SetNextWindowSizeConstraints({0, 0}, {std::numeric_limits<float>::max(), ImGui::GetContentRegionAvail().y * 0.45F});
                        ImGui::BeginChild(tool == Tool::training ? "training-controls" : "sampling-controls", {0, 0}, ImGuiChildFlags_AlwaysUseWindowPadding | ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysAutoResize, ImGuiWindowFlags_NoBackground | input);
                        if (tool == Tool::training) {
                            if (training.draw(renderer, session, status, catalog, entry, dataset.dataset)) view = View::training;
                            if (dataset.category >= 0) ImGui::TextDisabled("Trains on the full dataset");
                        } else if (sampling.draw(renderer, session, status, catalog, entry)) view = View::sampling;
                        ImGui::EndChild();
                        ImGui::PopStyleVar();
                    }
                }
                ImGui::PopID();
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                ImGui::BeginChild("dataset-browser", {0, 0}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground | input);
                const auto previous = selected;
                if (dataset.draw(catalog, selected, status.busy)) view = View::dataset;
                if (selected != previous) select_dataset();
                ImGui::EndChild();
                ImGui::EndChild();
                ImGui::PopStyleColor(8);
                ImGui::PopStyleVar(7);
            }
            dataset.receive(renderer);
            ImGui::SetCursorPos({visible_width, top});
            ImGui::BeginChild("canvas", {viewport->WorkSize.x - visible_width, height}, ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
            if ((training.attached || sampling.attached) && !status.error.empty()) ImGui::TextWrapped("%s", status.error.c_str());
            if (selected.empty()) {
                auto* draw           = ImGui::GetWindowDrawList();
                const auto origin    = ImGui::GetCursorScreenPos();
                const auto available = ImGui::GetContentRegionAvail();
                const ImVec2 center{origin.x + available.x * 0.5F, origin.y + available.y * 0.5F};
                constexpr const char* title = "Welcome to FlowDiT.";
                ImGui::PushFont(nullptr, 30);
                const auto text = ImGui::CalcTextSize(title);
                draw->AddText(ImGui::GetFont(), ImGui::GetFontSize(), {center.x - text.x * 0.5F, center.y - 30 * dpi}, IM_COL32(218, 219, 230, 255), title);
                ImGui::PopFont();
                constexpr const char* subtitle = "Press ` to browse datasets.";
                const auto hint                = ImGui::CalcTextSize(subtitle);
                draw->AddText({center.x - hint.x * 0.5F, center.y + 20 * dpi}, IM_COL32(119, 121, 137, 255), subtitle);
                draw->AddCircle({center.x, center.y - 88 * dpi}, 14 * dpi, IM_COL32(145, 142, 225, 180), 32, 1.5F * dpi);
                draw->AddCircleFilled({center.x + 14 * dpi, center.y - 100 * dpi}, 3 * dpi, IM_COL32(184, 182, 250, 255));
            } else if (view == View::dataset) dataset.draw_images();
            else if (view == View::training) training.draw_images();
            else sampling.draw_images();
            ImGui::EndChild();
            ImGui::EndDisabled();
            ImGui::End();
        }
    } // namespace
    int run(const std::span<const std::string_view> arguments) {
        if (!arguments.empty()) throw std::runtime_error{"Unknown Editor option: " + std::string{arguments.front()}};
        WindowPlatform window;
        Renderer renderer{window, 0};
        ImGui::StyleColorsDark();
        auto& style          = ImGui::GetStyle();
        style.WindowRounding = 16;
        style.ChildRounding = style.FrameRounding = style.GrabRounding = 8;
        style.PopupRounding                                            = 12;
        style.WindowBorderSize                                         = 0;
        style.PopupBorderSize                                          = 0;
        style.ChildBorderSize                                          = 0;
        style.FrameBorderSize                                          = 0;
        style.WindowPadding                                            = {20, 16};
        style.FramePadding                                             = {12, 9};
        style.ItemSpacing                                              = {8, 10};
        style.ScrollbarSize                                            = 8;
        style.ScrollbarRounding                                        = 8;
        style.Colors[ImGuiCol_Text]                                    = {0.93F, 0.93F, 0.96F, 1};
        style.Colors[ImGuiCol_TextDisabled]                            = {0.57F, 0.58F, 0.64F, 1};
        style.Colors[ImGuiCol_WindowBg]                                = {0.095F, 0.10F, 0.125F, 0.985F};
        style.Colors[ImGuiCol_ChildBg]                                 = {0, 0, 0, 0};
        style.Colors[ImGuiCol_PopupBg]                                 = {0.12F, 0.125F, 0.15F, 1};
        style.Colors[ImGuiCol_Border]                                  = {0.70F, 0.72F, 0.85F, 0.10F};
        style.Colors[ImGuiCol_FrameBg]                                 = {0.07F, 0.075F, 0.095F, 1};
        style.Colors[ImGuiCol_FrameBgHovered]                          = {0.14F, 0.145F, 0.18F, 1};
        style.Colors[ImGuiCol_FrameBgActive]                           = {0.16F, 0.16F, 0.21F, 1};
        style.Colors[ImGuiCol_Button]                                  = {0.17F, 0.175F, 0.215F, 1};
        style.Colors[ImGuiCol_ButtonHovered]                           = {0.23F, 0.23F, 0.29F, 1};
        style.Colors[ImGuiCol_ButtonActive]                            = {0.30F, 0.29F, 0.38F, 1};
        style.Colors[ImGuiCol_Header]                                  = {0.35F, 0.34F, 0.55F, 0.35F};
        style.Colors[ImGuiCol_HeaderHovered]                           = {0.45F, 0.44F, 0.67F, 0.35F};
        style.Colors[ImGuiCol_CheckMark]                               = {0.63F, 0.62F, 1, 1};
        {
            Workspace workspace{renderer};
            auto redraw_until = std::chrono::steady_clock::now() + std::chrono::milliseconds{750};
            for (;;) {
                const auto frame_started = std::chrono::steady_clock::now();
                glfwPollEvents();
                if (glfwWindowShouldClose(window.window) && !workspace.closing) {
                    workspace.closing = true;
                    workspace.session.shutdown();
                }
                if (!renderer.begin()) continue;
                workspace.receive();
                const bool loading_dataset = workspace.dataset.loading.valid();
                if (renderer.visible) workspace.draw();
                renderer.present();
                if (workspace.closing && workspace.status.finished) break;
                const auto& io = ImGui::GetIO();
                if (io.MouseDelta.x != 0 || io.MouseDelta.y != 0 || io.MouseWheel != 0 || ImGui::IsMouseClicked(ImGuiMouseButton_Left)) redraw_until = frame_started + std::chrono::milliseconds{750};
                const bool animating  = workspace.left_amount != static_cast<float>(workspace.left);
                const bool completing = workspace.status.stage == Stage::complete && frame_started - workspace.status.ended < std::chrono::seconds{1};
                const bool active     = renderer.visible && (animating || workspace.status.busy || loading_dataset || workspace.dataset.dirty || completing || ImGui::IsAnyItemActive() || frame_started < redraw_until);
                if (active) {
                    const double remaining = 1.0 / (animating ? 120.0 : 60.0) - std::chrono::duration<double>(std::chrono::steady_clock::now() - frame_started).count();
                    if (remaining > 0) glfwWaitEventsTimeout(remaining);
                } else {
                    glfwWaitEvents();
                    redraw_until = std::chrono::steady_clock::now() + std::chrono::milliseconds{750};
                }
            }
        }
        return 0;
    }
} // namespace flowdit::editor
