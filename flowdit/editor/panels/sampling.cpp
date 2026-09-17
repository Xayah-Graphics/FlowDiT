module;
#include <flowdit/cuda.h>
#include <imgui.h>
module flowdit.editor.panels.sampling;
import flowdit.editor.widgets.controls;
import std;
namespace flowdit::editor {
    void SamplingPanel::accept(Renderer& renderer, const SessionUpdate& update) {
        if (!attached || update.status.mode != Mode::sampling) return;
        progress = update.status;
        for (const auto& sample : update.samples) {
            if (sample->info.training_step) continue;
            result = sample->info;
            picture.upload(renderer, sample->images.model.image, sample->images.labels, sample->images.rgba.data());
        }
    }
    void SamplingPanel::preview(Renderer& renderer, const FrameInfo& info, const std::uint64_t texture) {
        if (picture.texture) renderer.retire(picture.texture);
        picture.texture = texture;
        picture.specification = info.model.image;
        picture.labels.resize(100);
        for (std::uint32_t i = 0; i < 100; ++i) picture.labels[i] = info.request.class_index.value_or(i % static_cast<std::uint32_t>(info.model.image.classes.size()));
    }
    void SamplingPanel::select(const DatasetEntry& dataset, std::string name, const std::size_t index) {
        run = std::move(name);
        checkpoint = {};
        category = -1;
        error.clear();
        if (dataset.runs.empty()) return;
        if (run.empty()) run = dataset.runs.begin()->first;
        const auto& entry = dataset.runs.at(run);
        error = entry.error;
        if (entry.checkpoints.empty()) return;
        checkpoint = entry.checkpoints.at(index);
        if (!checkpoint.error.empty()) error = checkpoint.error;
    }
    bool SamplingPanel::draw(Renderer& renderer, Session& session, SessionStatus& status, const Catalog& catalog, const DatasetEntry& dataset) {
        bool show{};
        ImGui::BeginDisabled(status.busy);
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("Checkpoint");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        const auto label = checkpoint.path.empty() ? std::string{"No checkpoint"} : run_label(run);
        if (ImGui::BeginCombo("##checkpoint", label.c_str())) {
            for (const auto& [name, entry] : dataset.runs) {
                ImGui::PushID(name.c_str());
                ImGui::TextDisabled("%s", run_label(name).c_str());
                if (!entry.error.empty()) ImGui::TextWrapped("%s", entry.error.c_str());
                if (entry.checkpoints.empty()) ImGui::TextDisabled("No checkpoints");
                for (std::size_t i = 0; i < entry.checkpoints.size(); ++i) {
                    const auto& item = entry.checkpoints[i];
                    const auto title = std::format("{} · Step {}", item.path.filename().string(), item.step) + (item.error.empty() ? "" : " / Error");
                    if (ImGui::Selectable(title.c_str(), item.path == checkpoint.path)) select(dataset, name, i);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s%s%s", name.c_str(), item.error.empty() ? "" : "\n", item.error.c_str());
                }
                ImGui::PopID();
                ImGui::Spacing();
            }
            ImGui::EndCombo();
        }
        if (!checkpoint.path.empty()) {
            ImGui::TextDisabled("Step %llu · EMA", checkpoint.step);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", checkpoint.path.string().c_str());
        }
        ImGui::Spacing();
        ImGui::BeginDisabled(checkpoint.path.empty() || !error.empty());
        const auto& model = checkpoint.model;
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("Class");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##class", category < 0 ? "All classes" : model.image.classes[category].c_str())) {
            if (ImGui::Selectable("All classes", category < 0)) category = -1;
            for (int i = 0; i < static_cast<int>(model.image.classes.size()); ++i)
                if (ImGui::Selectable(model.image.classes[i].c_str(), category == i)) category = i;
            ImGui::EndCombo();
        }
        if (ImGui::BeginTable("sampling-parameters", 2, ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoSavedSettings)) {
            ImGui::TableNextColumn();
            number_field("Steps", ImGuiDataType_U32, &request.step_count);
            ImGui::TableNextColumn();
            number_field("CFG", ImGuiDataType_Float, &request.guidance, "%.2g");
            ImGui::EndTable();
        }
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        ImGui::Spacing();
        if (status.busy && status.mode == Mode::sampling) {
            ImGui::BeginDisabled(stopping);
            if (ImGui::Button(stopping ? "Stopping..." : "Stop", {-1, 0})) {
                session.stop();
                stopping = true;
            }
            ImGui::EndDisabled();
        } else {
            ImGui::BeginDisabled(status.busy || checkpoint.path.empty() || !error.empty());
            if (ImGui::Button("Generate", {-1, 0})) {
                if (picture.texture) renderer.retire(picture.texture);
                picture = {};
                canvas = {};
                stopping = false;
                attached = true;
                request.class_index = category < 0 ? std::nullopt : std::optional<std::uint32_t>{static_cast<std::uint32_t>(category)};
                const auto path = catalog.inference(dataset.runs.at(run));
                result = {.path = path, .checkpoint = checkpoint.path, .request = request, .model = model};
                session.start(SampleRequest{.checkpoint = checkpoint.path, .output = path, .device = device, .sampling = request});
                status = {.mode = Mode::sampling, .stage = Stage::loading, .busy = true, .started = std::chrono::steady_clock::now()};
                progress = status;
                error.clear();
                show = true;
            }
            ImGui::EndDisabled();
        }
        if (picture.texture) {
            ImGui::SetCursorPosX(ImGui::GetWindowContentRegionMax().x - ImGui::CalcTextSize("Result").x - 24 * renderer.dpi);
            if (text_button("Result")) show = true;
        }
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4{});
        if (ImGui::CollapsingHeader("Parameters")) {
            ImGui::BeginDisabled(status.busy || checkpoint.path.empty() || !error.empty());
            number_field("Seed", ImGuiDataType_U64, &request.seed, nullptr, true);
            ImGui::EndDisabled();
        }
        ImGui::PopStyleColor();
        if (status.busy && status.mode == Mode::training) ImGui::TextDisabled("Available after training");
        else if (checkpoint.path.empty()) ImGui::TextDisabled("No checkpoint yet");
        if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
        return show;
    }
    void SamplingPanel::draw_images() {
        if (!picture.texture) {
            ImGui::TextDisabled(progress.busy ? "Generating images..." : "Generate images to view results.");
            return;
        }
        ImGui::BeginChild("sampling-images", {0, -ImGui::GetFrameHeightWithSpacing()});
        canvas.draw(picture);
        ImGui::EndChild();
        const auto& sampling = result.request;
        ImGui::TextDisabled("%s / %s", progress.stage == Stage::complete ? "Inference" : "Inference preview", sampling.class_index ? result.model.image.classes[*sampling.class_index].c_str() : "All classes");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%u steps / CFG %.2f / seed %llu\n%s\n%s", sampling.step_count, sampling.guidance, sampling.seed, result.checkpoint.string().c_str(), result.path.string().c_str());
    }
} // namespace flowdit::editor
