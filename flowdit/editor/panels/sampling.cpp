module;
#include <flowdit/cuda.h>
#include <imgui.h>
module flowdit.editor.panels.sampling;
import flowdit.editor.widgets.controls;
import std;
namespace flowdit::editor {
    void SamplingPanel::accept(Renderer& renderer, const SessionUpdate& update) {
        if (update.status.mode == Mode::sampling) progress = update.status;
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
    void SamplingPanel::open_checkpoint() {
        try {
            model = read_model_configuration(checkpoint);
            loaded_checkpoint = checkpoint;
            category = -1;
            error.clear();
        } catch (const std::exception& failure) {
            loaded_checkpoint.clear();
            error = failure.what();
        }
    }
    bool SamplingPanel::draw(Renderer& renderer, Session& session, SessionStatus& status) {
        bool show{};
        ImGui::BeginDisabled(status.busy);
        if (path_field("Checkpoint", checkpoint, renderer.window, false)) open_checkpoint();
        if (!loaded_checkpoint.empty() && loaded_checkpoint == checkpoint) ImGui::TextDisabled("%s / %u x %u", model.image.name.c_str(), model.image.width, model.image.height);
        ImGui::TextDisabled("Class");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##class", category < 0 ? "All classes" : model.image.classes[category].c_str())) {
            if (ImGui::Selectable("All classes", category < 0)) category = -1;
            for (int i = 0; i < static_cast<int>(model.image.classes.size()); ++i)
                if (ImGui::Selectable(model.image.classes[i].c_str(), category == i)) category = i;
            ImGui::EndCombo();
        }
        ImGui::TextDisabled("ODE steps");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputScalar("##steps", ImGuiDataType_U32, &request.step_count);
        ImGui::TextDisabled("CFG");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputFloat("##cfg", &request.guidance, 0, 0, "%.2f");
        ImGui::TextDisabled("Seed");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputScalar("##seed", ImGuiDataType_U64, &request.seed);
        path_field("Output folder", directory, renderer.window, true);
        ImGui::TextDisabled("EMA / Heun / 100 images");
        ImGui::EndDisabled();
        if (status.busy && status.mode == Mode::sampling) {
            ImGui::BeginDisabled(stopping);
            if (ImGui::Button(stopping ? "Stopping..." : "Stop", {-1, 0})) {
                session.stop();
                stopping = true;
            }
            ImGui::EndDisabled();
        } else {
            ImGui::BeginDisabled(status.busy || loaded_checkpoint.empty() || loaded_checkpoint != checkpoint);
            if (ImGui::Button("Generate", {-1, 0})) {
                if (picture.texture) renderer.retire(picture.texture);
                picture = {};
                canvas = {};
                stopping = false;
                request.class_index = category < 0 ? std::nullopt : std::optional<std::uint32_t>{static_cast<std::uint32_t>(category)};
                const auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                const auto path = std::filesystem::path{directory} / std::format("sample-{}.png", stamp);
                result = {.path = path, .checkpoint = checkpoint, .request = request, .model = model};
                session.start(SampleRequest{.checkpoint = checkpoint, .output = path, .device = device, .sampling = request});
                status = {.mode = Mode::sampling, .stage = Stage::loading, .busy = true, .started = std::chrono::steady_clock::now()};
                progress = status;
                error.clear();
                show = true;
            }
            ImGui::EndDisabled();
        }
        if (status.busy && status.mode == Mode::training) ImGui::TextDisabled("Training is running.");
        if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
        if (progress.mode == Mode::sampling) {
            ImGui::Spacing();
            ImGui::TextDisabled("%s", stage_names[static_cast<std::size_t>(progress.stage)].data());
            if (progress.busy && progress.sampling.step_count) {
                ImGui::TextDisabled("%u / %u steps", progress.sampling.step, progress.sampling.step_count);
                ImGui::ProgressBar(static_cast<float>(progress.sampling.step) / progress.sampling.step_count, {-1, 3 * renderer.dpi}, "");
            }
            if (!progress.error.empty()) ImGui::TextWrapped("%s", progress.error.c_str());
        }
        ImGui::BeginDisabled(!picture.texture);
        if (ImGui::Button("View results")) show = true;
        ImGui::EndDisabled();
        return show;
    }
    void SamplingPanel::draw_images() {
        if (!picture.texture) {
            ImGui::TextDisabled(progress.busy ? "Generating images..." : "Generate images to view results.");
            return;
        }
        ImGui::BeginChild("sampling-images", {0, -2 * ImGui::GetTextLineHeightWithSpacing()});
        canvas.draw(picture);
        ImGui::EndChild();
        const auto& sampling = result.request;
        ImGui::PushTextWrapPos(0);
        ImGui::TextDisabled("%s / %s / %u steps / CFG %.2f / seed %llu", progress.stage == Stage::complete ? "Inference" : "Inference preview", sampling.class_index ? result.model.image.classes[*sampling.class_index].c_str() : "All classes", sampling.step_count, sampling.guidance, sampling.seed);
        ImGui::PopTextWrapPos();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s\n%s", result.checkpoint.string().c_str(), result.path.string().c_str());
    }
} // namespace flowdit::editor
