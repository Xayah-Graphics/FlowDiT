module;
#include <flowdit/cuda.h>
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>
module flowdit.editor.panels.sampling;
import std;
namespace flowdit::editor {
    void SamplingPanel::accept(Renderer& renderer, const SessionUpdate& update) {
        for (const auto& sample : update.samples) {
            if (sample->info.training_step) continue;
            if (picture.texture && std::ranges::none_of(trajectory, [this](const Frame& frame) { return frame.texture == picture.texture; })) renderer.retire(picture.texture);
            picture.info   = sample->info;
            picture.images = sample->images;
            if (!trajectory.empty()) picture.texture = trajectory.back().texture;
            else {
                const auto& images = picture.images;
                const auto height  = static_cast<std::uint32_t>(images.labels.size()) * images.model.image.height;
                picture.texture    = renderer.texture({images.model.image.width, height});
                renderer.upload(picture.texture, images.rgba.data(), images.model.image.width, height, true);
            }
            history.push_back(sample->info);
            history_index = static_cast<int>(history.size()) - 1;
        }
    }
    void SamplingPanel::preview(const FrameInfo& info, const std::uint64_t texture) {
        Frame frame{.info = info, .texture = texture, .images = {.model = info.model, .nfe = info.progress.nfe, .labels = std::vector<std::uint32_t>(100)}};
        for (std::uint32_t i = 0; i < 100; ++i) frame.images.labels[i] = info.request.class_index.value_or(i % static_cast<std::uint32_t>(info.model.image.classes.size()));
        trajectory.push_back(std::move(frame));
        if (follow) frame_index = static_cast<int>(trajectory.size()) - 1;
    }
    void SamplingPanel::clear_trajectory(Renderer& renderer) {
        for (const auto& frame : trajectory)
            if (frame.texture != picture.texture) renderer.retire(frame.texture);
        trajectory.clear();
        frame_index = 0;
    }
    void SamplingPanel::open_picture(Renderer& renderer, const int index, const bool compare) {
        if (!compare) clear_trajectory(renderer);
        auto& target = compare ? comparison : picture;
        auto loaded  = output::read_sample(history[index].path);
        if (target.texture) renderer.retire(target.texture);
        target.info        = std::move(loaded.info);
        target.images      = std::move(loaded.images);
        const auto& images = target.images;
        const auto height  = static_cast<std::uint32_t>(images.labels.size()) * images.model.image.height;
        target.texture     = renderer.texture({images.model.image.width, height});
        renderer.upload(target.texture, images.rgba.data(), images.model.image.width, height, true);
    }
    void SamplingPanel::open_checkpoint() {
        try {
            model             = read_model_configuration(checkpoint);
            loaded_checkpoint = checkpoint;
            category          = -1;
            error.clear();
        } catch (const std::exception& failure) {
            loaded_checkpoint.clear();
            error = failure.what();
        }
    }
    void SamplingPanel::draw(Renderer& renderer, Session& session, const SessionStatus& status) {
        if (!ImGui::BeginTable("sampling-layout", 2, ImGuiTableFlags_Resizable)) return;
        ImGui::TableSetupColumn("Controls", ImGuiTableColumnFlags_WidthFixed, 285);
        ImGui::TableSetupColumn("Images", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableNextColumn();
        ImGui::BeginChild("sampling-controls");
        ImGui::TextUnformatted("SAMPLING");
        ImGui::Separator();
        ImGui::BeginDisabled(status.busy);
        ImGui::TextUnformatted("Checkpoint");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##checkpoint", &checkpoint);
        if (ImGui::Button("Load checkpoint")) open_checkpoint();
        if (loaded_checkpoint == checkpoint && !loaded_checkpoint.empty()) ImGui::Text("%s | %u x %u | %u channel(s)", model.image.name.c_str(), model.image.width, model.image.height, model.image.channels);
        ImGui::TextUnformatted("Output directory");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##output", &directory);
        ImGui::SetNextItemWidth(140);
        ImGui::Combo("Weights", &source, "Parameters\0EMA\0");
        ImGui::SetNextItemWidth(140);
        if (ImGui::BeginCombo("Class", category < 0 ? "All classes" : model.image.classes[category].c_str())) {
            if (ImGui::Selectable("All classes", category < 0)) category = -1;
            for (int i = 0; i < static_cast<int>(model.image.classes.size()); ++i)
                if (ImGui::Selectable(model.image.classes[i].c_str(), category == i)) category = i;
            ImGui::EndCombo();
        }
        ImGui::SetNextItemWidth(140);
        ImGui::Combo("Solver", &solver, "Euler\0Heun\0RK4\0");
        ImGui::SetNextItemWidth(140);
        ImGui::InputScalar("ODE steps", ImGuiDataType_U32, &request.step_count);
        ImGui::SetNextItemWidth(140);
        ImGui::InputFloat("CFG", &request.guidance, 0, 0, "%.2f");
        ImGui::SetNextItemWidth(140);
        ImGui::InputScalar("Seed", ImGuiDataType_U64, &request.seed);
        if (ImGui::Button("Random seed")) {
            static std::mt19937_64 random{std::random_device{}()};
            request.seed = random();
        }
        ImGui::Checkbox("Export FID images (50,000)", &fid);
        if (fid) ImGui::TextWrapped("Exports PNG images and manifest.csv for evaluation.");
        ImGui::BeginDisabled(loaded_checkpoint.empty() || loaded_checkpoint != checkpoint);
        if (ImGui::Button(fid ? "Export images" : "Generate 100 images", {-1, 34})) {
            try {
                clear_trajectory(renderer);
                if (picture.texture) renderer.retire(picture.texture);
                picture             = {};
                request.solver      = static_cast<SamplingSolver>(solver);
                request.class_index = category < 0 ? std::nullopt : std::optional<std::uint32_t>{static_cast<std::uint32_t>(category)};
                const auto stamp    = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                const auto path     = std::filesystem::path{directory} / std::format("{}-{}", fid ? "fid" : "sample", stamp);
                follow              = true;
                session.start(SampleRequest{.checkpoint = checkpoint, .output = fid ? path : std::filesystem::path{path.string() + ".png"}, .device = device, .source = static_cast<ParameterSource>(source), .sampling = request, .fid = fid});
                error.clear();
            } catch (const std::exception& failure) {
                error = failure.what();
            }
        }
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        ImGui::BeginDisabled(!status.busy || status.mode == Mode::training);
        if (ImGui::Button("Stop sampling")) session.stop();
        ImGui::EndDisabled();
        ImGui::Separator();
        ImGui::TextUnformatted("RESULTS");
        ImGui::BeginDisabled(status.busy);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##history", history_index < 0 ? "Saved results" : history[history_index].path.filename().string().c_str())) {
            for (int i = 0; i < history.size(); ++i)
                if (ImGui::Selectable(history[i].path.filename().string().c_str(), i == history_index)) {
                    history_index = i;
                    open_picture(renderer, i, false);
                }
            ImGui::EndCombo();
        }
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##comparison", comparison_index < 0 ? "Compare with..." : history[comparison_index].path.filename().string().c_str())) {
            if (ImGui::Selectable("No comparison")) comparison_index = -1;
            for (int i = 0; i < history.size(); ++i)
                if (ImGui::Selectable(history[i].path.filename().string().c_str(), i == comparison_index)) {
                    comparison_index = i;
                    open_picture(renderer, i, true);
                }
            ImGui::EndCombo();
        }
        ImGui::BeginDisabled(!picture.texture || picture.images.rgba.empty() || (!trajectory.empty() && trajectory[frame_index].texture != picture.texture));
        if (ImGui::Button("Save grid copy")) {
            const auto path = std::filesystem::path{directory} / (picture.info.path.stem().string() + "-grid.png");
            output::write_png(path, picture.images);
        }
        ImGui::BeginDisabled(canvas.selected < 0);
        if (ImGui::Button("Save selected image")) {
            const auto path = std::filesystem::path{directory} / std::format("{}-{:03}.png", picture.info.path.stem().string(), canvas.selected);
            output::write_png(path, picture.images, static_cast<std::size_t>(canvas.selected));
        }
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        if (!picture.info.path.empty()) ImGui::TextWrapped("%s", picture.info.path.string().c_str());
        if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
        ImGui::EndChild();
        ImGui::TableNextColumn();
        if (status.mode == Mode::sampling || status.mode == Mode::fid) {
            const auto& progress = status.sampling;
            ImGui::Text("ODE %u / %u | NFE %u | Model forwards %u | t %.3f", progress.step, progress.step_count, progress.nfe, progress.nfe * 2, progress.time);
            if (progress.step_count) ImGui::ProgressBar(static_cast<float>(progress.step) / progress.step_count, {-1, 8}, "");
            if (status.mode == Mode::fid) ImGui::Text("Exported %u / 50000", status.exported);
        }
        if (!trajectory.empty()) {
            ImGui::Checkbox("Follow live", &follow);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-100);
            if (ImGui::SliderInt("Trajectory", &frame_index, 0, static_cast<int>(trajectory.size()) - 1)) follow = false;
            const auto& frame = trajectory[frame_index];
            ImGui::Text("Showing x_t | step %u | t %.4f | seed %llu", frame.info.progress.step, frame.info.progress.time, frame.info.request.seed);
        }
        const std::uint64_t texture = trajectory.empty() ? picture.texture : trajectory[frame_index].texture;
        const auto& images          = trajectory.empty() || texture == picture.texture ? picture.images : trajectory[frame_index].images;
        if (comparison_index >= 0 && comparison.texture) {
            if (ImGui::BeginTable("comparison", 2, ImGuiTableFlags_SizingStretchSame)) {
                ImGui::TableNextColumn();
                ImGui::PushID(0);
                ImGui::BeginChild("current");
                canvas.draw(texture, images);
                ImGui::EndChild();
                ImGui::PopID();
                ImGui::TableNextColumn();
                ImGui::PushID(1);
                ImGui::BeginChild("reference");
                canvas.draw(comparison.texture, comparison.images);
                ImGui::EndChild();
                ImGui::PopID();
                ImGui::EndTable();
            }
        } else canvas.draw(texture, images);
        ImGui::EndTable();
    }
} // namespace flowdit::editor
