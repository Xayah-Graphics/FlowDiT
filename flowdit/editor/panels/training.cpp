module;
#include <flowdit/cuda.h>
#include <imgui.h>
#include <implot.h>
#include <misc/cpp/imgui_stdlib.h>
module flowdit.editor.panels.training;
import std;
namespace flowdit::editor {
    void TrainingPanel::accept(Renderer& renderer, const SessionUpdate& update) {
        if (update.status.mode == Mode::training && (update.status.busy || !update.metrics.empty() || !update.checkpoints.empty())) state = update.status.training;
        metrics.insert(metrics.end(), update.metrics.begin(), update.metrics.end());
        for (const auto& path : update.checkpoints)
            if (!std::ranges::contains(checkpoints, path)) checkpoints.push_back(path);
        if (!update.checkpoints.empty()) checkpoint = update.checkpoints.back().string();
        for (const auto& sample : update.samples) {
            if (!sample->info.training_step) continue;
            const auto existing = std::ranges::find(history, sample->info.path, &SampleInfo::path);
            if (existing == history.end()) history.push_back(sample->info);
            else *existing = sample->info;
            if (!follow) continue;
            const std::size_t source = static_cast<std::size_t>(sample->info.source);
            auto& picture            = previews[source];
            if (!picture.texture || picture.info.training_step != sample->info.training_step) {
                if (picture.texture) renderer.retire(picture.texture);
                const auto& images = sample->images;
                const auto height  = static_cast<std::uint32_t>(images.labels.size()) * images.model.image.height;
                picture.texture    = renderer.texture({images.model.image.width, height});
                renderer.upload(picture.texture, images.rgba.data(), images.model.image.width, height, true);
            }
            picture.info   = sample->info;
            picture.images = sample->images;
            selected_step  = sample->info.training_step;
        }
        if (update.batch) {
            batch = update.batch;
            if (batch_picture.texture) renderer.retire(batch_picture.texture);
            batch_picture.images  = batch->images;
            const auto& images    = batch->images;
            const auto height     = static_cast<std::uint32_t>(images.labels.size()) * images.model.image.height;
            batch_picture.texture = renderer.texture({images.model.image.width, height});
            renderer.upload(batch_picture.texture, images.rgba.data(), images.model.image.width, height, true);
        }
    }
    void TrainingPanel::preview(Renderer& renderer, const FrameInfo& info, const std::uint64_t texture) {
        auto& picture = previews[static_cast<std::size_t>(info.source)];
        if (picture.texture) renderer.retire(picture.texture);
        picture.texture = texture;
        picture.info    = {.request = info.request, .source = info.source, .training_step = info.training_step, .nfe = info.progress.nfe};
        picture.images  = {.model = info.model, .nfe = info.progress.nfe, .labels = std::vector<std::uint32_t>(100)};
        for (std::uint32_t i = 0; i < 100; ++i) picture.images.labels[i] = info.request.class_index.value_or(i % static_cast<std::uint32_t>(info.model.image.classes.size()));
        selected_step = info.training_step;
    }
    void TrainingPanel::open_run(Renderer& renderer) {
        try {
            configuration = output::read_configuration(directory);
            auto loaded   = output::read_history(directory);
            metrics       = std::move(loaded.metrics);
            history       = std::move(loaded.samples);
            checkpoints   = std::move(loaded.checkpoints);
            checkpoint    = output::latest_checkpoint(directory).string();
            state         = {.seed = configuration.seed};
            if (!metrics.empty()) state = {.step = metrics.back().step, .processed_samples = metrics.back().step * Trainer::batch, .seed = configuration.seed, .elapsed_seconds = metrics.back().training_seconds};
            batch.reset();
            if (!history.empty()) select_preview(renderer, history.back().training_step);
            error.clear();
        } catch (const std::exception& failure) {
            error = failure.what();
        }
    }
    void TrainingPanel::select_preview(Renderer& renderer, const std::uint64_t step) {
        for (auto& picture : previews) {
            if (picture.texture) renderer.retire(picture.texture);
            picture = {};
        }
        for (const auto& info : history) {
            if (info.training_step != step) continue;
            auto sample        = output::read_sample(info.path);
            auto& picture      = previews[static_cast<std::size_t>(info.source)];
            picture.info       = std::move(sample.info);
            picture.images     = std::move(sample.images);
            const auto& images = picture.images;
            const auto height  = static_cast<std::uint32_t>(images.labels.size()) * images.model.image.height;
            picture.texture    = renderer.texture({images.model.image.width, height});
            renderer.upload(picture.texture, images.rgba.data(), images.model.image.width, height, true);
        }
        selected_step = step;
        follow        = false;
    }
    void TrainingPanel::draw(Renderer& renderer, Session& session, const SessionStatus& status, const std::shared_ptr<const Dataset>& dataset, const std::string& dataset_path, const DatasetKind dataset_type) {
        if (!ImGui::BeginTable("training-layout", 2, ImGuiTableFlags_Resizable)) return;
        ImGui::TableSetupColumn("Controls", ImGuiTableColumnFlags_WidthFixed, 285);
        ImGui::TableSetupColumn("Monitor", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableNextColumn();
        ImGui::BeginChild("training-controls");
        ImGui::TextUnformatted("TRAINING RUN");
        ImGui::Separator();
        ImGui::BeginDisabled(status.busy);
        ImGui::TextUnformatted("Output directory");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##output", &directory);
        if (ImGui::Button("Open run")) open_run(renderer);
        ImGui::TextUnformatted("Resume checkpoint (empty = new)");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##resume", &checkpoint);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##checkpoints", "Saved checkpoints")) {
            for (const auto& path : checkpoints)
                if (ImGui::Selectable(path.filename().string().c_str())) checkpoint = path.string();
            ImGui::EndCombo();
        }
        ImGui::SetNextItemWidth(130);
        ImGui::InputScalar("Target step", ImGuiDataType_U64, &configuration.end_step);
        ImGui::SetNextItemWidth(130);
        ImGui::InputScalar("Seed", ImGuiDataType_U64, &configuration.seed);
        ImGui::Text("Batch %u | width 256 | 8 blocks", Trainer::batch);
        if (dataset) ImGui::Text("%s | %u x %u | %u channel(s)", dataset->specification.name.c_str(), dataset->specification.width, dataset->specification.height, dataset->specification.channels);
        ImGui::SetNextItemWidth(110);
        ImGui::InputFloat("Learning rate", &configuration.optimizer.learning_rate, 0, 0, "%.6f");
        if (ImGui::CollapsingHeader("Optimizer")) {
            ImGui::SetNextItemWidth(100);
            ImGui::InputFloat("Beta 1", &configuration.optimizer.first_decay, 0, 0, "%.3f");
            ImGui::SetNextItemWidth(100);
            ImGui::InputFloat("Beta 2", &configuration.optimizer.second_decay, 0, 0, "%.4f");
            ImGui::SetNextItemWidth(100);
            ImGui::InputFloat("Epsilon", &configuration.optimizer.epsilon, 0, 0, "%.1e");
            ImGui::SetNextItemWidth(100);
            ImGui::InputFloat("Weight decay", &configuration.optimizer.weight_decay, 0, 0, "%.5f");
            ImGui::SetNextItemWidth(100);
            ImGui::InputScalar("EMA half-life", ImGuiDataType_U64, &configuration.optimizer.exponential_average.half_life_samples);
            ImGui::SetNextItemWidth(100);
            ImGui::InputFloat("EMA ramp", &configuration.optimizer.exponential_average.ramp_up_ratio, 0, 0, "%.3f");
        }
        if (ImGui::CollapsingHeader("Intervals / fixed preview", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SetNextItemWidth(100);
            ImGui::InputScalar("Execute steps", ImGuiDataType_U32, &configuration.execution_steps);
            ImGui::SetNextItemWidth(100);
            ImGui::InputScalar("Log every", ImGuiDataType_U32, &configuration.log_interval);
            ImGui::SetNextItemWidth(100);
            ImGui::InputScalar("Preview every", ImGuiDataType_U32, &configuration.preview_interval);
            ImGui::SetNextItemWidth(100);
            ImGui::InputScalar("Save every", ImGuiDataType_U32, &configuration.save_interval);
            ImGui::SetNextItemWidth(100);
            ImGui::InputScalar("Preview steps", ImGuiDataType_U32, &configuration.preview.step_count);
            ImGui::SetNextItemWidth(100);
            ImGui::InputScalar("Preview seed", ImGuiDataType_U64, &configuration.preview.seed);
            ImGui::SetNextItemWidth(100);
            ImGui::InputFloat("Preview CFG", &configuration.preview.guidance, 0, 0, "%.2f");
        }
        ImGui::BeginDisabled(!dataset);
        if (ImGui::Button(checkpoint.empty() ? "Start training" : "Resume training", {-1, 34})) {
            configuration.dataset      = dataset_path;
            configuration.dataset_type = dataset_type;
            configuration.output       = directory;
            configuration.device       = device;
            if (checkpoint.empty()) {
                state = {};
                metrics.clear();
                history.clear();
                checkpoints.clear();
                for (auto& picture : previews) {
                    if (picture.texture) renderer.retire(picture.texture);
                    picture = {};
                }
            }
            batch.reset();
            follow = true;
            session.start(TrainRequest{configuration, dataset, checkpoint});
        }
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        if (!dataset) ImGui::TextWrapped("Open a dataset in the Dataset page first.");
        ImGui::BeginDisabled(!status.busy || status.mode != Mode::training);
        if (ImGui::Button(status.stage == Stage::paused ? "Continue" : "Pause")) {
            if (status.stage == Stage::paused) session.resume();
            else session.pause();
        }
        ImGui::SameLine();
        if (ImGui::Button("Save now")) session.save();
        ImGui::SameLine();
        if (ImGui::Button("Stop")) session.stop();
        if (ImGui::Button("Inspect last batch")) session.inspect();
        ImGui::EndDisabled();
        if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
        ImGui::EndChild();
        ImGui::TableNextColumn();
        ImGui::Text("Step %llu / %llu | %.2fs optimization", state.step, configuration.end_step, state.elapsed_seconds);
        const float fraction = static_cast<float>(state.step) / static_cast<float>(configuration.end_step);
        ImGui::ProgressBar(fraction, {-1, 8}, "");
        if (!metrics.empty()) {
            const auto& last = metrics.back();
            ImGui::Text("Loss %.6f | %.1f samples/s | %llu samples", last.loss, last.samples_per_second, state.processed_samples);
            const double eta = static_cast<double>(configuration.end_step > state.step ? configuration.end_step - state.step : 0) * Trainer::batch / last.samples_per_second;
            ImGui::SameLine();
            ImGui::Text("| ETA %.1f min", eta / 60);
        }
        ImGui::Checkbox("Smooth loss", &smooth);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        ImGui::SliderFloat("Weight", &smoothing, 0.01F, 1.0F, "%.2f");
        std::vector<double> steps, losses, smoothed, speeds;
        for (const auto& metric : metrics) {
            steps.push_back(static_cast<double>(metric.step));
            losses.push_back(metric.loss);
            smoothed.push_back(smoothed.empty() ? metric.loss : std::lerp(smoothed.back(), static_cast<double>(metric.loss), static_cast<double>(smoothing)));
            speeds.push_back(metric.samples_per_second);
        }
        if (ImPlot::BeginPlot("Loss", {-1, 180})) {
            ImPlot::SetupAxes("Step", "MSE", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("Recorded", steps.data(), losses.data(), static_cast<int>(steps.size()));
            if (smooth) ImPlot::PlotLine("Smoothed", steps.data(), smoothed.data(), static_cast<int>(steps.size()));
            ImPlot::EndPlot();
        }
        if (ImPlot::BeginPlot("Throughput", {-1, 150})) {
            ImPlot::SetupAxes("Step", "samples/s", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("Training", steps.data(), speeds.data(), static_cast<int>(steps.size()));
            ImPlot::EndPlot();
        }
        if (batch && ImGui::CollapsingHeader("Captured training batch")) {
            ImGui::Text("Actual x_t at step %llu; unconditional = dropped conditioning", batch->step);
            ImGui::BeginChild("batch-view", {-1, 300}, ImGuiChildFlags_Borders);
            batch_canvas.draw(batch_picture.texture, batch_picture.images, batch->times);
            ImGui::EndChild();
        }
        ImGui::Checkbox("Follow training previews", &follow);
        ImGui::SameLine();
        std::vector<std::uint64_t> milestones;
        for (const auto& info : history)
            if (!std::ranges::contains(milestones, info.training_step)) milestones.push_back(info.training_step);
        std::ranges::sort(milestones);
        ImGui::SetNextItemWidth(160);
        if (ImGui::BeginCombo("History", std::format("Step {}", selected_step).c_str())) {
            for (const auto step : milestones)
                if (ImGui::Selectable(std::format("Step {}", step).c_str(), step == selected_step)) select_preview(renderer, step);
            ImGui::EndCombo();
        }
        if (ImGui::BeginTable("preview-pair", 2, ImGuiTableFlags_SizingStretchSame)) {
            for (std::size_t source = 0; source < 2; ++source) {
                ImGui::TableNextColumn();
                ImGui::PushID(static_cast<int>(source));
                const auto& picture = previews[source];
                ImGui::Text("%s | step %llu", source == 0 ? "Parameters" : "EMA", picture.info.training_step);
                ImGui::BeginChild("preview", {0, std::max(200.0F, ImGui::GetContentRegionAvail().y)});
                canvas.draw(picture.texture, picture.images);
                ImGui::EndChild();
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        ImGui::EndTable();
    }
} // namespace flowdit::editor
