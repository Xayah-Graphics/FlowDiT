module;
#include <flowdit/cuda.h>
#include <imgui.h>
#include <implot.h>
module flowdit.editor.panels.training;
import flowdit.editor.widgets.controls;
import std;
namespace flowdit::editor {
    void TrainingPanel::accept(Renderer& renderer, const SessionUpdate& update) {
        if (update.status.mode == Mode::training) progress = update.status;
        metrics.insert(metrics.end(), update.metrics.begin(), update.metrics.end());
        for (const auto& sample : update.samples) {
            if (!sample->info.training_step || sample->info.source != ParameterSource::exponential_average) continue;
            preview = sample->info;
            picture.upload(renderer, sample->images.model.image, sample->images.labels, sample->images.rgba.data());
        }
    }
    bool TrainingPanel::draw(Renderer& renderer, Session& session, SessionStatus& status, const std::shared_ptr<const Dataset>& dataset, const std::string& dataset_path, const DatasetKind dataset_type, const bool loading) {
        bool show{};
        ImGui::TextUnformatted("Training");
        ImGui::SameLine();
        if (ImGui::SmallButton("View preview")) show = true;
        ImGui::BeginDisabled(status.busy);
        ImGui::SetNextItemWidth(-1);
        ImGui::Combo("##start-from", &resume, "New training\0Resume checkpoint\0");
        if (resume) path_field("Checkpoint", checkpoint, renderer.window, false);
        ImGui::TextDisabled("Target step");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputScalar("##target", ImGuiDataType_U64, &configuration.end_step);
        ImGui::TextDisabled("Learning rate");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputFloat("##learning-rate", &configuration.optimizer.learning_rate, 0, 0, "%.6f");
        if (resume) ImGui::TextDisabled("Seed restored from checkpoint.");
        else {
            ImGui::TextDisabled("Seed");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputScalar("##seed", ImGuiDataType_U64, &configuration.seed);
        }
        path_field("Output folder", directory, renderer.window, true);
        ImGui::EndDisabled();
        if (status.busy && status.mode == Mode::training) {
            ImGui::BeginDisabled(stopping);
            if (ImGui::Button(stopping ? "Stopping..." : "Stop and save", {-1, 0})) {
                session.stop();
                stopping = true;
            }
            ImGui::EndDisabled();
        } else {
            ImGui::BeginDisabled(status.busy || !dataset || loading || (resume && checkpoint.empty()));
            if (ImGui::Button("Start training", {-1, 0})) {
                configuration.dataset = dataset_path;
                configuration.dataset_type = dataset_type;
                configuration.output = directory;
                configuration.device = device;
                if (picture.texture) renderer.retire(picture.texture);
                picture = {};
                preview = {};
                canvas = {};
                metrics.clear();
                stopping = false;
                target = configuration.end_step;
                session.start(TrainRequest{configuration, dataset, resume ? checkpoint : std::string{}});
                status = {.mode = Mode::training, .stage = Stage::loading, .busy = true, .started = std::chrono::steady_clock::now()};
                progress = status;
                show = true;
            }
            ImGui::EndDisabled();
        }
        if (status.busy && status.mode != Mode::training) ImGui::TextDisabled("Inference is running.");
        if (!dataset) ImGui::TextDisabled("Open a dataset first.");
        if (progress.mode != Mode::training) return show;
        ImGui::Spacing();
        ImGui::TextDisabled("%s", stopping && progress.busy ? "Stopping and saving" : stage_names[static_cast<std::size_t>(progress.stage)].data());
        ImGui::Text("%llu / %llu steps", progress.training.step, target);
        ImGui::ProgressBar(static_cast<float>(progress.training.step) / static_cast<float>(target), {-1, 3 * renderer.dpi}, "");
        if (!metrics.empty()) ImGui::Text("Loss %.4f", metrics.back().loss);
        const auto seconds = std::chrono::duration_cast<std::chrono::seconds>((progress.busy ? std::chrono::steady_clock::now() : progress.ended) - progress.started).count();
        ImGui::TextDisabled("Elapsed %lld:%02lld", seconds / 60, seconds % 60);
        if (!progress.error.empty()) ImGui::TextWrapped("%s", progress.error.c_str());
        if (metrics.empty()) return show;
        std::vector<double> steps, losses;
        for (const auto& metric : metrics) {
            steps.push_back(static_cast<double>(metric.step));
            losses.push_back(metric.loss);
        }
        if (ImPlot::BeginPlot("##loss", {-1, 145 * renderer.dpi}, ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText | ImPlotFlags_NoFrame | ImPlotFlags_NoInputs)) {
            const auto axes = ImPlotAxisFlags_NoHighlight | (metrics.size() > 1 ? ImPlotAxisFlags_AutoFit : ImPlotAxisFlags_None);
            ImPlot::SetupAxes(nullptr, nullptr, axes, axes);
            if (metrics.size() == 1) {
                ImPlot::SetupAxisLimits(ImAxis_X1, std::max(0.0, steps.front() - 100), steps.front() + 100, ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, std::max(0.01, losses.front() * 1.2), ImGuiCond_Always);
            }
            ImPlot::SetupAxisFormat(ImAxis_X1, "%.0f");
            ImPlot::SetupAxisFormat(ImAxis_Y1, "%.2f");
            ImPlot::PlotLine("Loss", steps.data(), losses.data(), static_cast<int>(steps.size()), {ImPlotProp_LineColor, ImVec4{0.55F, 0.70F, 0.63F, 1}, ImPlotProp_LineWeight, 1.5F * renderer.dpi, ImPlotProp_Marker, metrics.size() == 1 ? ImPlotMarker_Circle : ImPlotMarker_None});
            if (ImPlot::IsPlotHovered() && !metrics.empty()) {
                const auto point = ImPlot::GetPlotMousePos();
                const auto closest = std::ranges::min_element(metrics, {}, [&point](const TrainingRecord& row) { return std::abs(static_cast<double>(row.step) - point.x); });
                ImGui::SetTooltip("Step %llu\nLoss %.6f", closest->step, closest->loss);
            }
            ImPlot::EndPlot();
        }
        return show;
    }
    void TrainingPanel::draw_images() {
        if (!picture.texture) {
            ImGui::TextDisabled(progress.mode != Mode::training ? "Start training to see EMA previews." : progress.busy ? "Waiting for the first EMA preview..." : "This run did not produce an EMA preview.");
            return;
        }
        ImGui::BeginChild("training-images", {0, -ImGui::GetFrameHeightWithSpacing()});
        canvas.draw(picture);
        ImGui::EndChild();
        ImGui::TextDisabled("Training preview / EMA / step %llu", preview.training_step);
    }
} // namespace flowdit::editor
