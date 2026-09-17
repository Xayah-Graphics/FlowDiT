module;
#include <flowdit/cuda.h>
#include <imgui.h>
module flowdit.editor.panels.training;
import flowdit.editor.widgets.controls;
import std;
namespace flowdit::editor {
    void TrainingPanel::select(Renderer& renderer, const Catalog& catalog, const DatasetEntry& dataset, std::string name) {
        if (picture.texture) renderer.retire(picture.texture);
        picture = {};
        preview = {};
        canvas = {};
        progress = {};
        metrics.clear();
        error.clear();
        stopping = false;
        attached = false;
        elapsed_base = 0;
        run = std::move(name);
        configuration = catalog.training(dataset);
        if (run.empty()) return;
        try {
            const auto& entry = dataset.runs.at(run);
            if (!entry.error.empty()) throw std::runtime_error{entry.error};
            configuration = entry.configuration;
            const auto history = output::read_history(configuration.output);
            metrics = history.metrics;
            progress.mode = Mode::training;
            progress.stage = Stage::stopped;
            if (!entry.checkpoints.empty()) {
                const auto& checkpoint = entry.checkpoints.front();
                if (!checkpoint.error.empty()) throw std::runtime_error{checkpoint.error};
                configuration.seed = checkpoint.seed;
                progress.training.step = checkpoint.step;
                progress.training.seed = checkpoint.seed;
                elapsed_base = checkpoint.training_seconds;
                if (checkpoint.step >= configuration.end_step) progress.stage = Stage::complete;
            }
            if (!history.preview.empty()) {
                const auto loaded = output::read_sample(history.preview);
                preview = loaded.info;
                picture.upload(renderer, loaded.images.model.image, loaded.images.labels, loaded.images.rgba.data());
            }
        } catch (const std::exception& failure) {
            error = failure.what();
        }
    }
    void TrainingPanel::accept(Renderer& renderer, const SessionUpdate& update) {
        if (!attached || update.status.mode != Mode::training) return;
        progress = update.status;
        metrics.insert(metrics.end(), update.metrics.begin(), update.metrics.end());
        for (const auto& sample : update.samples) {
            if (!sample->info.training_step || sample->info.source != ParameterSource::exponential_average) continue;
            preview = sample->info;
            picture.upload(renderer, sample->images.model.image, sample->images.labels, sample->images.rgba.data());
        }
    }
    bool TrainingPanel::draw(Renderer& renderer, Session& session, SessionStatus& status, const Catalog& catalog, const DatasetEntry& entry, const std::shared_ptr<const Dataset>& dataset) {
        bool show{};
        const bool running = status.busy && status.mode == Mode::training;
        ImGui::BeginDisabled(status.busy);
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("Run");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##run", run.empty() ? "New training" : run_label(run).c_str())) {
            if (ImGui::Selectable("New training", run.empty())) select(renderer, catalog, entry);
            for (const auto& [name, history] : entry.runs) {
                const auto label = run_label(name) + (history.error.empty() ? "" : " / Error") + "##" + name;
                if (ImGui::Selectable(label.c_str(), name == run)) select(renderer, catalog, entry, name);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", name.c_str());
            }
            ImGui::EndCombo();
        }
        ImGui::EndDisabled();
        ImGui::Spacing();
        const auto state = running && stopping ? std::string_view{"Stopping and saving"} : stage_names[static_cast<std::size_t>(progress.stage)];
        if (run.empty()) ImGui::TextDisabled("Ready to train");
        else ImGui::TextDisabled("%.*s / step %llu", static_cast<int>(state.size()), state.data(), progress.training.step);
        ImGui::BeginDisabled(status.busy);
        number_field("Target steps", ImGuiDataType_U64, &configuration.end_step);
        ImGui::EndDisabled();
        ImGui::Spacing();
        if (running) {
            ImGui::BeginDisabled(stopping);
            if (ImGui::Button(stopping ? "Stopping..." : "Stop and save", {-1, 0})) {
                session.stop();
                stopping = true;
            }
            ImGui::EndDisabled();
        } else {
            const bool resumable = !run.empty() && entry.runs.contains(run) && !entry.runs.at(run).checkpoints.empty();
            ImGui::BeginDisabled(status.busy || !dataset || !error.empty() || (!run.empty() && !resumable) || (!run.empty() && configuration.end_step <= progress.training.step));
            if (ImGui::Button(run.empty() ? "Start training" : "Continue training", {-1, 0})) {
                std::filesystem::path checkpoint;
                if (run.empty()) {
                    configuration.output = catalog.training(entry).output;
                    run = configuration.output.filename().string();
                } else {
                    const auto& latest = entry.runs.at(run).checkpoints.front();
                    checkpoint = latest.path;
                    elapsed_base = latest.training_seconds;
                }
                stopping = false;
                attached = true;
                session.start(TrainRequest{configuration, dataset, checkpoint});
                status = {.mode = Mode::training, .stage = Stage::loading, .busy = true, .started = std::chrono::steady_clock::now()};
                progress = status;
                show = true;
            }
            ImGui::EndDisabled();
        }
        if (!metrics.empty()) {
            ImGui::Spacing();
            ImGui::TextDisabled("Training loss / %.4f", metrics.back().loss);
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4{});
            ImGui::PlotLines("##loss", [](void* data, const int index) { return static_cast<float>(static_cast<const TrainingRecord*>(data)[index].loss); }, metrics.data(), static_cast<int>(metrics.size()), 0, nullptr, std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), {-1, 60 * renderer.dpi});
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) {
                const float padding = ImGui::GetStyle().FramePadding.x;
                const float position = (ImGui::GetIO().MousePos.x - ImGui::GetItemRectMin().x - padding) / (ImGui::GetItemRectSize().x - padding * 2);
                const auto index = static_cast<std::size_t>(std::clamp(position, 0.0F, 1.0F) * (metrics.size() - 1));
                ImGui::SetTooltip("Step %llu\nLoss %.6f", metrics[index].step, metrics[index].loss);
            }
        }
        if (picture.texture) {
            ImGui::SetCursorPosX(ImGui::GetWindowContentRegionMax().x - ImGui::CalcTextSize("Preview").x - 24 * renderer.dpi);
            if (text_button("Preview")) show = true;
        }
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4{});
        if (ImGui::CollapsingHeader("Parameters")) {
            ImGui::BeginDisabled(status.busy);
            number_field("Learning rate", ImGuiDataType_Float, &configuration.optimizer.learning_rate, "%.4g", true);
            ImGui::BeginDisabled(!run.empty());
            number_field("Seed", ImGuiDataType_U64, &configuration.seed, nullptr, true);
            if (!run.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Restored from checkpoint");
            ImGui::EndDisabled();
            ImGui::EndDisabled();
        }
        ImGui::PopStyleColor();
        if (status.busy && status.mode != Mode::training) ImGui::TextDisabled("Available after inference");
        if (!dataset) ImGui::TextDisabled("Loading dataset...");
        if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
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
        ImGui::TextDisabled("EMA preview / Step %llu", preview.training_step);
    }
} // namespace flowdit::editor
