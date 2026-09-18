module;
#include <flowdit/cuda.h>
#include <imgui.h>
module flowdit.editor.panels.training;
import flowdit.editor.widgets.controls;
import std;
namespace flowdit::editor {
    void TrainingPanel::select(Renderer& renderer, const Catalog& catalog, const DatasetEntry& dataset, std::string name) {
        if (picture.texture) renderer.retire(picture.texture);
        picture  = {};
        preview  = {};
        canvas   = {};
        progress = {};
        metrics.clear();
        error.clear();
        stopping      = false;
        attached      = false;
        elapsed_base  = 0;
        first_step    = 0;
        run           = std::move(name);
        configuration = catalog.training(dataset);
        if (run.empty()) return;
        try {
            const auto& entry = dataset.runs.at(run);
            if (!entry.error.empty()) throw std::runtime_error{entry.error};
            configuration      = entry.configuration;
            const auto history = output::read_history(configuration.output);
            metrics            = history.metrics;
            progress.mode      = Mode::training;
            progress.stage     = Stage::stopped;
            if (!entry.checkpoints.empty()) {
                const auto& checkpoint = entry.checkpoints.front();
                if (!checkpoint.error.empty()) throw std::runtime_error{checkpoint.error};
                configuration.seed     = checkpoint.seed;
                progress.training.step = checkpoint.step;
                progress.training.seed = checkpoint.seed;
                elapsed_base           = checkpoint.training_seconds;
                if (checkpoint.step >= configuration.end_step) progress.stage = Stage::complete;
            }
            if (!history.preview.empty()) {
                const auto loaded = output::read_sample(history.preview);
                preview           = loaded.info;
                picture.upload(renderer, loaded.info.image, loaded.info.labels, loaded.rgba.data());
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
            if ((!sample->info.training_step && !sample->info.reconstruction) || sample->info.source != ParameterSource::exponential_average) continue;
            preview = sample->info;
            picture.upload(renderer, sample->info.image, sample->info.labels, sample->rgba.data());
        }
    }
    bool TrainingPanel::draw(Renderer& renderer, Session& session, SessionStatus& status, const Catalog& catalog, const DatasetEntry& entry, const std::shared_ptr<const Dataset>& dataset) {
        bool show{};
        const bool running = status.busy && status.mode == Mode::training;
        ImGui::BeginDisabled(status.busy);
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("Stage");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##stage", configuration.stage == TrainingStage::autoencoder ? "Autoencoder" : "FlowDiT")) {
            for (const auto stage : {TrainingStage::autoencoder, TrainingStage::flowdit})
                if (ImGui::Selectable(stage == TrainingStage::autoencoder ? "Autoencoder" : "FlowDiT", configuration.stage == stage)) {
                    select(renderer, catalog, entry);
                    configuration = catalog.training(entry, stage);
                }
            ImGui::EndCombo();
        }
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("Run");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##run", run.empty() ? "New training" : run_label(run).c_str())) {
            if (ImGui::Selectable("New training", run.empty())) {
                const auto stage = configuration.stage;
                select(renderer, catalog, entry);
                configuration = catalog.training(entry, stage);
            }
            for (const auto& [name, history] : entry.runs) {
                if (history.configuration.stage != configuration.stage) continue;
                auto checkpoints  = history.checkpoints | std::views::filter([](const CheckpointEntry& checkpoint) { return checkpoint.error.empty(); });
                const auto count  = std::ranges::distance(checkpoints);
                const auto step   = count ? checkpoints.front().step : 0;
                const auto detail = history.error.empty() ? std::format(" · Step {} / {} · {} checkpoint{}", step, history.configuration.end_step, count, count == 1 ? "" : "s") : " / Error";
                const auto label  = run_label(name) + detail + "##" + name;
                if (ImGui::Selectable(label.c_str(), name == run)) select(renderer, catalog, entry, name);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", name.c_str());
            }
            ImGui::EndCombo();
        }
        if (configuration.stage == TrainingStage::flowdit) {
            ImGui::BeginDisabled(!run.empty());
            ImGui::AlignTextToFramePadding();
            ImGui::TextDisabled("Autoencoder");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-1);
            const auto selected = configuration.autoencoder_checkpoint.empty() ? std::string{"Select checkpoint"} : checkpoint_label(configuration.autoencoder_checkpoint);
            if (ImGui::BeginCombo("##autoencoder", selected.c_str())) {
                for (const auto& history : entry.runs | std::views::values) {
                    if (!history.error.empty() || history.configuration.stage != TrainingStage::autoencoder) continue;
                    for (const auto& checkpoint : history.checkpoints) {
                        if (!checkpoint.error.empty()) continue;
                        const auto path = std::filesystem::relative(checkpoint.path, entry.directory);
                        if (ImGui::Selectable((checkpoint_label(path) + "##" + path.generic_string()).c_str(), path == configuration.autoencoder_checkpoint)) configuration.autoencoder_checkpoint = path;
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::EndDisabled();
        }
        ImGui::EndDisabled();
        ImGui::Spacing();
        auto state = running && stopping ? std::string_view{"Stopping and saving"} : stage_names[static_cast<std::size_t>(progress.stage)];
        if (progress.stage == Stage::complete) state = progress.training.step >= configuration.end_step ? "Target reached" : "Stopped";
        if (run.empty()) ImGui::TextDisabled("Ready to train");
        else ImGui::TextDisabled("%.*s / step %llu", static_cast<int>(state.size()), state.data(), progress.training.step);
        if (running && progress.training_started != std::chrono::steady_clock::time_point{}) {
            if (progress.training.step > first_step) {
                const double steps = static_cast<double>(progress.training.step - first_step);
                const double rate  = steps / (progress.training.elapsed_seconds - elapsed_base);
                ImGui::TextDisabled("%.2f step/s", rate);
                if (!stopping && progress.training.step < configuration.end_step) {
                    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - progress.training_started).count();
                    const auto remaining = static_cast<std::uint64_t>(std::ceil((configuration.end_step - progress.training.step) * elapsed / steps));
                    ImGui::SameLine();
                    ImGui::TextDisabled("· ETA ~%llu:%02llu:%02llu", remaining / 3600, remaining / 60 % 60, remaining % 60);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Based on this session, including observed preview and checkpoint time.\nExcludes initialization and latent cache preparation.");
                }
            } else ImGui::TextDisabled("Measuring speed...");
        }
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
            ImGui::BeginDisabled(status.busy || !dataset || !error.empty() || (configuration.stage == TrainingStage::flowdit && configuration.autoencoder_checkpoint.empty()) || (!run.empty() && !resumable) || (!run.empty() && configuration.end_step <= progress.training.step));
            if (ImGui::Button(run.empty() ? "Start training" : "Continue training", {-1, 0})) {
                std::filesystem::path checkpoint;
                first_step   = 0;
                elapsed_base = 0;
                if (run.empty()) {
                    configuration.output = catalog.training(entry, configuration.stage).output;
                    run                  = configuration.output.filename().string();
                } else {
                    const auto& latest = entry.runs.at(run).checkpoints.front();
                    checkpoint         = latest.path;
                    elapsed_base       = latest.training_seconds;
                    first_step         = latest.step;
                }
                stopping = false;
                attached = true;
                session.start(TrainRequest{configuration, dataset, checkpoint});
                status   = {.mode = Mode::training, .stage = Stage::loading, .busy = true, .started = std::chrono::steady_clock::now()};
                progress = status;
                show     = true;
            }
            ImGui::EndDisabled();
        }
        if (!metrics.empty()) {
            ImGui::Spacing();
            ImGui::TextDisabled(configuration.stage == TrainingStage::autoencoder ? "Reconstruction / %.4f" : "Flow matching / %.4f", metrics.back().loss);
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4{});
            ImGui::PlotLines("##loss", [](void* data, const int index) { return static_cast<float>(static_cast<const TrainingRecord*>(data)[index].loss); }, metrics.data(), static_cast<int>(metrics.size()), 0, nullptr, std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), {-1, 60 * renderer.dpi});
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) {
                const float padding  = ImGui::GetStyle().FramePadding.x;
                const float position = (ImGui::GetIO().MousePos.x - ImGui::GetItemRectMin().x - padding) / (ImGui::GetItemRectSize().x - padding * 2);
                const auto index     = static_cast<std::size_t>(std::clamp(position, 0.0F, 1.0F) * (metrics.size() - 1));
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
            number_field("Batch", ImGuiDataType_U32, &configuration.batch, nullptr, true);
            if (configuration.stage == TrainingStage::autoencoder) number_field("Accumulate", ImGuiDataType_U32, &configuration.autoencoder.accumulation, nullptr, true);
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
            ImGui::TextDisabled(progress.mode != Mode::training ? "Start training to see previews." : progress.busy ? "Waiting for the first preview..." : "This run did not produce a preview.");
            return;
        }
        ImGui::BeginChild("training-images", {0, -ImGui::GetFrameHeightWithSpacing()});
        canvas.draw(picture);
        ImGui::EndChild();
        ImGui::TextDisabled(preview.reconstruction ? "Original / Reconstruction · Step %llu" : "EMA preview / Step %llu", preview.training_step);
    }
} // namespace flowdit::editor
