module;
#include <imgui.h>
module flowdit.editor.panels.dataset;
import flowdit.editor.widgets.controls;
import std;
namespace flowdit::editor {
    void DatasetPanel::open() {
        const std::filesystem::path path{directory};
        loading = std::async(std::launch::async, [path, type = kind] { return std::make_shared<const Dataset>(load_dataset(type, path)); });
        error.clear();
    }
    void DatasetPanel::receive(Renderer& renderer) {
        if (loading.valid() && loading.wait_for(std::chrono::seconds{0}) == std::future_status::ready) {
            try {
                dataset = loading.get();
                loaded_directory = directory;
                loaded_kind = kind;
                counts.assign(dataset->specification.classes.size(), 0);
                for (const auto label : dataset->labels) ++counts[label];
                category = -1;
                page = 0;
                canvas = {};
                dirty = true;
            } catch (const std::exception& failure) {
                error = failure.what();
            }
        }
        if (!dirty || !dataset) return;
        indices.clear();
        for (std::uint32_t i = 0; i < dataset->labels.size(); ++i)
            if (category < 0 || dataset->labels[i] == category) indices.push_back(i);
        const std::size_t begin = static_cast<std::size_t>(page) * 24;
        const auto count = std::min(24uz, indices.size() - begin);
        const auto& specification = dataset->specification;
        const std::size_t pixels = static_cast<std::size_t>(specification.width) * specification.height;
        std::vector<std::uint32_t> labels(count);
        std::vector<std::uint8_t> rgba(count * pixels * 4uz);
        for (std::size_t image = 0; image < count; ++image) {
            const auto index = indices[begin + image];
            labels[image] = dataset->labels[index];
            for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    const std::size_t source_channel = specification.channels == 1u ? 0uz : channel;
                    rgba[(image * pixels + pixel) * 4uz + channel] = dataset->images[index * pixels * specification.channels + source_channel * pixels + pixel];
                }
                rgba[(image * pixels + pixel) * 4uz + 3uz] = 255;
            }
        }
        picture.upload(renderer, specification, labels, rgba.data());
        dirty = false;
    }
    bool DatasetPanel::draw(Renderer& renderer, const bool busy) {
        bool show{};
        ImGui::TextUnformatted(dataset ? dataset->specification.name.c_str() : "No dataset");
        ImGui::SameLine();
        ImGui::BeginDisabled(!dataset);
        if (ImGui::SmallButton("View images")) show = true;
        ImGui::EndDisabled();
        if (dataset) {
            const auto& image = dataset->specification;
            ImGui::TextDisabled("%zu images / %zu classes", dataset->labels.size(), image.classes.size());
            ImGui::TextDisabled("%u x %u / %s", image.width, image.height, image.channels == 1u ? "Grayscale" : "RGB");
        }
        ImGui::BeginDisabled(busy || loading.valid());
        if (ImGui::Button("Open dataset...")) ImGui::OpenPopup("Open dataset");
        ImGui::EndDisabled();
        if (dataset && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", loaded_directory.c_str());
        ImGui::SetNextWindowSize({440 * renderer.dpi, 0}, ImGuiCond_Appearing);
        if (ImGui::BeginPopupModal("Open dataset", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings)) {
            ImGui::TextDisabled("Format");
            int selected_kind = static_cast<int>(kind);
            ImGui::SetNextItemWidth(-1);
            if (ImGui::Combo("##format", &selected_kind, "CIFAR-10\0MNIST\0")) kind = static_cast<DatasetKind>(selected_kind);
            path_field("Dataset folder", directory, renderer.window, true);
            if (ImGui::Button("Open")) {
                open();
                show = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
        if (loading.valid()) ImGui::TextDisabled("Loading images...");
        if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
        if (!dataset) return show;
        ImGui::TextDisabled("Browse class");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##class", category < 0 ? "All classes" : dataset->specification.classes[category].c_str())) {
            if (ImGui::Selectable("All classes", category < 0)) {
                category = -1;
                dirty = true;
            }
            for (int i = 0; i < static_cast<int>(counts.size()); ++i) {
                const auto label = std::format("{} ({})", dataset->specification.classes[i], counts[i]);
                if (ImGui::Selectable(label.c_str(), category == i, counts[i] ? ImGuiSelectableFlags_None : ImGuiSelectableFlags_Disabled)) {
                    category = i;
                    dirty = true;
                }
            }
            ImGui::EndCombo();
        }
        if (dirty) {
            page = 0;
            canvas = {};
            show = true;
        }
        return show;
    }
    void DatasetPanel::draw_images(Renderer& renderer) {
        receive(renderer);
        if (!picture.texture) {
            ImGui::TextDisabled(loading.valid() ? "Loading dataset..." : "Open a dataset to browse images.");
            return;
        }
        ImGui::BeginChild("dataset-images", {0, -ImGui::GetFrameHeightWithSpacing()});
        canvas.draw(picture);
        ImGui::EndChild();
        if (canvas.selected >= 0) {
            const auto index = indices[static_cast<std::size_t>(page) * 24 + canvas.selected];
            ImGui::TextDisabled("%s / #%u / %s / %u x %u", picture.specification.name.c_str(), index, picture.specification.classes[picture.labels[canvas.selected]].c_str(), picture.specification.width, picture.specification.height);
            return;
        }
        ImGui::BeginDisabled(page == 0);
        if (ImGui::SmallButton("Previous")) {
            --page;
            canvas = {};
            dirty = true;
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextDisabled("%s / %d of %zu", picture.specification.name.c_str(), page + 1, (indices.size() + 23) / 24);
        ImGui::SameLine();
        ImGui::BeginDisabled(static_cast<std::size_t>(page + 1) * 24 >= indices.size());
        if (ImGui::SmallButton("Next")) {
            ++page;
            canvas = {};
            dirty = true;
        }
        ImGui::EndDisabled();
    }
} // namespace flowdit::editor
