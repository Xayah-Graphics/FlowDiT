module;
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>
module flowdit.editor.panels.dataset;
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
                dataset          = loading.get();
                loaded_directory = directory;
                loaded_kind      = kind;
                counts.assign(dataset->specification.classes.size(), 0);
                for (const auto label : dataset->labels) ++counts[label];
                category        = -1;
                page            = 0;
                canvas.selected = -1;
                dirty           = true;
            } catch (const std::exception& failure) {
                error = failure.what();
            }
        }
        if (!dirty || !dataset) return;
        indices.clear();
        for (std::uint32_t i = 0; i < dataset->labels.size(); ++i)
            if (category < 0 || dataset->labels[i] == category) indices.push_back(i);
        page                      = std::clamp(page, 0, static_cast<int>((indices.size() - 1) / 100));
        const std::size_t begin   = static_cast<std::size_t>(page) * 100;
        const auto count          = std::min(100uz, indices.size() - begin);
        const auto& specification = dataset->specification;
        const std::size_t pixels  = static_cast<std::size_t>(specification.width) * specification.height;
        picture.images            = {.model = {specification}, .nfe = 0u, .labels = std::vector<std::uint32_t>(count), .rgba = std::vector<std::uint8_t>(count * pixels * 4uz)};
        for (std::size_t image = 0; image < count; ++image) {
            const auto index             = indices[begin + image];
            picture.images.labels[image] = dataset->labels[index];
            for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    const std::size_t source_channel                              = specification.channels == 1u ? 0uz : channel;
                    picture.images.rgba[(image * pixels + pixel) * 4uz + channel] = dataset->images[index * pixels * specification.channels + source_channel * pixels + pixel];
                }
                picture.images.rgba[(image * pixels + pixel) * 4uz + 3uz] = 255;
            }
        }
        if (picture.texture) renderer.retire(picture.texture);
        picture.texture = renderer.texture({specification.width, static_cast<std::uint32_t>(count) * specification.height});
        renderer.upload(picture.texture, picture.images.rgba.data(), specification.width, static_cast<std::uint32_t>(count) * specification.height, true);
        dirty = false;
    }
    void DatasetPanel::draw(Renderer& renderer, const bool busy) {
        ImGui::BeginDisabled(busy || loading.valid());
        ImGui::SetNextItemWidth(140);
        int selected_kind = static_cast<int>(kind);
        if (ImGui::Combo("Format", &selected_kind, "CIFAR-10\0MNIST\0")) kind = static_cast<DatasetKind>(selected_kind);
        ImGui::SetNextItemWidth(-150);
        ImGui::InputText("##directory", &directory);
        ImGui::SameLine();
        if (ImGui::Button("Open dataset")) open();
        ImGui::EndDisabled();
        if (loading.valid()) ImGui::TextDisabled("Reading training images...");
        if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
        if (!dataset) return;
        const auto& specification = dataset->specification;
        ImGui::SetNextItemWidth(180);
        if (ImGui::BeginCombo("Class", category < 0 ? "All classes" : specification.classes[category].c_str())) {
            if (ImGui::Selectable("All classes", category < 0)) {
                category        = -1;
                page            = 0;
                dirty           = true;
                canvas.selected = -1;
            }
            for (int i = 0; i < static_cast<int>(counts.size()); ++i) {
                const auto label = std::format("{} ({})", specification.classes[i], counts[i]);
                if (ImGui::Selectable(label.c_str(), category == i, counts[i] ? ImGuiSelectableFlags_None : ImGuiSelectableFlags_Disabled)) {
                    category        = i;
                    page            = 0;
                    dirty           = true;
                    canvas.selected = -1;
                }
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        ImGui::Text("%s | %zu images | %u x %u | %u channel(s)", specification.name.c_str(), indices.size(), specification.width, specification.height, specification.channels);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110);
        ImGui::InputInt("Index", &jump, 0, 0);
        ImGui::SameLine();
        if (ImGui::Button("Go")) {
            jump            = std::clamp(jump, 0, static_cast<int>(dataset->labels.size()) - 1);
            category        = -1;
            page            = jump / 100;
            canvas.selected = jump % 100;
            dirty           = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Random")) {
            static std::mt19937 random{std::random_device{}()};
            const auto selected = std::uniform_int_distribution<std::size_t>{0, indices.size() - 1}(random);
            page                = static_cast<int>(selected / 100);
            canvas.selected     = static_cast<int>(selected % 100);
            dirty               = true;
        }
        if (ImGui::Button("Previous") && page > 0) {
            --page;
            canvas.selected = -1;
            dirty           = true;
        }
        ImGui::SameLine();
        ImGui::Text("Page %d / %zu", page + 1, (indices.size() + 99) / 100);
        ImGui::SameLine();
        if (ImGui::Button("Next") && static_cast<std::size_t>(page + 1) * 100 < indices.size()) {
            ++page;
            canvas.selected = -1;
            dirty           = true;
        }
        receive(renderer);
        if (canvas.selected >= 0) {
            ImGui::SameLine();
            ImGui::Text("Dataset index: %u", indices[static_cast<std::size_t>(page) * 100 + canvas.selected]);
        }
        canvas.draw(picture.texture, picture.images);
    }
} // namespace flowdit::editor
