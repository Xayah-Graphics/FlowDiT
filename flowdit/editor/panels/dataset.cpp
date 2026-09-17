module;
#include <imgui.h>
module flowdit.editor.panels.dataset;
import flowdit.editor.widgets.controls;
import std;
namespace flowdit::editor {
    void DatasetPanel::open(Renderer& renderer, const DatasetEntry& entry) {
        if (picture.texture) renderer.retire(picture.texture);
        picture = {};
        canvas = {};
        dataset.reset();
        counts.clear();
        indices.clear();
        category = -1;
        page = 0;
        dirty = filter_dirty = false;
        loading = std::async(std::launch::async, [path = entry.directory] { return std::make_shared<const Dataset>(load_dataset(path)); });
        error.clear();
    }
    void DatasetPanel::receive(Renderer& renderer) {
        if (loading.valid() && loading.wait_for(std::chrono::seconds{0}) == std::future_status::ready) {
            try {
                dataset = loading.get();
                counts.assign(dataset->specification.classes.size(), 0);
                for (const auto label : dataset->labels) ++counts[label];
                category = -1;
                page = 0;
                canvas = {};
                dirty = filter_dirty = true;
            } catch (const std::exception& failure) {
                error = failure.what();
            }
        }
        if (!dirty || !dataset) return;
        if (filter_dirty) {
            indices.clear();
            for (std::uint32_t i = 0; i < dataset->labels.size(); ++i)
                if (category < 0 || dataset->labels[i] == category) indices.push_back(i);
            filter_dirty = false;
        }
        const std::size_t begin = static_cast<std::size_t>(page) * 24;
        const auto count = std::min(24uz, indices.size() - begin);
        const auto& specification = dataset->specification;
        const std::size_t pixels = static_cast<std::size_t>(specification.width) * specification.height;
        ImageBatch batch;
        dataset->read(std::span{indices}.subspan(begin, count), batch);
        std::vector<std::uint8_t> rgba(count * pixels * 4uz);
        for (std::size_t image = 0; image < count; ++image) {
            for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    const std::size_t source_channel = specification.channels == 1u ? 0uz : channel;
                    rgba[(image * pixels + pixel) * 4uz + channel] = batch.pixels[image * pixels * specification.channels + source_channel * pixels + pixel];
                }
                rgba[(image * pixels + pixel) * 4uz + 3uz] = 255;
            }
        }
        picture.upload(renderer, specification, batch.labels, rgba.data());
        dirty = false;
    }
    bool DatasetPanel::draw_browse() {
        bool show{};
        ImGui::BeginDisabled(!dataset);
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("Browse class");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##browse-class", category < 0 ? "All classes" : dataset->specification.classes[category].c_str())) {
            if (ImGui::Selectable("All classes", category < 0)) {
                category = -1;
                show = true;
            }
            for (int i = 0; i < static_cast<int>(counts.size()); ++i) {
                const auto label = std::format("{} · {}", dataset->specification.classes[i], counts[i]);
                if (ImGui::Selectable(label.c_str(), category == i, counts[i] ? ImGuiSelectableFlags_None : ImGuiSelectableFlags_Disabled)) {
                    category = i;
                    show = true;
                }
            }
            ImGui::EndCombo();
        }
        ImGui::EndDisabled();
        if (show) {
            page = 0;
            canvas = {};
            dirty = filter_dirty = true;
        }
        return show;
    }
    bool DatasetPanel::draw(const Catalog& catalog, std::string& selected, const bool busy) {
        bool show{};
        const auto previous = selected;
        const float dpi = ImGui::GetStyle().FontScaleDpi;
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 4 * dpi});
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{6 * dpi, 4 * dpi});
        ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2{0, 0.5F});
        for (const auto& [key, entry] : catalog.datasets) {
            ImGui::PushID(key.c_str());
            const auto origin = ImGui::GetCursorScreenPos();
            const float width = ImGui::GetContentRegionAvail().x;
            const bool available = entry.info.has_value() && entry.error.empty();
            ImGui::BeginDisabled(!available || ((busy || loading.valid()) && key != previous));
            if (ImGui::Selectable(entry.info ? entry.info->specification.name.c_str() : key.c_str(), selected == key, ImGuiSelectableFlags_None, {0, ImGui::GetFrameHeight()})) {
                selected = key;
                page = 0;
                canvas = {};
                dirty = previous == key;
                show = true;
            }
            const auto detail = !entry.error.empty() ? std::string{"Error"} : entry.info ? std::format("{}", entry.info->count) : "Unsupported";
            const float text_width = ImGui::CalcTextSize(detail.c_str()).x;
            ImGui::GetWindowDrawList()->AddText({origin.x + width - text_width - 6 * dpi, origin.y + 4 * dpi}, ImGui::GetColorU32(ImGuiCol_TextDisabled), detail.c_str());
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s%s%s", key.c_str(), entry.error.empty() ? "" : "\n", entry.error.c_str());
            ImGui::EndDisabled();
            ImGui::PopID();
        }
        ImGui::PopStyleVar(3);
        if (catalog.datasets.empty()) ImGui::TextWrapped("No datasets in %s", Catalog::directory.string().c_str());
        return show;
    }
    void DatasetPanel::draw_images() {
        if (!picture.texture) {
            ImGui::TextDisabled(loading.valid() ? "Loading dataset..." : "Select a dataset to browse images.");
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
        const float dpi = ImGui::GetStyle().FontScaleDpi;
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{12 * dpi, (40 * dpi - ImGui::GetFontSize()) * 0.5F});
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("%s / %s", picture.specification.name.c_str(), category < 0 ? "All classes" : picture.specification.classes[category].c_str());
        const auto pages = std::format("{} / {}", page + 1, (indices.size() + 23) / 24);
        const float pagination_width = ImGui::CalcTextSize(pages.c_str()).x + ImGui::CalcTextSize("‹›").x + 48 * ImGui::GetStyle().FontScaleDpi + 3 * ImGui::GetStyle().ItemSpacing.x;
        ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - pagination_width);
        ImGui::BeginDisabled(page == 0);
        if (text_button("‹##previous")) {
            --page;
            canvas = {};
            dirty = true;
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextDisabled("%s", pages.c_str());
        ImGui::SameLine();
        ImGui::BeginDisabled(static_cast<std::size_t>(page + 1) * 24 >= indices.size());
        if (text_button("›##next")) {
            ++page;
            canvas = {};
            dirty = true;
        }
        ImGui::EndDisabled();
        ImGui::PopStyleVar();
    }
} // namespace flowdit::editor
