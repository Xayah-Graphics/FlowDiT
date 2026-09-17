export module flowdit.editor.panels.dataset;
import flowdit.runtime.catalog;
import flowdit.editor.graphics.renderer;
export import flowdit.editor.viewing.canvas;
import std;
export namespace flowdit::editor {
    struct DatasetPanel final {
        std::shared_ptr<const Dataset> dataset;
        std::future<std::shared_ptr<const Dataset>> loading;
        std::string error;
        int category{-1}, page{};
        std::vector<int> counts;
        std::vector<std::uint32_t> indices;
        Picture picture;
        Canvas canvas;
        bool dirty{}, filter_dirty{};
        void open(Renderer& renderer, const DatasetEntry& entry);
        void receive(Renderer& renderer);
        bool draw_browse();
        bool draw(const Catalog& catalog, std::string& selected, bool busy);
        void draw_images();
    };
} // namespace flowdit::editor
