export module flowdit.editor.panels.dataset;
import flowdit.dataset.load;
import flowdit.editor.graphics.renderer;
export import flowdit.editor.viewing.canvas;
import std;
export namespace flowdit::editor {
    struct DatasetPanel final {
        std::string directory, loaded_directory;
        DatasetKind kind{DatasetKind::cifar10}, loaded_kind{DatasetKind::cifar10};
        std::shared_ptr<const Dataset> dataset;
        std::future<std::shared_ptr<const Dataset>> loading;
        std::string error;
        int category{-1}, page{};
        std::vector<int> counts;
        std::vector<std::uint32_t> indices;
        Picture picture;
        Canvas canvas;
        bool dirty{};
        void open();
        void receive(Renderer& renderer);
        bool draw(Renderer& renderer, bool busy);
        void draw_images(Renderer& renderer);
    };
} // namespace flowdit::editor
