export module flowdit.editor.panels.training;
export import flowdit.runtime.session;
import flowdit.runtime.catalog;
import flowdit.editor.graphics.renderer;
import flowdit.editor.viewing.canvas;
import std;
export namespace flowdit::editor {
    struct TrainingPanel final {
        RunConfiguration configuration;
        SessionStatus progress;
        int device{};
        std::string run, error;
        std::vector<TrainingRecord> metrics;
        Picture picture;
        SampleInfo preview;
        Canvas canvas;
        double elapsed_base{};
        bool stopping{}, attached{};
        void select(Renderer& renderer, const Catalog& catalog, const DatasetEntry& dataset, std::string name = {});
        void accept(Renderer& renderer, const SessionUpdate& update);
        bool draw(Renderer& renderer, Session& session, SessionStatus& status, const Catalog& catalog, const DatasetEntry& entry, const std::shared_ptr<const Dataset>& dataset);
        void draw_images();
    };
} // namespace flowdit::editor
