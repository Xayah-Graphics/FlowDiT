export module flowdit.editor.panels.training;
export import flowdit.runtime.session;
import flowdit.editor.graphics.renderer;
import flowdit.editor.viewing.canvas;
import std;
export namespace flowdit::editor {
    struct TrainingPanel final {
        RunConfiguration configuration;
        SessionStatus progress;
        int device{}, resume{};
        std::string directory, checkpoint;
        std::vector<TrainingRecord> metrics;
        Picture picture;
        SampleInfo preview;
        Canvas canvas;
        std::uint64_t target{};
        bool stopping{};
        void accept(Renderer& renderer, const SessionUpdate& update);
        bool draw(Renderer& renderer, Session& session, SessionStatus& status, const std::shared_ptr<const Dataset>& dataset, const std::string& dataset_path, DatasetKind dataset_type, bool loading);
        void draw_images();
    };
} // namespace flowdit::editor
