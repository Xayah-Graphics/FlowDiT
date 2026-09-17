export module flowdit.editor.panels.training;
export import flowdit.runtime.session;
import flowdit.editor.graphics.renderer;
import flowdit.editor.viewing.canvas;
import std;
export namespace flowdit::editor {
    struct TrainingPanel final {
        RunConfiguration configuration;
        TrainingState state;
        int device{};
        std::string directory, checkpoint, error;
        std::vector<TrainingRecord> metrics;
        std::vector<SampleInfo> history;
        std::vector<std::filesystem::path> checkpoints;
        std::array<Picture, 2> previews;
        Picture batch_picture;
        std::shared_ptr<const TrainingBatch> batch;
        Canvas canvas, batch_canvas;
        bool follow{true}, smooth{true};
        float smoothing{0.15F};
        std::uint64_t selected_step{};
        void accept(Renderer& renderer, const SessionUpdate& update);
        void preview(Renderer& renderer, const FrameInfo& info, std::uint64_t texture);
        void open_run(Renderer& renderer);
        void select_preview(Renderer& renderer, std::uint64_t step);
        void draw(Renderer& renderer, Session& session, const SessionStatus& status, const std::shared_ptr<const Dataset>& dataset, const std::string& dataset_path, DatasetKind dataset_type);
    };
} // namespace flowdit::editor
