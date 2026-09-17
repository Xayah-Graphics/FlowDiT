export module flowdit.editor.panels.sampling;
export import flowdit.runtime.session;
import flowdit.runtime.catalog;
import flowdit.editor.graphics.renderer;
import flowdit.editor.viewing.canvas;
import std;
export namespace flowdit::editor {
    struct SamplingPanel final {
        std::string run, error;
        CheckpointEntry checkpoint;
        SamplingRequest request;
        SessionStatus progress;
        int device{}, category{-1};
        bool stopping{}, attached{};
        Picture picture;
        SampleInfo result;
        Canvas canvas;
        void accept(Renderer& renderer, const SessionUpdate& update);
        void preview(Renderer& renderer, const FrameInfo& info, std::uint64_t texture);
        void select(const DatasetEntry& dataset, std::string name = {}, std::size_t index = 0);
        bool draw(Renderer& renderer, Session& session, SessionStatus& status, const Catalog& catalog, const DatasetEntry& dataset);
        void draw_images();
    };
} // namespace flowdit::editor
