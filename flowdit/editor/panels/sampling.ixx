export module flowdit.editor.panels.sampling;
export import flowdit.runtime.session;
import flowdit.editor.graphics.renderer;
import flowdit.editor.viewing.canvas;
import std;
export namespace flowdit::editor {
    struct SamplingPanel final {
        std::string checkpoint, loaded_checkpoint, directory, error;
        ModelConfiguration model;
        SamplingRequest request;
        SessionStatus progress;
        int device{}, category{-1};
        bool stopping{};
        Picture picture;
        SampleInfo result;
        Canvas canvas;
        void accept(Renderer& renderer, const SessionUpdate& update);
        void preview(Renderer& renderer, const FrameInfo& info, std::uint64_t texture);
        void open_checkpoint();
        bool draw(Renderer& renderer, Session& session, SessionStatus& status);
        void draw_images();
    };
} // namespace flowdit::editor
