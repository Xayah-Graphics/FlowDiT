export module flowdit.editor.panels.sampling;
export import flowdit.runtime.session;
import flowdit.editor.graphics.renderer;
import flowdit.editor.viewing.canvas;
import std;
export namespace flowdit::editor {
    struct SamplingPanel final {
        struct Frame final {
            FrameInfo info;
            std::uint64_t texture{};
            SamplingResult images;
        };
        std::string checkpoint, loaded_checkpoint, directory, error;
        ModelConfiguration model;
        SamplingRequest request;
        int device{}, source{1}, category{-1}, solver{1};
        bool fid{}, follow{true};
        int frame_index{}, history_index{-1}, comparison_index{-1};
        std::vector<Frame> trajectory;
        std::vector<SampleInfo> history;
        Picture picture, comparison;
        Canvas canvas;
        void accept(Renderer& renderer, const SessionUpdate& update);
        void preview(const FrameInfo& info, std::uint64_t texture);
        void clear_trajectory(Renderer& renderer);
        void open_picture(Renderer& renderer, int index, bool compare);
        void open_checkpoint();
        void draw(Renderer& renderer, Session& session, const SessionStatus& status);
    };
} // namespace flowdit::editor
