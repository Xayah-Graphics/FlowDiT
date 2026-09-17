import std;
import flowdit.headless;
#if defined(FLOWDIT_HAS_EDITOR)
import flowdit.editor;
#endif
int main(const int argc, char** argv) try {
    const std::vector<std::string_view> arguments{argv + 1, argv + argc};
    if (arguments.empty() || arguments.front() == "--gui") {
#if defined(FLOWDIT_HAS_EDITOR)
        return flowdit::editor::run(std::span{arguments}.subspan(arguments.empty() ? 0 : 1));
#else
        if (!arguments.empty()) throw std::runtime_error{"This build does not contain the Editor"};
        return flowdit::headless::run(arguments);
#endif
    }
    return flowdit::headless::run(arguments);
} catch (const std::exception& error) {
    std::println(std::cerr, "FlowDiT: {}", error.what());
    return 1;
}
