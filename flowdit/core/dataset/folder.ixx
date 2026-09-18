export module flowdit.dataset.folder;
export import flowdit.dataset.types;
import std;
export namespace flowdit {
    Dataset load_folder(const std::filesystem::path& directory, bool test = false);
}
