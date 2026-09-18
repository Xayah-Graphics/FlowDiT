export module flowdit.serialization.digest;
import std;
export namespace flowdit::serialization {
    std::string digest(std::span<const std::byte> bytes);
    std::string digest_file(const std::filesystem::path& path);
} // namespace flowdit::serialization
