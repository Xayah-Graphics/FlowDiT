module flowdit.serialization.digest;
import std;
namespace flowdit::serialization {
    namespace {
        struct Sha256 final {
            std::array<std::uint32_t, 8> state{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
            std::array<std::byte, 64> buffer{};
            std::uint64_t size{};
            void block(const std::byte* input) {
                constexpr std::array<std::uint32_t, 64> constants{0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
                std::array<std::uint32_t, 64> w{};
                for (unsigned i = 0; i < 16; ++i) {
                    std::memcpy(&w[i], input + i * 4, 4);
                    if constexpr (std::endian::native == std::endian::little) w[i] = std::byteswap(w[i]);
                }
                for (unsigned i = 16; i < 64; ++i) w[i] = w[i - 16] + (std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] + (std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10));
                auto v = state;
                for (unsigned i = 0; i < 64; ++i) {
                    const auto t1 = v[7] + (std::rotr(v[4], 6) ^ std::rotr(v[4], 11) ^ std::rotr(v[4], 25)) + ((v[4] & v[5]) ^ (~v[4] & v[6])) + constants[i] + w[i];
                    const auto t2 = (std::rotr(v[0], 2) ^ std::rotr(v[0], 13) ^ std::rotr(v[0], 22)) + ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
                    v             = {t1 + t2, v[0], v[1], v[2], v[3] + t1, v[4], v[5], v[6]};
                }
                for (unsigned i = 0; i < 8; ++i) state[i] += v[i];
            }
            void update(std::span<const std::byte> data) {
                for (const auto byte : data) {
                    buffer[size++ % 64] = byte;
                    if (size % 64 == 0) block(buffer.data());
                }
            }
            std::string finish() {
                const auto bits = size * 8;
                const std::array<std::byte, 1> marker{std::byte{0x80}}, zero{};
                update(marker);
                while (size % 64 != 56) update(zero);
                for (int i = 7; i >= 0; --i) {
                    const std::array<std::byte, 1> b{static_cast<std::byte>(bits >> (i * 8))};
                    update(b);
                }
                std::string result;
                for (const auto value : state) result += std::format("{:08x}", value);
                return result;
            }
        };
    } // namespace
    std::string digest(std::span<const std::byte> bytes) {
        Sha256 hash;
        hash.update(bytes);
        return hash.finish();
    }
    std::string digest_file(const std::filesystem::path& path) {
        std::ifstream file{path, std::ios::binary};
        if (!file) throw std::runtime_error{"Cannot read " + path.string()};
        file.exceptions(std::ios::badbit);
        std::array<std::byte, 65536> buffer;
        Sha256 hash;
        while (file) {
            file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
            hash.update({buffer.data(), static_cast<std::size_t>(file.gcount())});
        }
        return hash.finish();
    }
} // namespace flowdit::serialization
