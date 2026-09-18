module;
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>
#include <nlohmann/json.hpp>
module flowdit.io.output;
import std;
namespace flowdit {
    void to_json(nlohmann::json& json, const SamplingRequest& request) {
        json = {{"solver", static_cast<int>(request.solver)}, {"steps", request.step_count}, {"count", request.count}, {"guidance", request.guidance}, {"seed", request.seed}, {"class", request.class_index ? nlohmann::json(*request.class_index) : nlohmann::json(nullptr)}};
    }
    void from_json(const nlohmann::json& json, SamplingRequest& request) {
        request.solver     = static_cast<SamplingSolver>(json.at("solver").get<int>());
        request.step_count = json.at("steps");
        request.count      = json.at("count");
        request.guidance   = json.at("guidance");
        request.seed       = json.at("seed");
        if (json.at("class").is_null()) request.class_index.reset();
        else request.class_index = json.at("class").get<std::uint32_t>();
    }
    void to_json(nlohmann::json& json, const SampleInfo& info) {
        json = {{"reconstruction", info.reconstruction}, {"checkpoint", info.checkpoint.generic_string()}, {"request", info.request}, {"source", static_cast<int>(info.source)}, {"training_step", info.training_step}, {"nfe", info.nfe}, {"model", nlohmann::json::parse(serialize_model(info.model))}, {"image", nlohmann::json::parse(serialize_image(info.image))}, {"labels", info.labels}};
    }
    void from_json(const nlohmann::json& json, SampleInfo& info) {
        info.reconstruction = json.at("reconstruction");
        info.checkpoint     = json.at("checkpoint").get<std::string>();
        info.request        = json.at("request").get<SamplingRequest>();
        info.source         = static_cast<ParameterSource>(json.at("source").get<int>());
        info.training_step  = json.at("training_step");
        info.nfe            = json.at("nfe");
        info.model          = deserialize_model(json.at("model").dump());
        info.image          = deserialize_image(json.at("image").dump());
        info.labels         = json.at("labels").get<std::vector<std::uint32_t>>();
    }
} // namespace flowdit
namespace flowdit::output {
    void write_configuration(const RunConfiguration& configuration) {
        const auto& optimizer = configuration.optimizer;
        const nlohmann::json json{{"stage", static_cast<int>(configuration.stage)}, {"autoencoder", nlohmann::json::parse(serialize_autoencoder(configuration.autoencoder))}, {"autoencoder_checkpoint", configuration.autoencoder_checkpoint.generic_string()}, {"log_interval", configuration.log_interval}, {"preview_interval", configuration.preview_interval}, {"save_interval", configuration.save_interval}, {"dataset", configuration.dataset.generic_string()}, {"model", nlohmann::json::parse(serialize_model(configuration.model))}, {"image", nlohmann::json::parse(serialize_image(configuration.image))}, {"batch", configuration.batch}, {"horizontal_flip", configuration.horizontal_flip}, {"preview", configuration.preview}, {"end_step", configuration.end_step}, {"seed", configuration.seed},
            {"optimizer", {{"learning_rate", optimizer.learning_rate}, {"first_decay", optimizer.first_decay}, {"second_decay", optimizer.second_decay}, {"epsilon", optimizer.epsilon}, {"weight_decay", optimizer.weight_decay}, {"ema_half_life", optimizer.exponential_average.half_life_samples}, {"ema_ramp", optimizer.exponential_average.ramp_up_ratio}}}};
        std::ofstream file{configuration.output / "run.json"};
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file << json.dump(2) << '\n';
    }
    RunConfiguration read_configuration(const std::filesystem::path& directory) {
        std::ifstream file{directory / "run.json"};
        file.exceptions(std::ios::failbit | std::ios::badbit);
        const auto json = nlohmann::json::parse(file);
        RunConfiguration result;
        result.stage                                           = static_cast<TrainingStage>(json.at("stage").get<int>());
        result.autoencoder                                     = deserialize_autoencoder(json.at("autoencoder").dump());
        result.autoencoder_checkpoint                          = json.at("autoencoder_checkpoint").get<std::string>();
        result.log_interval                                    = json.at("log_interval");
        result.preview_interval                                = json.at("preview_interval");
        result.save_interval                                   = json.at("save_interval");
        result.dataset                                         = json.at("dataset").get<std::string>();
        result.output                                          = directory;
        result.model                                           = deserialize_model(json.at("model").dump());
        result.image                                           = deserialize_image(json.at("image").dump());
        result.batch                                           = json.at("batch");
        result.horizontal_flip                                 = json.at("horizontal_flip");
        result.preview                                         = json.at("preview").get<SamplingRequest>();
        result.end_step                                        = json.at("end_step");
        result.seed                                            = json.at("seed");
        const auto& optimizer                                  = json.at("optimizer");
        result.optimizer.learning_rate                         = optimizer.at("learning_rate");
        result.optimizer.first_decay                           = optimizer.at("first_decay");
        result.optimizer.second_decay                          = optimizer.at("second_decay");
        result.optimizer.epsilon                               = optimizer.at("epsilon");
        result.optimizer.weight_decay                          = optimizer.at("weight_decay");
        result.optimizer.exponential_average.half_life_samples = optimizer.at("ema_half_life");
        result.optimizer.exponential_average.ramp_up_ratio     = optimizer.at("ema_ramp");
        return result;
    }
    void write_sample(const SampleOutput& sample) {
        const auto& image             = sample.info.image;
        const std::size_t image_bytes = static_cast<std::size_t>(image.width) * image.height * 4uz;
        const std::uint32_t columns   = std::min(static_cast<std::uint32_t>(sample.info.labels.size()), sample.info.reconstruction ? 2u : 10u);
        const std::uint32_t width     = columns * image.width;
        const std::uint32_t height    = static_cast<std::uint32_t>((sample.info.labels.size() + columns - 1u) / columns) * image.height;
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 4uz);
        for (std::size_t i = 0; i < sample.info.labels.size(); ++i)
            for (std::uint32_t y = 0; y < image.height; ++y) {
                const std::size_t destination = ((i / columns * image.height + y) * width + i % columns * image.width) * 4uz;
                std::memcpy(pixels.data() + destination, sample.rgba.data() + i * image_bytes + y * image.width * 4uz, image.width * 4uz);
            }
        if (!stbi_write_png(sample.info.path.string().c_str(), width, height, 4, pixels.data(), width * 4u)) throw std::runtime_error{"Cannot write PNG: " + sample.info.path.string()};
        auto path = sample.info.path;
        path.replace_extension(".json");
        std::ofstream file{path};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file << nlohmann::json(sample.info).dump(2) << '\n';
    }
    SampleOutput read_sample(const std::filesystem::path& path) {
        auto metadata = path;
        metadata.replace_extension(".json");
        std::ifstream file{metadata};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        SampleOutput result;
        result.info      = nlohmann::json::parse(file).get<SampleInfo>();
        result.info.path = path;
        int width{}, height{}, channels{};
        const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels{stbi_load(path.string().c_str(), &width, &height, &channels, 4), stbi_image_free};
        if (!pixels) throw std::runtime_error{"Cannot read PNG: " + path.string()};
        const auto& image             = result.info.image;
        const std::size_t image_bytes = static_cast<std::size_t>(image.width) * image.height * 4uz;
        result.rgba.resize(result.info.labels.size() * image_bytes);
        const std::size_t columns = std::min(result.info.labels.size(), result.info.reconstruction ? 2uz : 10uz);
        for (std::size_t index = 0; index < result.info.labels.size(); ++index)
            for (std::size_t y = 0; y < image.height; ++y) std::memcpy(result.rgba.data() + index * image_bytes + y * image.width * 4uz, pixels.get() + ((index / columns * image.height + y) * width + index % columns * image.width) * 4uz, image.width * 4uz);
        return result;
    }
    RunHistory read_history(const std::filesystem::path& directory) {
        RunHistory result;
        std::ifstream csv;
        csv.exceptions(std::ios::badbit | std::ios::failbit);
        csv.open(directory / "training.csv");
        csv.exceptions(std::ios::badbit);
        std::string line;
        std::getline(csv, line);
        while (std::getline(csv, line)) {
            std::ranges::replace(line, ',', ' ');
            std::istringstream row{line};
            TrainingRecord record;
            row >> record.step >> record.loss >> record.samples_per_second >> record.training_seconds;
            for (auto& value : record.components) row >> value;
            result.metrics.push_back(record);
        }
        if (!std::filesystem::exists(directory / "samples")) return result;
        std::uint64_t latest{};
        for (const auto& entry : std::filesystem::directory_iterator{directory / "samples"}) {
            const auto name = entry.path().filename().string();
            if (!name.starts_with("step-") || !name.ends_with("-ema.json")) continue;
            std::uint64_t step{};
            std::from_chars(name.data() + 5, name.data() + name.size() - 9, step);
            if (step >= latest) {
                latest         = step;
                result.preview = entry.path();
                result.preview.replace_extension(".png");
            }
        }
        return result;
    }
} // namespace flowdit::output
