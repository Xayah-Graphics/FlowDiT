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
        json = {{"solver", static_cast<int>(request.solver)}, {"steps", request.step_count}, {"guidance", request.guidance}, {"seed", request.seed}, {"class", request.class_index ? nlohmann::json(*request.class_index) : nlohmann::json(nullptr)}};
    }
    void from_json(const nlohmann::json& json, SamplingRequest& request) {
        request.solver     = static_cast<SamplingSolver>(json.at("solver").get<int>());
        request.step_count = json.at("steps");
        request.guidance   = json.at("guidance");
        request.seed       = json.at("seed");
        if (json.at("class").is_null()) request.class_index.reset();
        else request.class_index = json.at("class").get<std::uint32_t>();
    }
    void to_json(nlohmann::json& json, const SampleInfo& info) {
        json = {{"checkpoint", info.checkpoint.generic_string()}, {"request", info.request}, {"source", static_cast<int>(info.source)}, {"training_step", info.training_step}, {"nfe", info.nfe}, {"model", nlohmann::json::parse(serialize_model(info.model))}, {"labels", info.labels}};
    }
    void from_json(const nlohmann::json& json, SampleInfo& info) {
        info.checkpoint    = json.at("checkpoint").get<std::string>();
        info.request       = json.at("request").get<SamplingRequest>();
        info.source        = static_cast<ParameterSource>(json.at("source").get<int>());
        info.training_step = json.at("training_step");
        info.nfe           = json.at("nfe");
        info.model         = deserialize_model(json.at("model").dump());
        info.labels        = json.at("labels").get<std::vector<std::uint32_t>>();
    }
} // namespace flowdit
namespace flowdit::output {
    void write_configuration(const RunConfiguration& configuration) {
        const auto& optimizer = configuration.optimizer;
        const nlohmann::json json{{"dataset", configuration.dataset.generic_string()}, {"dataset_type", static_cast<int>(configuration.dataset_type)}, {"patch_size", configuration.patch_size}, {"output", configuration.output.generic_string()}, {"end_step", configuration.end_step}, {"seed", configuration.seed}, {"device", configuration.device}, {"execution_steps", configuration.execution_steps}, {"log_interval", configuration.log_interval}, {"preview_interval", configuration.preview_interval}, {"save_interval", configuration.save_interval}, {"preview", configuration.preview}, {"optimizer", {{"learning_rate", optimizer.learning_rate}, {"first_decay", optimizer.first_decay}, {"second_decay", optimizer.second_decay}, {"epsilon", optimizer.epsilon}, {"weight_decay", optimizer.weight_decay}, {"ema_half_life", optimizer.exponential_average.half_life_samples}, {"ema_ramp", optimizer.exponential_average.ramp_up_ratio}}}};
        std::ofstream file{configuration.output / "run.json"};
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file << json.dump(2) << '\n';
    }
    RunConfiguration read_configuration(const std::filesystem::path& directory) {
        std::ifstream file{directory / "run.json"};
        file.exceptions(std::ios::failbit | std::ios::badbit);
        const auto json = nlohmann::json::parse(file);
        RunConfiguration result;
        result.dataset                                         = json.at("dataset").get<std::string>();
        result.output                                          = directory;
        result.dataset_type                                    = static_cast<DatasetKind>(json.at("dataset_type").get<int>());
        result.patch_size                                      = json.at("patch_size");
        result.end_step                                        = json.at("end_step");
        result.seed                                            = json.at("seed");
        result.device                                          = json.at("device");
        result.execution_steps                                 = json.at("execution_steps");
        result.log_interval                                    = json.at("log_interval");
        result.preview_interval                                = json.at("preview_interval");
        result.save_interval                                   = json.at("save_interval");
        result.preview                                         = json.at("preview").get<SamplingRequest>();
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
    void write_png(const std::filesystem::path& path, const SamplingResult& images, const std::optional<std::size_t> index) {
        const std::size_t image_bytes = static_cast<std::size_t>(images.model.image.width) * images.model.image.height * 4uz;
        if (index) {
            if (!stbi_write_png(path.string().c_str(), images.model.image.width, images.model.image.height, 4, images.rgba.data() + *index * image_bytes, images.model.image.width * 4u)) throw std::runtime_error{"Cannot write PNG: " + path.string()};
            return;
        }
        constexpr std::uint32_t columns = 10u;
        const std::uint32_t width       = columns * images.model.image.width;
        const std::uint32_t height      = static_cast<std::uint32_t>((images.labels.size() + columns - 1u) / columns) * images.model.image.height;
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 4uz);
        for (std::size_t image = 0; image < images.labels.size(); ++image)
            for (std::uint32_t y = 0; y < images.model.image.height; ++y) {
                const std::size_t destination = ((image / columns * images.model.image.height + y) * width + image % columns * images.model.image.width) * 4uz;
                std::memcpy(pixels.data() + destination, images.rgba.data() + image * image_bytes + y * images.model.image.width * 4uz, images.model.image.width * 4uz);
            }
        if (!stbi_write_png(path.string().c_str(), width, height, 4, pixels.data(), width * 4u)) throw std::runtime_error{"Cannot write PNG: " + path.string()};
    }
    void write_sample(const SampleOutput& sample) {
        write_png(sample.info.path, sample.images);
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
        const auto& image             = result.info.model.image;
        const std::size_t image_bytes = static_cast<std::size_t>(image.width) * image.height * 4uz;
        result.images                 = {.model = result.info.model, .nfe = result.info.nfe, .labels = result.info.labels, .rgba = std::vector<std::uint8_t>(result.info.labels.size() * image_bytes)};
        constexpr std::size_t columns = 10;
        for (std::size_t index = 0; index < result.images.labels.size(); ++index)
            for (std::size_t y = 0; y < image.height; ++y) std::memcpy(result.images.rgba.data() + index * image_bytes + y * image.width * 4uz, pixels.get() + ((index / columns * image.height + y) * width + index % columns * image.width) * 4uz, image.width * 4uz);
        return result;
    }
    RunHistory read_history(const std::filesystem::path& directory) {
        RunHistory result;
        std::ifstream csv{directory / "training.csv"};
        std::string line;
        std::getline(csv, line);
        while (std::getline(csv, line)) {
            std::ranges::replace(line, ',', ' ');
            std::istringstream row{line};
            TrainingRecord record;
            row >> record.step >> record.loss >> record.samples_per_second >> record.training_seconds;
            result.metrics.push_back(record);
        }
        for (const auto& entry : std::filesystem::directory_iterator{directory / "samples"}) {
            if (entry.path().extension() != ".json") continue;
            std::ifstream file{entry.path()};
            auto info = nlohmann::json::parse(file).get<SampleInfo>();
            info.path = entry.path();
            info.path.replace_extension(".png");
            result.samples.push_back(std::move(info));
        }
        std::ranges::sort(result.samples, {}, &SampleInfo::training_step);
        for (const auto& entry : std::filesystem::directory_iterator{directory / "checkpoints"})
            if (entry.path().extension() == ".safetensors") result.checkpoints.push_back(entry.path());
        if (std::filesystem::exists(directory / "final.safetensors")) result.checkpoints.push_back(directory / "final.safetensors");
        std::ranges::sort(result.checkpoints);
        return result;
    }
    std::filesystem::path latest_checkpoint(const std::filesystem::path& directory) {
        std::filesystem::path latest;
        if (std::filesystem::exists(directory / "final.safetensors")) latest = directory / "final.safetensors";
        for (const auto& entry : std::filesystem::directory_iterator{directory / "checkpoints"})
            if (entry.path().extension() == ".safetensors" && (latest.empty() || entry.last_write_time() > std::filesystem::last_write_time(latest))) latest = entry.path();
        if (latest.empty()) throw std::runtime_error{"No checkpoint in " + directory.string()};
        return latest;
    }
    std::string_view solver_name(const SamplingSolver solver) {
        switch (solver) {
        case SamplingSolver::euler: return "Euler";
        case SamplingSolver::heun: return "Heun";
        case SamplingSolver::rk4: return "RK4";
        }
        std::unreachable();
    }
} // namespace flowdit::output
