#include "qwen.h"
#include "orpheus.h"
#include "../src/JSON.h"
#include "../src/audio_process.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <unordered_map>

namespace chatllm::omnivoice
{
    static constexpr int MAX_CODEBOOKS = 8;
    static constexpr int MAX_UPSAMPLING_RATIOS = 8;

    struct Config : public qwen::v3::Config
    {
        int audio_vocab_size;
        int audio_mask_id;
        int num_audio_codebook;
        int audio_codebook_weights[MAX_CODEBOOKS];
    };

    struct InferenceOptions
    {
        double speed = 1.0;
        double duration = -1.0;
        int num_step = 32;
        double guidance_scale = 2.0;
        double t_shift = 0.1;
        double layer_penalty_factor = 5.0;
        double position_temperature = 5.0;
        double class_temperature = 0.0;
        bool denoise = true;
        bool preprocess_prompt = true;
        bool postprocess_output = true;
    };

    namespace detail
    {
        static bool is_ascii(const std::string &s)
        {
            for (unsigned char c : s)
                if (c & 0x80)
                    return false;
            return true;
        }

        static std::vector<uint32_t> utf8_to_codepoints(const std::string &text)
        {
            std::vector<uint32_t> out;
            size_t i = 0;
            while (i < text.size())
            {
                uint32_t cp = 0;
                unsigned char c = (unsigned char)text[i];
                size_t len = 1;
                if ((c & 0x80) == 0)
                {
                    cp = c;
                }
                else if ((c & 0xE0) == 0xC0 && (i + 1) < text.size())
                {
                    cp = ((uint32_t)(c & 0x1F) << 6) |
                         ((uint32_t)(text[i + 1] & 0x3F));
                    len = 2;
                }
                else if ((c & 0xF0) == 0xE0 && (i + 2) < text.size())
                {
                    cp = ((uint32_t)(c & 0x0F) << 12) |
                         ((uint32_t)(text[i + 1] & 0x3F) << 6) |
                         ((uint32_t)(text[i + 2] & 0x3F));
                    len = 3;
                }
                else if ((c & 0xF8) == 0xF0 && (i + 3) < text.size())
                {
                    cp = ((uint32_t)(c & 0x07) << 18) |
                         ((uint32_t)(text[i + 1] & 0x3F) << 12) |
                         ((uint32_t)(text[i + 2] & 0x3F) << 6) |
                         ((uint32_t)(text[i + 3] & 0x3F));
                    len = 4;
                }
                else
                {
                    cp = c;
                }
                out.push_back(cp);
                i += len;
            }
            return out;
        }

        static double char_weight(uint32_t cp)
        {
            if (((cp >= 'A') && (cp <= 'Z')) || ((cp >= 'a') && (cp <= 'z')))
                return 1.0;
            if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r')
                return 0.2;
            if ((cp >= '0') && (cp <= '9'))
                return 3.5;

            if ((cp >= 0x3000 && cp <= 0x303F) || (cp >= 0xFF00 && cp <= 0xFF65))
                return 0.5;
            if ((cp >= 0x3040 && cp <= 0x30FF))
                return 2.2;
            if ((cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x3130 && cp <= 0x318F) || (cp >= 0xAC00 && cp <= 0xD7AF))
                return 2.5;
            if ((cp >= 0x0600 && cp <= 0x08FF) || (cp >= 0xFB50 && cp <= 0xFDFF) || (cp >= 0xFE70 && cp <= 0xFEFF))
                return 1.5;
            if ((cp >= 0x4E00 && cp <= 0x9FFF) || cp > 0x20000)
                return 3.0;

            if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
                return 0.5;

            return 1.0;
        }

        static double text_weight(const std::string &text)
        {
            double total = 0.0;
            for (uint32_t cp : utf8_to_codepoints(text))
                total += char_weight(cp);
            return total;
        }

        static std::string replace_newlines_with_periods(const std::string &text);

        static bool contains_cjk(const std::string &text)
        {
            for (uint32_t cp : utf8_to_codepoints(text))
                if (cp >= 0x4E00 && cp <= 0x9FFF)
                    return true;
            return false;
        }

        static bool ends_with_any(const std::string &text, const std::vector<std::string> &suffixes)
        {
            for (const auto &suffix : suffixes)
            {
                if ((text.size() >= suffix.size()) &&
                    (text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0))
                    return true;
            }
            return false;
        }

        static std::string add_punctuation(const std::string &text)
        {
            std::string out = utils::trim(text);
            if (out.empty())
                return out;

            static const std::vector<std::string> punct({
                ".", "!", "?", "...", "。", "！", "？", "…"
            });
            if (!ends_with_any(out, punct))
                out += contains_cjk(out) ? "。" : ".";
            return out;
        }

        static std::string combine_text(const std::string &text, const std::string &ref_text = "")
        {
            std::string full_text;
            if (!utils::trim(ref_text).empty())
                full_text = utils::trim(ref_text) + " " + utils::trim(text);
            else
                full_text = utils::trim(text);
            return replace_newlines_with_periods(full_text);
        }

        static void trim_silence_edges(std::vector<float> &samples, int sample_rate, int lead_ms, int trail_ms)
        {
            if (samples.empty() || sample_rate <= 0)
                return;

            constexpr float silence_threshold = 0.00316227766f; // -50 dBFS
            int first = -1;
            int last = -1;
            for (int i = 0; i < (int)samples.size(); i++)
            {
                if (std::fabs(samples[i]) > silence_threshold)
                {
                    if (first < 0)
                        first = i;
                    last = i;
                }
            }
            if (first < 0 || last < first)
                return;

            const int keep_lead = sample_rate * lead_ms / 1000;
            const int keep_trail = sample_rate * trail_ms / 1000;
            const int start = std::max(0, first - keep_lead);
            const int end = std::min((int)samples.size(), last + keep_trail + 1);
            if (start <= 0 && end >= (int)samples.size())
                return;
            samples = std::vector<float>(samples.begin() + start, samples.begin() + end);
        }

        static int estimate_target_tokens(const std::string &text, double speed, const std::string &ref_text = "", int ref_tokens = -1)
        {
            const std::string default_ref_text = "Nice to meet you.";
            const std::string ref = utils::trim(ref_text).empty() ? default_ref_text : utils::trim(ref_text);
            const double base_ref_tokens = ref_tokens > 0 ? (double)ref_tokens : 25.0;
            double ref_weight = text_weight(ref);
            if (ref_weight <= 0.0)
                ref_weight = text_weight(default_ref_text);

            double target_weight = text_weight(text);
            if (target_weight <= 0.0)
                target_weight = 1.0;

            double est = base_ref_tokens * target_weight / ref_weight;
            if ((speed > 0.0) && (speed != 1.0))
                est /= speed;

            const double low_threshold = 50.0;
            const double boost_strength = 3.0;
            if (est < low_threshold)
            {
                const double alpha = 1.0 / boost_strength;
                est = low_threshold * std::pow(est / low_threshold, alpha);
            }

            if (est < 1.0)
                est = 1.0;
            return (int)std::ceil(est);
        }

        static double sample_gumbel(std::mt19937 &rng)
        {
            std::uniform_real_distribution<double> dist(1e-6, 1.0 - 1e-6);
            double u = dist(rng);
            return -std::log(-std::log(u));
        }

        static std::vector<double> get_time_steps(int num_step, double t_shift)
        {
            std::vector<double> timesteps;
            timesteps.reserve(num_step + 1);
            for (int i = 0; i <= num_step; i++)
            {
                double t = (double)i / (double)num_step;
                double shifted = t_shift * t / (1.0 + (t_shift - 1.0) * t);
                timesteps.push_back(shifted);
            }
            return timesteps;
        }

        static void log_softmax(const float *logits, int vocab_size, std::vector<double> &out)
        {
            out.resize(vocab_size);
            double max_v = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < vocab_size; i++)
                max_v = std::max(max_v, (double)logits[i]);

            double sum = 0.0;
            for (int i = 0; i < vocab_size; i++)
            {
                out[i] = std::exp((double)logits[i] - max_v);
                sum += out[i];
            }

            double log_sum = std::log(sum);
            for (int i = 0; i < vocab_size; i++)
                out[i] = std::log(out[i]) - log_sum;
        }

        static std::string replace_newlines_with_periods(const std::string &text)
        {
            std::string out;
            out.reserve(text.size());
            bool just_wrote_period = false;
            for (char c : text)
            {
                if (c == '\r' || c == '\n')
                {
                    if (!just_wrote_period)
                    {
                        out.push_back('.');
                        just_wrote_period = true;
                    }
                }
                else
                {
                    out.push_back(c);
                    just_wrote_period = (c == '.');
                }
            }
            return utils::trim(out);
        }

        static const std::unordered_map<std::string, std::string> &language_map()
        {
            static const std::unordered_map<std::string, std::string> map({
                {"english", "en"},
                {"chinese", "zh"},
                {"japanese", "ja"},
                {"korean", "ko"},
                {"french", "fr"},
                {"german", "de"},
                {"spanish", "es"},
                {"italian", "it"},
                {"portuguese", "pt"},
                {"russian", "ru"},
                {"arabic", "ar"},
                {"hindi", "hi"},
                {"cantonese", "yue"},
            });
            return map;
        }

        static const std::vector<std::set<std::string>> &instruct_categories()
        {
            static const std::vector<std::set<std::string>> categories({
                {"male", "female", "男", "女"},
                {"child", "teenager", "young adult", "middle-aged", "elderly", "儿童", "少年", "青年", "中年", "老年"},
                {"very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch", "极低音调", "低音调", "中音调", "高音调", "极高音调"},
                {"whisper", "耳语"},
                {"american accent", "british accent", "australian accent", "chinese accent", "canadian accent", "indian accent",
                 "korean accent", "portuguese accent", "russian accent", "japanese accent"},
                {"河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话", "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话"},
            });
            return categories;
        }

        static const std::set<std::string> &all_instruct_items()
        {
            static std::set<std::string> items = []() {
                std::set<std::string> all;
                for (const auto &cat : instruct_categories())
                    all.insert(cat.begin(), cat.end());
                return all;
            }();
            return items;
        }

        static bool contains_non_ascii(const std::string &s)
        {
            return !is_ascii(s);
        }
    }

    class Tokenizer : public qwen::v3::Tokenizer
    {
    public:
        Tokenizer(const BaseConfig &config)
            : qwen::v3::Tokenizer(config, nullptr)
        {}

        size_t load(tokenizer::DataReader *buffer, int n_vocab) override
        {
            size_t r = qwen::v3::Tokenizer::load(buffer, n_vocab);

            denoise_token_id         = tp->PieceToId("<|denoise|>");
            lang_start_token_id      = tp->PieceToId("<|lang_start|>");
            lang_end_token_id        = tp->PieceToId("<|lang_end|>");
            instruct_start_token_id  = tp->PieceToId("<|instruct_start|>");
            instruct_end_token_id    = tp->PieceToId("<|instruct_end|>");
            text_start_token_id      = tp->PieceToId("<|text_start|>");
            text_end_token_id        = tp->PieceToId("<|text_end|>");

            CHATLLM_CHECK(text_start_token_id >= 0) << "<|text_start|> is missing from OmniVoice tokenizer";
            CHATLLM_CHECK(text_end_token_id >= 0) << "<|text_end|> is missing from OmniVoice tokenizer";
            return r;
        }

        static std::string normalize_language(const std::string &language)
        {
            std::string s = utils::to_lower(utils::trim(language));
            if ((s.size() == 0) || (s == "auto") || (s == "none"))
                return "";

            const auto &map = detail::language_map();
            auto it = map.find(s);
            if (it != map.end())
                return it->second;

            bool simple_code = true;
            for (char c : s)
            {
                if (!(((c >= 'a') && (c <= 'z')) || (c == '-') || (c == '_')))
                {
                    simple_code = false;
                    break;
                }
            }
            if (simple_code && (s.size() <= 5))
                return s;

            return "";
        }

        static std::string normalize_instruct(const std::string &instruct)
        {
            std::string s = utils::trim(instruct);
            if (s.size() == 0)
                return "";

            for (size_t pos = s.find("，"); pos != std::string::npos; pos = s.find("，", pos))
                s.replace(pos, std::string("，").size(), ",");

            std::vector<std::string> items;
            utils::split(s, ",", items);

            std::vector<std::string> normalized;
            normalized.reserve(items.size());
            for (auto &item : items)
            {
                item = utils::trim(item);
                if (item.size() == 0)
                    continue;

                if (detail::is_ascii(item))
                    item = utils::to_lower(item);

                CHATLLM_CHECK(detail::all_instruct_items().count(item) > 0)
                    << "unsupported OmniVoice instruct item `" << item << "`";
                normalized.push_back(item);
            }

            for (const auto &cat : detail::instruct_categories())
            {
                int hit = 0;
                for (const auto &item : normalized)
                    if (cat.count(item) > 0)
                        hit++;
                CHATLLM_CHECK(hit <= 1) << "conflicting OmniVoice instruct items in the same category";
            }

            const std::string separator = std::any_of(normalized.begin(), normalized.end(),
                [](const std::string &item) { return detail::contains_non_ascii(item); }) ? "，" : ", ";
            return utils::join(normalized, separator);
        }

        void build_style_tokens(bool denoise, std::vector<int> &ids) const
        {
            std::string lang = language.size() > 0 ? language : "None";
            std::string style = denoise ? "<|denoise|>" : "";
            style += "<|lang_start|>" + lang + "<|lang_end|>";
            style += "<|instruct_start|>" + (instruct.size() > 0 ? instruct : "None") + "<|instruct_end|>";
            BaseTokenizer::encode(style, ids);
        }

        void wrap_text_tokens(const std::vector<int> &input_ids, std::vector<int> &ids) const
        {
            ids.clear();
            ids.reserve(input_ids.size() + 2);
            ids.push_back(text_start_token_id);
            ids.insert(ids.end(), input_ids.begin(), input_ids.end());
            ids.push_back(text_end_token_id);
        }

    public:
        std::string language = "auto";
        std::string instruct = "";
        std::string ref_text = "";
        std::string ref_audio_file = "";
        int denoise_token_id = -1;
        int lang_start_token_id = -1;
        int lang_end_token_id = -1;
        int instruct_start_token_id = -1;
        int instruct_end_token_id = -1;
        int text_start_token_id = -1;
        int text_end_token_id = -1;
    };

    namespace higgs
    {
        struct DecoderConfig
        {
            int sample_rate = 24000;
            int frame_rate = 25;
            int hop_length = 960;
            int hidden_size = 1024;
            int semantic_hidden_size = 768;
            int acoustic_hidden_size = 256;
            int encoder_hidden_size = 64;
            int decoder_hidden_size = 1024;
            int codebook_size = 1024;
            int codebook_dim = 64;
            int num_quantizers = 8;
            int downsampling_ratios[MAX_UPSAMPLING_RATIOS];
            int num_downsampling_ratios = 0;
            int upsampling_ratios[MAX_UPSAMPLING_RATIOS];
            int num_upsampling_ratios = 0;
        };

        class DacResidualUnit : public Block
        {
        public:
            DacResidualUnit(InitContext *ctx, int dimension, int dilation)
                : snake1(ctx, dimension),
                  conv1(ctx, dimension, dimension, 7, 1, ((7 - 1) * dilation) / 2, dilation),
                  snake2(ctx, dimension),
                  conv2(ctx, dimension, dimension, 1)
            {
            }

            ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_state) override
            {
                ggml::tensor *output = snake1.forward(ctx, hidden_state);
                output = conv1.forward(ctx, output);
                output = snake2.forward(ctx, output);
                output = conv2.forward(ctx, output);

                const int64_t padding = (hidden_state->ne[0] - output->ne[0]) / 2;
                if (padding > 0)
                {
                    hidden_state = ggml::view_3d(ctx, hidden_state, output->ne[0], hidden_state->ne[1], hidden_state->ne[2],
                        hidden_state->ne[0] * ggml::element_size(hidden_state),
                        hidden_state->ne[0] * hidden_state->ne[1] * ggml::element_size(hidden_state),
                        padding * ggml::element_size(hidden_state));
                }

                return ggml::add(ctx, hidden_state, output);
            }

            int64_t get_param_num(bool effective_only) const override
            {
                int64_t r = 0;
                r += snake1.get_param_num(effective_only);
                r +=  conv1.get_param_num(effective_only);
                r += snake2.get_param_num(effective_only);
                r +=  conv2.get_param_num(effective_only);
                return r;
            }

            void load(const std::string &path, TensorLoader *loader) override
            {
                snake1.load(path + "snake1.", loader);
                 conv1.load(path + "conv1.", loader);
                snake2.load(path + "snake2.", loader);
                 conv2.load(path + "conv2.", loader);
            }

        public:
            orpheus::snac::Snake1D snake1;
            Conv1D conv1;
            orpheus::snac::Snake1D snake2;
            Conv1D conv2;
        };

        static std::vector<float> read_tensor_as_float(ggml::tensor *tensor)
        {
            std::vector<float> out(ggml::nelements(tensor));
            if (ggml::type_of(tensor) == GGML_TYPE_F32)
            {
                Backend::read_tensor_data(tensor, out.data());
                return out;
            }

            std::vector<uint8_t> raw(ggml::nbytes(tensor));
            Backend::read_tensor_data(tensor, raw.data());
            ggml::to_float(ggml::type_of(tensor), raw.data(), out.data(), ggml::get_dim(tensor, 0), ggml::nrows(tensor));
            return out;
        }

        class DacEncoderBlock : public Block
        {
        public:
            DacEncoderBlock(InitContext *ctx, int input_dim, int output_dim, int stride)
                : res_unit1(ctx, input_dim, 1),
                  res_unit2(ctx, input_dim, 3),
                  res_unit3(ctx, input_dim, 9),
                  snake1(ctx, input_dim),
                  conv1(ctx, input_dim, output_dim, 2 * stride, stride, (stride + 1) / 2)
            {
            }

            ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_state) override
            {
                hidden_state = res_unit1.forward(ctx, hidden_state);
                hidden_state = res_unit2.forward(ctx, hidden_state);
                hidden_state = res_unit3.forward(ctx, hidden_state);
                hidden_state = snake1.forward(ctx, hidden_state);
                hidden_state = conv1.forward(ctx, hidden_state);
                return hidden_state;
            }

            int64_t get_param_num(bool effective_only) const override
            {
                int64_t r = 0;
                r += res_unit1.get_param_num(effective_only);
                r += res_unit2.get_param_num(effective_only);
                r += res_unit3.get_param_num(effective_only);
                r += snake1.get_param_num(effective_only);
                r += conv1.get_param_num(effective_only);
                return r;
            }

            void load(const std::string &path, TensorLoader *loader) override
            {
                res_unit1.load(path + "res_unit1.", loader);
                res_unit2.load(path + "res_unit2.", loader);
                res_unit3.load(path + "res_unit3.", loader);
                  snake1.load(path + "snake1.", loader);
                   conv1.load(path + "conv1.", loader);
            }

        public:
            DacResidualUnit res_unit1;
            DacResidualUnit res_unit2;
            DacResidualUnit res_unit3;
            orpheus::snac::Snake1D snake1;
            Conv1D conv1;
        };

        class DacEncoder : public Block
        {
        public:
            DacEncoder(InitContext *ctx, const DecoderConfig &config)
                : conv1(ctx, 1, config.encoder_hidden_size, 7, 1, 3),
                  snake1(ctx, config.encoder_hidden_size << config.num_downsampling_ratios),
                  conv2(ctx, config.encoder_hidden_size << config.num_downsampling_ratios, config.acoustic_hidden_size, 3, 1, 1)
            {
                int input_dim = config.encoder_hidden_size;
                for (int i = 0; i < config.num_downsampling_ratios; i++)
                {
                    int output_dim = input_dim << 1;
                    blocks.emplace_back(std::make_unique<DacEncoderBlock>(ctx, input_dim, output_dim, config.downsampling_ratios[i]));
                    input_dim = output_dim;
                }
            }

            ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_state) override
            {
                hidden_state = conv1.forward(ctx, hidden_state);
                for (auto &block : blocks)
                    hidden_state = block->forward(ctx, hidden_state);
                hidden_state = snake1.forward(ctx, hidden_state);
                hidden_state = conv2.forward(ctx, hidden_state);
                return hidden_state;
            }

            int64_t get_param_num(bool effective_only) const override
            {
                int64_t r = 0;
                r += conv1.get_param_num(effective_only);
                for (auto &block : blocks)
                    r += block->get_param_num(effective_only);
                r += snake1.get_param_num(effective_only);
                r += conv2.get_param_num(effective_only);
                return r;
            }

            void load(const std::string &path, TensorLoader *loader) override
            {
                conv1.load(path + "conv1.", loader);
                for (size_t i = 0; i < blocks.size(); i++)
                    blocks[i]->load(path + "block." + std::to_string(i) + ".", loader);
                snake1.load(path + "snake1.", loader);
                conv2.load(path + "conv2.", loader);
            }

        public:
            Conv1D conv1;
            std::vector<std::unique_ptr<DacEncoderBlock>> blocks;
            orpheus::snac::Snake1D snake1;
            Conv1D conv2;
        };

        class DacDecoderBlock : public Block
        {
        public:
            DacDecoderBlock(InitContext *ctx, int input_dim, int output_dim, int stride)
                : snake1(ctx, input_dim),
                  conv_t1(ctx, input_dim, output_dim, 2 * stride, stride, (stride + 1) / 2, stride % 2),
                  res_unit1(ctx, output_dim, 1),
                  res_unit2(ctx, output_dim, 3),
                  res_unit3(ctx, output_dim, 9)
            {
            }

            ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_state) override
            {
                hidden_state = snake1.forward(ctx, hidden_state);
                hidden_state = conv_t1.forward(ctx, hidden_state);
                hidden_state = res_unit1.forward(ctx, hidden_state);
                hidden_state = res_unit2.forward(ctx, hidden_state);
                hidden_state = res_unit3.forward(ctx, hidden_state);
                return hidden_state;
            }

            int64_t get_param_num(bool effective_only) const override
            {
                int64_t r = 0;
                r += snake1.get_param_num(effective_only);
                r += conv_t1.get_param_num(effective_only);
                r += res_unit1.get_param_num(effective_only);
                r += res_unit2.get_param_num(effective_only);
                r += res_unit3.get_param_num(effective_only);
                return r;
            }

            void load(const std::string &path, TensorLoader *loader) override
            {
                 snake1.load(path + "snake1.", loader);
                 conv_t1.load(path + "conv_t1.", loader);
                res_unit1.load(path + "res_unit1.", loader);
                res_unit2.load(path + "res_unit2.", loader);
                res_unit3.load(path + "res_unit3.", loader);
            }

        public:
            orpheus::snac::Snake1D snake1;
            ConvTransposed1D conv_t1;
            DacResidualUnit res_unit1;
            DacResidualUnit res_unit2;
            DacResidualUnit res_unit3;
        };

        class DacDecoder : public Block
        {
        public:
            DacDecoder(InitContext *ctx, const DecoderConfig &config)
                : conv1(ctx, config.acoustic_hidden_size, config.decoder_hidden_size, 7, 1, 3),
                  snake1(ctx, config.decoder_hidden_size >> config.num_upsampling_ratios),
                  conv2(ctx, config.decoder_hidden_size >> config.num_upsampling_ratios, 1, 7, 1, 3)
            {
                int input_dim = config.decoder_hidden_size;
                for (int i = 0; i < config.num_upsampling_ratios; i++)
                {
                    int output_dim = input_dim / 2;
                    blocks.emplace_back(std::make_unique<DacDecoderBlock>(ctx, input_dim, output_dim, config.upsampling_ratios[i]));
                    input_dim = output_dim;
                }
            }

            ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_state) override
            {
                hidden_state = conv1.forward(ctx, hidden_state);
                for (auto &block : blocks)
                    hidden_state = block->forward(ctx, hidden_state);
                hidden_state = snake1.forward(ctx, hidden_state);
                hidden_state = conv2.forward(ctx, hidden_state);
                return hidden_state;
            }

            int64_t get_param_num(bool effective_only) const override
            {
                int64_t r = 0;
                r += conv1.get_param_num(effective_only);
                for (auto &block : blocks)
                    r += block->get_param_num(effective_only);
                r += snake1.get_param_num(effective_only);
                r += conv2.get_param_num(effective_only);
                return r;
            }

            void load(const std::string &path, TensorLoader *loader) override
            {
                conv1.load(path + "conv1.", loader);
                for (size_t i = 0; i < blocks.size(); i++)
                    blocks[i]->load(path + "block." + std::to_string(i) + ".", loader);
                snake1.load(path + "snake1.", loader);
                conv2.load(path + "conv2.", loader);
            }

        public:
            Conv1D conv1;
            std::vector<std::unique_ptr<DacDecoderBlock>> blocks;
            orpheus::snac::Snake1D snake1;
            Conv1D conv2;
        };

        class VectorQuantize : public Block
        {
        public:
            VectorQuantize(InitContext *ctx, int hidden_size, int codebook_size, int codebook_dim)
                : hidden_size(hidden_size),
                  codebook_size(codebook_size),
                  codebook_dim(codebook_dim),
                  project_in(ctx, hidden_size, codebook_dim),
                  project_out(ctx, codebook_dim, hidden_size),
                  codebook(ctx, codebook_size, codebook_dim)
            {
            }

            ggml::tensor *dequantize(ComputeContext *ctx, ggml::tensor *embed_id)
            {
                ggml::tensor *quantized = codebook.forward(ctx, embed_id);
                quantized = ggml::reshape_2d(ctx, quantized, ggml::get_dim(quantized, 0), ggml::nelements(embed_id));
                quantized = project_out.forward(ctx, quantized);
                return quantized;
            }

            void prepare_encode(void)
            {
                project_in_weight = read_tensor_as_float(project_in.weight);
                project_in_bias = read_tensor_as_float(project_in.bias);
                project_out_weight = read_tensor_as_float(project_out.weight);
                project_out_bias = read_tensor_as_float(project_out.bias);
                codebook_weight = read_tensor_as_float(codebook.weight);

                codebook_norm.resize(codebook_size);
                for (int i = 0; i < codebook_size; i++)
                {
                    const float *cb = &codebook_weight[i * codebook_dim];
                    float norm = 0.0f;
                    for (int j = 0; j < codebook_dim; j++)
                        norm += cb[j] * cb[j];
                    codebook_norm[i] = norm;
                }
                cache_ready = true;
            }

            int encode_and_dequantize(const float *input, std::vector<float> &decoded) const
            {
                CHATLLM_CHECK(cache_ready) << "OmniVoice quantizer encode cache is not ready";

                std::vector<float> projected(codebook_dim);
                for (int out_idx = 0; out_idx < codebook_dim; out_idx++)
                {
                    float v = project_in_bias[out_idx];
                    const float *w = &project_in_weight[out_idx * hidden_size];
                    for (int in_idx = 0; in_idx < hidden_size; in_idx++)
                        v += w[in_idx] * input[in_idx];
                    projected[out_idx] = v;
                }

                int best_index = 0;
                float best_score = -std::numeric_limits<float>::infinity();
                for (int code = 0; code < codebook_size; code++)
                {
                    const float *cb = &codebook_weight[code * codebook_dim];
                    float dot = 0.0f;
                    for (int dim = 0; dim < codebook_dim; dim++)
                        dot += projected[dim] * cb[dim];
                    const float score = 2.0f * dot - codebook_norm[code];
                    if (score > best_score)
                    {
                        best_score = score;
                        best_index = code;
                    }
                }

                decoded.resize(hidden_size);
                const float *cb = &codebook_weight[best_index * codebook_dim];
                for (int out_idx = 0; out_idx < hidden_size; out_idx++)
                {
                    float v = project_out_bias[out_idx];
                    const float *w = &project_out_weight[out_idx * codebook_dim];
                    for (int in_idx = 0; in_idx < codebook_dim; in_idx++)
                        v += w[in_idx] * cb[in_idx];
                    decoded[out_idx] = v;
                }

                return best_index;
            }

            int64_t get_param_num(bool effective_only) const override
            {
                return project_in.get_param_num(effective_only)
                    + project_out.get_param_num(effective_only)
                    + codebook.get_param_num(effective_only);
            }

            void load(const std::string &path, TensorLoader *loader) override
            {
                 project_in.load(path + "project_in.", loader);
                project_out.load(path + "project_out.", loader);
                    codebook.load(path + "codebook.", loader);
                cache_ready = false;
            }

        public:
            const int hidden_size;
            const int codebook_size;
            const int codebook_dim;
            Linear project_in;
            Linear project_out;
            Embedding codebook;
            bool cache_ready = false;
            std::vector<float> project_in_weight;
            std::vector<float> project_in_bias;
            std::vector<float> project_out_weight;
            std::vector<float> project_out_bias;
            std::vector<float> codebook_weight;
            std::vector<float> codebook_norm;
        };

        class ResidualVectorQuantize : public Block
        {
        public:
            ResidualVectorQuantize(InitContext *ctx, int hidden_size, int codebook_size, int codebook_dim, int num_quantizers)
                : hidden_size(hidden_size),
                  num_quantizers(num_quantizers)
            {
                for (int i = 0; i < num_quantizers; i++)
                    quantizers.emplace_back(std::make_unique<VectorQuantize>(ctx, hidden_size, codebook_size, codebook_dim));
            }

            ggml::tensor *dequantize(ComputeContext *ctx, const std::vector<ggml::tensor *> &codes)
            {
                CHATLLM_CHECK((int)codes.size() == num_quantizers) << "invalid number of quantizer codes";
                ggml::tensor *quantized = nullptr;
                for (int i = 0; i < num_quantizers; i++)
                {
                    ggml::tensor *decoded = quantizers[i]->dequantize(ctx, codes[i]);
                    quantized = quantized ? ggml::add(ctx, quantized, decoded) : decoded;
                }
                return quantized;
            }

            void prepare_encode(void)
            {
                for (auto &quantizer : quantizers)
                    quantizer->prepare_encode();
            }

            void encode(const std::vector<float> &embeddings, int num_frames, std::vector<int> &flat_codes) const
            {
                CHATLLM_CHECK(num_frames >= 0) << "invalid OmniVoice frame count";
                CHATLLM_CHECK((int)embeddings.size() == num_frames * hidden_size) << "invalid OmniVoice embedding shape";

                std::vector<float> residual = embeddings;
                std::vector<float> decoded(hidden_size);
                flat_codes.assign(num_frames * num_quantizers, 0);

                for (int q = 0; q < num_quantizers; q++)
                {
                    for (int frame = 0; frame < num_frames; frame++)
                    {
                        float *frame_residual = residual.data() + frame * hidden_size;
                        const int code = quantizers[q]->encode_and_dequantize(frame_residual, decoded);
                        flat_codes[frame * num_quantizers + q] = code;
                        for (int hidden = 0; hidden < hidden_size; hidden++)
                            frame_residual[hidden] -= decoded[hidden];
                    }
                }
            }

            int64_t get_param_num(bool effective_only) const override
            {
                int64_t r = 0;
                for (auto &quantizer : quantizers)
                    r += quantizer->get_param_num(effective_only);
                return r;
            }

            void load(const std::string &path, TensorLoader *loader) override
            {
                for (int i = 0; i < num_quantizers; i++)
                    quantizers[i]->load(path + "quantizers." + std::to_string(i) + ".", loader);
            }

        public:
            const int hidden_size;
            const int num_quantizers;
            std::vector<std::unique_ptr<VectorQuantize>> quantizers;
        };

        class Model
        {
        public:
            Model(InitContext *ctx, const DecoderConfig &config)
                : config(config),
                  encoder(ctx, config),
                  fc(ctx, config.hidden_size, config.hidden_size),
                  fc2(ctx, config.hidden_size, config.acoustic_hidden_size),
                  quantizer(ctx, config.hidden_size, config.codebook_size, config.codebook_dim, config.num_quantizers),
                  decoder(ctx, config)
            {
            }

            void load(const std::string &path, ModelLoader &loader)
            {
                encoder.load(path + "acoustic_encoder.", &loader);
                     fc.load(path + "fc.", &loader);
                    fc2.load(path + "fc2.", &loader);
                quantizer.load(path + "quantizer.", &loader);
                  decoder.load(path + "acoustic_decoder.", &loader);
            }

            void prepare_encode(void)
            {
                quantizer.prepare_encode();
            }

            ggml::tensor *encode(ComputeContext *ctx, ggml::tensor *waveform)
            {
                ggml::tensor *hidden_state = encoder.forward(ctx, waveform);
                hidden_state = ggml::permute(ctx, hidden_state, 1, 0, 2, 3);
                hidden_state = ggml::cont(ctx, hidden_state);

                const int num_frames = ggml::get_dim(hidden_state, 1);
                ggml::tensor *semantic_zero = ggml::new_zeros(ctx, ggml::type_of(hidden_state), config.semantic_hidden_size, num_frames);
                hidden_state = ggml::concat(ctx, hidden_state, semantic_zero, 0);
                hidden_state = fc.forward(ctx, hidden_state);
                return hidden_state;
            }

            ggml::tensor *decode(ComputeContext *ctx, const std::vector<ggml::tensor *> &codes)
            {
                ggml::tensor *hidden_state = quantizer.dequantize(ctx, codes);
                hidden_state = fc2.forward(ctx, hidden_state);
                hidden_state = ggml::permute(ctx, hidden_state, 1, 0, 2, 3);
                hidden_state = ggml::cont(ctx, hidden_state);
                hidden_state = decoder.forward(ctx, hidden_state);
                return hidden_state;
            }

        public:
            const DecoderConfig config;
            DacEncoder encoder;
            Linear fc;
            Linear fc2;
            ResidualVectorQuantize quantizer;
            DacDecoder decoder;
        };

        class Generation
        {
        public:
            Generation(const RuntimeConfig &runtime_config, size_t GRAPH_SIZE = 4096)
                : eval(runtime_config, "omnivoice_audio", GRAPH_SIZE, 1),
                  _ctx(eval.get_backend_context())
            {
            }

            bool load_more(ggml::type dtype, const json::JSON &config_json, int num_quantizers)
            {
                const auto cfg = config_json["audio_tokenizer-config.json"];
                const auto acoustic_cfg = cfg["acoustic_model_config"];
                if (!cfg.IsObject() || !acoustic_cfg.IsObject())
                    return false;

                DecoderConfig config;
                config.codebook_size = (int)cfg["codebook_size"].ToInt();
                config.codebook_dim = (int)cfg["codebook_dim"].ToInt();
                config.semantic_hidden_size = (int)cfg["semantic_model_config"]["hidden_size"].ToInt();
                config.acoustic_hidden_size = (int)acoustic_cfg["hidden_size"].ToInt();
                config.encoder_hidden_size = (int)acoustic_cfg["encoder_hidden_size"].ToInt();
                config.decoder_hidden_size = (int)acoustic_cfg["decoder_hidden_size"].ToInt();
                config.hidden_size = config.acoustic_hidden_size + config.semantic_hidden_size;
                config.sample_rate = (int)cfg["sample_rate"].ToInt();
                config.num_quantizers = num_quantizers;

                CHATLLM_CHECK(acoustic_cfg["downsampling_ratios"].IsArray()) << "OmniVoice audio tokenizer missing downsampling_ratios";
                CHATLLM_CHECK(acoustic_cfg["downsampling_ratios"].length() <= MAX_UPSAMPLING_RATIOS) << "too many OmniVoice downsampling ratios";
                config.num_downsampling_ratios = acoustic_cfg["downsampling_ratios"].length();
                for (int i = 0; i < config.num_downsampling_ratios; i++)
                    config.downsampling_ratios[i] = (int)acoustic_cfg["downsampling_ratios"][i].ToInt();

                CHATLLM_CHECK(acoustic_cfg["upsampling_ratios"].IsArray()) << "OmniVoice audio tokenizer missing upsampling_ratios";
                CHATLLM_CHECK(acoustic_cfg["upsampling_ratios"].length() <= MAX_UPSAMPLING_RATIOS) << "too many OmniVoice upsampling ratios";
                config.num_upsampling_ratios = acoustic_cfg["upsampling_ratios"].length();
                for (int i = 0; i < config.num_upsampling_ratios; i++)
                    config.upsampling_ratios[i] = (int)acoustic_cfg["upsampling_ratios"][i].ToInt();

                int hop_length = (int)acoustic_cfg["hop_length"].ToInt();
                const auto pre_cfg = config_json["audio_tokenizer-preprocessor_config.json"];
                if (pre_cfg.IsObject())
                {
                    if (pre_cfg["sampling_rate"].IsIntegral() || pre_cfg["sampling_rate"].IsFloating())
                        config.sample_rate = (int)pre_cfg["sampling_rate"].ToInt();
                    if (pre_cfg["hop_length"].IsIntegral() || pre_cfg["hop_length"].IsFloating())
                        hop_length = (int)pre_cfg["hop_length"].ToInt();
                }
                CHATLLM_CHECK(hop_length > 0) << "invalid OmniVoice audio tokenizer hop_length";
                config.hop_length = hop_length;
                config.frame_rate = (config.sample_rate + hop_length - 1) / hop_length;

                const size_t tensor_ovhd = ggml_tensor_overhead();
                const size_t ctx_size = tensor_ovhd * 2048;
                _ctx.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
                _ctx.dtype = dtype;

                this->config = config;
                model.reset(new Model(&_ctx, this->config));
                loaded = true;
                return true;
            }

            void load(ModelLoader &loader)
            {
                if (!model.get())
                    return;
                loader.push_allocator_manager(eval.get_layer_allocators());
                model->load("audio_tokenizer.", loader);
                loader.pop_allocator_manager();
                model->prepare_encode();
            }

            void encode(const GenerationConfig &gen_config, const std::vector<float> &audio_samples, std::vector<int> &flat_codes)
            {
                CHATLLM_CHECK(loaded && model.get()) << "OmniVoice audio tokenizer is not loaded";
                CHATLLM_CHECK(config.hop_length > 0) << "OmniVoice audio tokenizer hop_length is invalid";
                CHATLLM_CHECK(!audio_samples.empty()) << "OmniVoice reference audio is empty";
                CHATLLM_CHECK((audio_samples.size() % (size_t)config.hop_length) == 0)
                    << "OmniVoice reference audio must be trimmed to a multiple of hop_length";

                std::vector<int64_t> shape;
                std::vector<uint8_t> buf;
                ggml::tensor *audio_tensor = nullptr;
                eval.evaluate(gen_config,
                    [&](ComputeContext *ctx) -> ggml::tensor * {
                        audio_tensor = ggml::new_tensor_2d(ctx, GGML_TYPE_F32, audio_samples.size(), 1);
                        return model->encode(ctx, audio_tensor);
                    },
                    [&](ComputeContext *ctx) {
                        (void)ctx;
                        Backend::write_tensor_data(audio_tensor, audio_samples.data(), 0, audio_samples.size() * sizeof(audio_samples[0]));
                    },
                    ggml::type::GGML_TYPE_F32,
                    shape,
                    buf);

                CHATLLM_CHECK(shape.size() >= 2) << "OmniVoice tokenizer encode output has invalid shape";
                const int num_frames = (int)shape[1];
                std::vector<float> embeddings(buf.size() / sizeof(float));
                memcpy(embeddings.data(), buf.data(), buf.size());
                model->quantizer.encode(embeddings, num_frames, flat_codes);
            }

            void decode(const GenerationConfig &gen_config, const std::vector<int> &flat_codes, std::vector<float> &pcm_samples)
            {
                CHATLLM_CHECK(loaded && model.get()) << "OmniVoice audio tokenizer is not loaded";
                CHATLLM_CHECK(config.num_quantizers > 0) << "OmniVoice audio tokenizer num_quantizers is invalid";
                CHATLLM_CHECK((flat_codes.size() % config.num_quantizers) == 0) << "OmniVoice audio codes are malformed";

                const int num_frames = (int)(flat_codes.size() / config.num_quantizers);
                std::vector<std::vector<int>> per_codebook(config.num_quantizers, std::vector<int>(num_frames));
                for (int frame = 0; frame < num_frames; frame++)
                    for (int codebook = 0; codebook < config.num_quantizers; codebook++)
                        per_codebook[codebook][frame] = flat_codes[frame * config.num_quantizers + codebook];

                std::vector<int64_t> shape;
                std::vector<uint8_t> buf;
                std::vector<ggml::tensor *> code_tensors;
                code_tensors.reserve(config.num_quantizers);
                eval.evaluate(gen_config,
                    [&](ComputeContext *ctx) -> ggml::tensor * {
                        code_tensors.clear();
                        for (int i = 0; i < config.num_quantizers; i++)
                            code_tensors.push_back(ggml::new_tensor_1d(ctx, GGML_TYPE_I32, num_frames));
                        ggml::tensor *r = model->decode(ctx, code_tensors);
                        return r;
                    },
                    [&](ComputeContext *ctx) {
                        (void)ctx;
                        for (int i = 0; i < config.num_quantizers; i++)
                            Backend::write_tensor_data(code_tensors[i], per_codebook[i].data());
                    },
                    ggml::type::GGML_TYPE_F32,
                    shape,
                    buf);

                pcm_samples.resize(buf.size() / sizeof(float));
                memcpy(pcm_samples.data(), buf.data(), buf.size());
            }

            int get_sample_rate(void) const
            {
                return config.sample_rate;
            }

            int get_frame_rate(void) const
            {
                return config.frame_rate;
            }

            int get_hop_length(void) const
            {
                return config.hop_length;
            }

            bool is_loaded(void) const
            {
                return loaded && model.get();
            }

        private:
            TensorGraphEvaluator eval;
            InitContext _ctx;
            bool loaded = false;
            DecoderConfig config;
            std::unique_ptr<Model> model;
        };
    }

    class ConditionalGeneration : public qwen::v3::ConditionalGeneration
    {
    private:
        typedef qwen::v3::ConditionalGeneration Base;

        struct SequenceInputs
        {
            std::vector<int> text_ids;
            std::vector<int> audio_ids_shifted;
            std::vector<float> audio_mask;
            int ref_audio_len = 0;
            int target_offset = 0;
            int target_len = 0;
        };

        struct ReferenceConditioning
        {
            std::vector<int> audio_tokens;
            std::string text;
            double rms = -1.0;
        };

    public:
        ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = MODEL_TYPE_OMNIVOICE)
            : Base(config, runtime_config, type, true, 2),
              ov_config(config),
              audio_embeddings(&w_ctx_, config.audio_vocab_size * config.num_audio_codebook, config.hidden_size),
              audio_heads(&w_ctx_, config.hidden_size, config.audio_vocab_size * config.num_audio_codebook, false),
              audio_tokenizer(runtime_config)
        {
            CHATLLM_CHECK(config.num_audio_codebook <= MAX_CODEBOOKS) << "too many OmniVoice codebooks";
            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                CHATLLM_CHECK(!config.layer_is_sparse[i]) << "OmniVoice native path currently supports dense Qwen3 backbones only";
                auto *layer = dynamic_cast<qwen::v3::QWen3Block *>(transformer->get_layer(i));
                CHATLLM_CHECK(layer != nullptr) << "failed to access OmniVoice Qwen3 block";
                layer->attention.causal = false;
            }

            if (auto *steps = dynamic_cast<LMFinalSteps *>(transformer->get_final_steps()))
                steps->set_read_last_n(config.max_length);
        }

        bool load_more(const json::JSON &config_json) override
        {
            bool r = audio_tokenizer.load_more(this->config.dtype, config_json, ov_config.num_audio_codebook);
            CHATLLM_CHECK(r) << "OmniVoice audio tokenizer decode config is missing";
            return true;
        }

        void load(ModelLoader &loader) override
        {
            Base::load(loader);
            audio_embeddings.load("omnivoice.audio_embeddings.", &loader);
            audio_heads.load("omnivoice.audio_heads.", &loader);
            audio_tokenizer.load(loader);
        }

        void set_additional_args(const std::map<std::string, std::string> &args) override
        {
            auto *tok = dynamic_cast<Tokenizer *>(tokenizer);
            CHATLLM_CHECK(tok != nullptr) << "OmniVoice tokenizer is not set";

            tok->language = Tokenizer::normalize_language(utils::get_opt(args, "language", tok->language));
            tok->instruct = Tokenizer::normalize_instruct(utils::get_opt(args, "instruct", tok->instruct));
            tok->ref_audio_file = utils::get_opt(args, "ref_audio_file", tok->ref_audio_file);
            tok->ref_text = utils::get_opt(args, "ref_text", tok->ref_text);

            options.speed = utils::get_opt(args, "speed", options.speed);
            options.duration = utils::get_opt(args, "duration", options.duration);
            options.num_step = utils::get_opt(args, "num_step", options.num_step);
            options.guidance_scale = utils::get_opt(args, "guidance_scale", options.guidance_scale);
            options.t_shift = utils::get_opt(args, "t_shift", options.t_shift);
            options.layer_penalty_factor = utils::get_opt(args, "layer_penalty_factor", options.layer_penalty_factor);
            options.position_temperature = utils::get_opt(args, "position_temperature", options.position_temperature);
            options.class_temperature = utils::get_opt(args, "class_temperature", options.class_temperature);
            options.denoise = utils::get_opt(args, "denoise", options.denoise);
            options.preprocess_prompt = utils::get_opt(args, "preprocess_prompt", options.preprocess_prompt);
            options.postprocess_output = utils::get_opt(args, "postprocess_output", options.postprocess_output);

            if (options.duration <= 0.0)
                options.duration = -1.0;
        }

        void speech_synthesis(const GenerationConfig &gen_config, const std::vector<int> &input_ids,
            std::vector<int16_t> &audio, int &sample_rate, int &channels) override
        {
            auto *tok = dynamic_cast<Tokenizer *>(tokenizer);
            CHATLLM_CHECK(tok != nullptr) << "OmniVoice tokenizer is not set";
            CHATLLM_CHECK(audio_tokenizer.is_loaded()) << "OmniVoice audio tokenizer decode path is not loaded";

            sample_rate = audio_tokenizer.get_sample_rate();
            channels = 1;

            const std::string prompt_text = tok->decode(input_ids);
            const ReferenceConditioning ref = prepare_reference_conditioning(gen_config, *tok);
            const bool has_ref_audio = !ref.audio_tokens.empty();
            const int ref_audio_frames = has_ref_audio ? (int)(ref.audio_tokens.size() / ov_config.num_audio_codebook) : 0;
            int target_len = options.duration > 0.0 ?
                std::max(1, (int)std::round(options.duration * audio_tokenizer.get_frame_rate())) :
                detail::estimate_target_tokens(prompt_text, options.speed, ref.text, ref_audio_frames);

            SequenceInputs cond = build_cond_inputs(prompt_text, ref, target_len, *tok);
            SequenceInputs uncond = build_uncond_inputs(target_len);

            const int total_tokens = ov_config.num_audio_codebook * target_len;
            std::vector<int> tokens(total_tokens, ov_config.audio_mask_id);
            std::vector<int> predicted(total_tokens, ov_config.audio_mask_id);
            std::vector<double> scores(total_tokens, -std::numeric_limits<double>::infinity());

            auto timesteps = detail::get_time_steps(options.num_step + 1, options.t_shift);
            std::vector<int> schedule;
            schedule.reserve(options.num_step);
            int remain = total_tokens;
            for (int step = 0; step < options.num_step; step++)
            {
                int reveal = step == options.num_step - 1 ?
                    remain :
                    std::min((int)std::ceil((double)total_tokens * (timesteps[step + 1] - timesteps[step])), remain);
                schedule.push_back(reveal);
                remain -= reveal;
            }

            std::mt19937 rng(gen_config.get_seed());

            for (int step = 0; step < options.num_step; step++)
            {
                std::vector<float> cond_logits;
                std::vector<float> uncond_logits;
                CHATLLM_CHECK(run_model_logits(gen_config, cond, cond_logits)) << "OmniVoice conditional forward failed";
                CHATLLM_CHECK(run_model_logits(gen_config, uncond, uncond_logits)) << "OmniVoice unconditional forward failed";

                std::fill(scores.begin(), scores.end(), -std::numeric_limits<double>::infinity());

                for (int pos = 0; pos < target_len; pos++)
                {
                    for (int codebook = 0; codebook < ov_config.num_audio_codebook; codebook++)
                    {
                        const int flat_idx = pos * ov_config.num_audio_codebook + codebook;
                        if (tokens[flat_idx] != ov_config.audio_mask_id)
                            continue;

                        const float *cond_ptr = &cond_logits[ov_config.audio_vocab_size * (codebook + ov_config.num_audio_codebook * (cond.target_offset + pos))];
                        const float *uncond_ptr = &uncond_logits[ov_config.audio_vocab_size * (codebook + ov_config.num_audio_codebook * pos)];

                        std::vector<double> cond_log_probs;
                        std::vector<double> uncond_log_probs;
                        std::vector<double> combined_log_probs;
                        if (options.guidance_scale != 0.0)
                        {
                            detail::log_softmax(cond_ptr, ov_config.audio_vocab_size, cond_log_probs);
                            detail::log_softmax(uncond_ptr, ov_config.audio_vocab_size, uncond_log_probs);
                            combined_log_probs.resize(ov_config.audio_vocab_size);
                            for (int v = 0; v < ov_config.audio_vocab_size; v++)
                                combined_log_probs[v] = cond_log_probs[v] + options.guidance_scale * (cond_log_probs[v] - uncond_log_probs[v]);

                            std::vector<float> tmp(ov_config.audio_vocab_size);
                            for (int v = 0; v < ov_config.audio_vocab_size; v++)
                                tmp[v] = (float)combined_log_probs[v];
                            detail::log_softmax(tmp.data(), ov_config.audio_vocab_size, combined_log_probs);
                        }
                        else
                        {
                            detail::log_softmax(cond_ptr, ov_config.audio_vocab_size, combined_log_probs);
                        }

                        combined_log_probs[ov_config.audio_mask_id] = -std::numeric_limits<double>::infinity();

                        int best_token = 0;
                        double best_score = -std::numeric_limits<double>::infinity();
                        if (options.class_temperature > 0.0)
                        {
                            int top_k = std::max(1, (int)std::ceil(ov_config.audio_vocab_size * 0.1));
                            std::vector<int> order(ov_config.audio_vocab_size);
                            std::iota(order.begin(), order.end(), 0);
                            std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                                [&](int a, int b) { return combined_log_probs[a] > combined_log_probs[b]; });
                            for (int i = 0; i < top_k; i++)
                            {
                                int token_id = order[i];
                                double sampled = combined_log_probs[token_id] / options.class_temperature + detail::sample_gumbel(rng);
                                if (sampled > best_score)
                                {
                                    best_score = sampled;
                                    best_token = token_id;
                                }
                            }
                            best_score = combined_log_probs[best_token];
                        }
                        else
                        {
                            for (int v = 0; v < ov_config.audio_vocab_size; v++)
                            {
                                if (combined_log_probs[v] > best_score)
                                {
                                    best_score = combined_log_probs[v];
                                    best_token = v;
                                }
                            }
                        }

                        predicted[flat_idx] = best_token;
                        double score = best_score - ((double)codebook * options.layer_penalty_factor);
                        if (options.position_temperature > 0.0)
                            score = score / options.position_temperature + detail::sample_gumbel(rng);
                        scores[flat_idx] = score;
                    }
                }

                int unfilled = 0;
                for (int token : tokens)
                    if (token == ov_config.audio_mask_id)
                        unfilled++;

                int reveal = std::min(schedule[step], unfilled);
                if (reveal <= 0)
                    continue;

                std::vector<int> order(scores.size());
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + reveal, order.end(),
                    [&](int a, int b) { return scores[a] > scores[b]; });

                for (int i = 0; i < reveal; i++)
                {
                    int index = order[i];
                    if (!std::isfinite(scores[index]))
                        continue;
                    tokens[index] = predicted[index];
                }

                update_target_audio_ids(cond, tokens);
                update_target_audio_ids(uncond, tokens);
            }

            std::vector<float> pcm_samples;
            audio_tokenizer.decode(gen_config, tokens, pcm_samples);
            postprocess_audio(pcm_samples, has_ref_audio, ref.rms);

            audio.resize(pcm_samples.size());
            for (size_t i = 0; i < pcm_samples.size(); i++)
            {
                float x = std::clamp(pcm_samples[i], -1.0f, 1.0f);
                audio[i] = (int16_t)std::lrintf(x * 32767.0f);
            }
        }

    private:
        SequenceInputs build_cond_inputs(const std::string &prompt_text, const ReferenceConditioning &ref, int target_len, const Tokenizer &tok) const
        {
            std::vector<int> style_ids;
            tok.build_style_tokens(options.denoise, style_ids);

            std::vector<int> wrapped_text_ids;
            {
                std::vector<int> text_ids;
                tok.encode(detail::combine_text(prompt_text, ref.text), text_ids);
                tok.wrap_text_tokens(text_ids, wrapped_text_ids);
            }

            SequenceInputs seq;
            seq.ref_audio_len = ref.audio_tokens.empty() ? 0 : (int)(ref.audio_tokens.size() / ov_config.num_audio_codebook);
            seq.target_offset = (int)(style_ids.size() + wrapped_text_ids.size() + seq.ref_audio_len);
            seq.target_len = target_len;
            seq.text_ids.reserve(seq.target_offset + target_len);
            seq.text_ids.insert(seq.text_ids.end(), style_ids.begin(), style_ids.end());
            seq.text_ids.insert(seq.text_ids.end(), wrapped_text_ids.begin(), wrapped_text_ids.end());
            seq.audio_mask.assign(style_ids.size() + wrapped_text_ids.size(), 0.0f);

            for (int i = 0; i < seq.ref_audio_len; i++)
            {
                seq.text_ids.push_back(ov_config.audio_mask_id);
                seq.audio_mask.push_back(1.0f);
            }

            for (int i = 0; i < target_len; i++)
            {
                seq.text_ids.push_back(ov_config.audio_mask_id);
                seq.audio_mask.push_back(1.0f);
            }

            initialize_audio_ids(seq, ref.audio_tokens);
            return seq;
        }

        SequenceInputs build_uncond_inputs(int target_len) const
        {
            SequenceInputs seq;
            seq.ref_audio_len = 0;
            seq.target_offset = 0;
            seq.target_len = target_len;
            seq.text_ids.assign(target_len, ov_config.audio_mask_id);
            seq.audio_mask.assign(target_len, 1.0f);
            initialize_audio_ids(seq, {});
            return seq;
        }

        void initialize_audio_ids(SequenceInputs &seq, const std::vector<int> &ref_audio_tokens) const
        {
            const int total_len = (int)seq.text_ids.size();
            const int ref_offset = seq.target_offset - seq.ref_audio_len;
            CHATLLM_CHECK(ref_audio_tokens.empty() || ((int)ref_audio_tokens.size() == seq.ref_audio_len * ov_config.num_audio_codebook))
                << "invalid OmniVoice reference audio token shape";

            seq.audio_ids_shifted.resize(total_len * ov_config.num_audio_codebook);
            for (int pos = 0; pos < total_len; pos++)
            {
                for (int codebook = 0; codebook < ov_config.num_audio_codebook; codebook++)
                {
                    int raw_id = 0;
                    if ((pos >= ref_offset) && (pos < seq.target_offset) && !ref_audio_tokens.empty())
                    {
                        raw_id = ref_audio_tokens[(pos - ref_offset) * ov_config.num_audio_codebook + codebook];
                    }
                    else if (pos >= seq.target_offset)
                    {
                        raw_id = ov_config.audio_mask_id;
                    }
                    seq.audio_ids_shifted[pos * ov_config.num_audio_codebook + codebook] =
                        codebook * ov_config.audio_vocab_size + raw_id;
                }
            }
        }

        void update_target_audio_ids(SequenceInputs &seq, const std::vector<int> &tokens) const
        {
            for (int pos = 0; pos < seq.target_len; pos++)
            {
                for (int codebook = 0; codebook < ov_config.num_audio_codebook; codebook++)
                {
                    int token = tokens[pos * ov_config.num_audio_codebook + codebook];
                    seq.audio_ids_shifted[(seq.target_offset + pos) * ov_config.num_audio_codebook + codebook] =
                        codebook * ov_config.audio_vocab_size + token;
                }
            }
        }

        ReferenceConditioning prepare_reference_conditioning(const GenerationConfig &gen_config, const Tokenizer &tok)
        {
            ReferenceConditioning ref;
            const bool has_ref_audio = tok.ref_audio_file.size() > 0;
            const bool has_ref_text = tok.ref_text.size() > 0;
            CHATLLM_CHECK(has_ref_audio == has_ref_text)
                << "native OmniVoice voice cloning requires both ref_audio_file and ref_text";
            if (!has_ref_audio)
                return ref;

            std::vector<float> ref_samples;
            CHATLLM_CHECK(audio::load(tok.ref_audio_file.c_str(), ref_samples, audio_tokenizer.get_sample_rate()))
                << "failed to load OmniVoice reference audio `" << tok.ref_audio_file << "`";
            CHATLLM_CHECK(!ref_samples.empty()) << "OmniVoice reference audio is empty";

            double sum_sq = 0.0;
            for (float x : ref_samples)
                sum_sq += (double)x * (double)x;
            ref.rms = std::sqrt(sum_sq / std::max<size_t>(1, ref_samples.size()));
            if ((ref.rms > 0.0) && (ref.rms < 0.1))
            {
                const float scale = (float)(0.1 / ref.rms);
                for (auto &x : ref_samples)
                    x *= scale;
            }

            if (options.preprocess_prompt)
            {
                detail::trim_silence_edges(ref_samples, audio_tokenizer.get_sample_rate(), 100, 200);
                ref.text = detail::add_punctuation(tok.ref_text);
            }
            else
            {
                ref.text = utils::trim(tok.ref_text);
            }

            const int hop_length = audio_tokenizer.get_hop_length();
            CHATLLM_CHECK(hop_length > 0) << "OmniVoice audio tokenizer hop_length is invalid";
            const int clip_size = (int)(ref_samples.size() % (size_t)hop_length);
            if (clip_size > 0)
                ref_samples.resize(ref_samples.size() - clip_size);
            CHATLLM_CHECK(!ref_samples.empty()) << "reference audio is empty after trimming to hop_length";

            audio_tokenizer.encode(gen_config, ref_samples, ref.audio_tokens);
            return ref;
        }

        bool run_model_logits(const GenerationConfig &gen_config, const SequenceInputs &inputs, std::vector<float> &output)
        {
            const int seq_len = (int)inputs.text_ids.size();

            ForwardContext ctx(&backend_context);
            ctx.user_options = w_ctx_.user_options;
            ctx.gctx = GGMLContext({.mem_size = backend_context.buf_compute_meta.size(), .mem_buffer = backend_context.buf_compute_meta.data(), .no_alloc = true});
            ctx.gf = ggml::new_graph_custom(&ctx, GRAPH_SIZE, false);

            set_dbg_ctx(&ctx);
            transformer->set_ctx(seq_len);

            ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
            ggml::tensor *text_ids_tensor = ggml::new_tensor_1d(&ctx, GGML_TYPE_I32, seq_len);
            ggml::tensor *audio_ids_tensor = ggml::new_tensor_2d(&ctx, GGML_TYPE_I32, ov_config.num_audio_codebook, seq_len);
            ggml::tensor *audio_mask_tensor = ggml::new_tensor_2d(&ctx, GGML_TYPE_F32, 1, seq_len);

            const auto custom_embedding = [this, audio_ids_tensor, audio_mask_tensor, seq_len](ComputeContext *ctx, ggml::tensor *input)
            {
                ggml::tensor *text_emb = transformer->word_embeddings->forward(ctx, input);
                text_emb = ggml::reshape_2d(ctx, text_emb, ov_config.hidden_size, seq_len);

                ggml::tensor *audio_emb = audio_embeddings.forward(ctx, audio_ids_tensor);
                audio_emb = ggml::sum(ctx, audio_emb, 1);
                audio_emb = ggml::reshape_2d(ctx, audio_emb, ov_config.hidden_size, seq_len);

                ggml::tensor *mask = ggml::repeat(ctx, audio_mask_tensor, audio_emb);
                ggml::tensor *delta = ggml::sub(ctx, audio_emb, text_emb);
                delta = ggml::mul(ctx, delta, mask);
                return ggml::add(ctx, text_emb, delta);
            };

            transformer->custom_embedding = custom_embedding;
            ggml::tensor *r = transformer->forward(&ctx, text_ids_tensor, 0);

            ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);
            ggml::tensor *hidden_states = transformer->last_hidden_state;
            CHATLLM_CHECK(hidden_states != nullptr) << "OmniVoice hidden states are unavailable";
            r = audio_heads.forward(&ctx, hidden_states);
            r = ggml::reshape_3d(&ctx, r, ov_config.audio_vocab_size, ov_config.num_audio_codebook, seq_len);

            if (ggml::type_of(r) != GGML_TYPE_F32)
            {
                ggml::tensor *t = ggml::new_tensor_like(&ctx, GGML_TYPE_F32, r);
                r = ggml::cpy(&ctx, r, t);
            }

            ggml::set_output(r);
            ggml::build_forward_expand(&ctx, r);

            output.resize(ggml::nbytes(r) / sizeof(float));

            if (!ctx.allocate())
            {
                set_dbg_ctx(nullptr);
                transformer->custom_embedding = nullptr;
                return false;
            }

            Backend::write_tensor_data(text_ids_tensor, inputs.text_ids.data(), 0, inputs.text_ids.size() * sizeof(inputs.text_ids[0]));
            Backend::write_tensor_data(audio_ids_tensor, inputs.audio_ids_shifted.data(), 0, inputs.audio_ids_shifted.size() * sizeof(inputs.audio_ids_shifted[0]));
            Backend::write_tensor_data(audio_mask_tensor, inputs.audio_mask.data(), 0, inputs.audio_mask.size() * sizeof(inputs.audio_mask[0]));

            if (gen_config.dump_dot.size() > 0)
            {
                backend_context.dump_graph(ctx.get_cgraph(), gen_config.dump_dot.c_str());
                exit(-1);
            }

            transformer->before_eval(&ctx);
            ctx.compute();
            Backend::read_tensor_data(r, output.data());

            transformer->custom_embedding = nullptr;
            set_dbg_ctx(nullptr);
            ctx.reset();
            return true;
        }

        void postprocess_audio(std::vector<float> &pcm_samples, bool has_ref_audio, double ref_rms) const
        {
            if (pcm_samples.empty())
                return;

            if (options.postprocess_output)
            {
                if (has_ref_audio)
                {
                    if ((ref_rms > 0.0) && (ref_rms < 0.1))
                    {
                        const float scale = (float)(ref_rms / 0.1);
                        for (auto &x : pcm_samples)
                            x *= scale;
                    }
                }
                else
                {
                    float peak = 0.0f;
                    for (float x : pcm_samples)
                        peak = std::max(peak, std::fabs(x));
                    if (peak > 1e-6f)
                    {
                        float scale = 0.5f / peak;
                        for (auto &x : pcm_samples)
                            x *= scale;
                    }
                }

                const int sample_rate = audio_tokenizer.get_sample_rate();
                const int fade_samples = sample_rate / 10;
                const int pad_samples = sample_rate / 10;
                const int k = std::min(fade_samples, (int)pcm_samples.size() / 2);

                for (int i = 0; i < k; i++)
                {
                    float fade_in = (float)i / (float)k;
                    float fade_out = (float)(k - i) / (float)k;
                    pcm_samples[i] *= fade_in;
                    pcm_samples[pcm_samples.size() - k + i] *= fade_out;
                }

                std::vector<float> padded;
                padded.assign(pad_samples, 0.0f);
                padded.insert(padded.end(), pcm_samples.begin(), pcm_samples.end());
                padded.insert(padded.end(), pad_samples, 0.0f);
                pcm_samples.swap(padded);
            }
        }

    private:
        const Config ov_config;
        Embedding audio_embeddings;
        Linear audio_heads;
        higgs::Generation audio_tokenizer;
        InferenceOptions options;
    };
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(OMNIVOICE, omnivoice, 1);
}
