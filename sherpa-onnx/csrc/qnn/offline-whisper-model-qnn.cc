// sherpa-onnx/csrc/qnn/offline-whisper-model-qnn.cc
//
// Whisper encoder-decoder inference on Qualcomm NPU via QNN.

#include "sherpa-onnx/csrc/qnn/offline-whisper-model-qnn.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/qnn/macros.h"
#include "sherpa-onnx/csrc/qnn/qnn-backend.h"
#include "sherpa-onnx/csrc/qnn/qnn-model.h"

namespace sherpa_onnx {

// Whisper special tokens
static constexpr int32_t SOT = 50258;
static constexpr int32_t EOT = 50257;
static constexpr int32_t TRANSLATE = 50358;
static constexpr int32_t TRANSCRIBE = 50359;
static constexpr int32_t NO_TIMESTAMPS = 50363;
static constexpr int32_t MAX_DECODE_TOKENS = 200;
static constexpr int32_t NUM_LAYERS = 12;
static constexpr int32_t NUM_HEADS = 12;
static constexpr int32_t HEAD_DIM = 64;
static constexpr int32_t ENCODER_SEQ = 1500;
static constexpr int32_t VOCAB_SIZE = 51865;

static int32_t GetLanguageToken(const std::string &lang) {
  static const char *langs[] = {
    "en","zh","de","es","ru","ko","fr","ja","pt","tr",
    "pl","ca","nl","ar","sv","it","id","hi","fi","vi",
    "he","uk","el","ms","cs","ro","da","hu","ta","no",
    "th","ur","hr","bg","lt","la","mi","ml","cy","sk",
    "te","fa","lv","bn","sr","az","sl","kn","et","mk",
    "br","eu","is","hy","ne","mn","bs","kk","sq","sw",
    "gl","mr","pa","si","km","sn","yo","so","af","oc",
    "ka","be","tg","sd","gu","am","yi","lo","uz","fo",
    "ht","ps","tk","nn","mt","sa","lb","my","bo","tl",
    "mg","as","tt","haw","ln","ha","ba","jw","su",nullptr
  };
  for (int i = 0; langs[i]; i++) {
    if (lang == langs[i]) return 50259 + i;
  }
  return 50259; // default English
}

class OfflineWhisperModelQnn::Impl {
 public:
  explicit Impl(const WhisperQnnConfig &config) : config_(config) {
    // Initialize QNN backend
    backend_ = std::make_unique<QnnBackend>(config.backend_lib, config.debug);

    // Load encoder context binary
    SHERPA_ONNX_LOGE("Loading encoder context binary...");
    encoder_ = std::make_unique<QnnModel>(
        config.encoder_context_binary, config.system_lib,
        backend_.get(), BinaryContextTag{}, config.debug);

    if (!encoder_->IsInitialized()) {
      SHERPA_ONNX_LOGE("Failed to load encoder");
      return;
    }
    SHERPA_ONNX_LOGE("Encoder loaded: %d inputs, %d outputs",
        static_cast<int32_t>(encoder_->InputTensorNames().size()),
        static_cast<int32_t>(encoder_->OutputTensorNames().size()));

    // Load decoder context binary
    SHERPA_ONNX_LOGE("Loading decoder context binary...");
    decoder_ = std::make_unique<QnnModel>(
        config.decoder_context_binary, config.system_lib,
        backend_.get(), BinaryContextTag{}, config.debug);

    if (!decoder_->IsInitialized()) {
      SHERPA_ONNX_LOGE("Failed to load decoder");
      return;
    }
    SHERPA_ONNX_LOGE("Decoder loaded: %d inputs, %d outputs",
        static_cast<int32_t>(decoder_->InputTensorNames().size()),
        static_cast<int32_t>(decoder_->OutputTensorNames().size()));

    initialized_ = true;
    SHERPA_ONNX_LOGE("Whisper QNN model initialized successfully");
  }

  bool IsInitialized() const { return initialized_; }

  std::string Run(const float *mel_features, int32_t mel_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) return "";

    // === Step 1: Run encoder ===
    // Input: mel [1, 80, 3000]
    encoder_->SetInputTensorData("input_features", mel_features, mel_size);
    encoder_->Run();

    // Get cross-attention KV cache from encoder output
    // These are passed as inputs to the decoder
    struct KVCache {
      std::vector<float> k;
      std::vector<float> v;
    };
    std::vector<KVCache> cross_cache(NUM_LAYERS);

    for (int i = 0; i < NUM_LAYERS; i++) {
      std::string k_name = "k_cache_cross_" + std::to_string(i);
      std::string v_name = "v_cache_cross_" + std::to_string(i);
      cross_cache[i].k = encoder_->GetOutputTensorData(k_name);
      cross_cache[i].v = encoder_->GetOutputTensorData(v_name);
    }

    SHERPA_ONNX_LOGE("Encoder done, cross-cache[0].k size=%d",
        static_cast<int32_t>(cross_cache[0].k.size()));

    // === Step 2: Run decoder autoregressively ===
    int32_t lang_token = GetLanguageToken(config_.language);
    int32_t task_token = config_.translate ? TRANSLATE : TRANSCRIBE;

    // Initial tokens: <|startoftranscript|> <|lang|> <|task|> <|notimestamps|>
    std::vector<int32_t> tokens = {SOT, lang_token, task_token, NO_TIMESTAMPS};

    // Self-attention KV cache (initialized to zeros)
    int32_t self_cache_size = NUM_HEADS * 1 * HEAD_DIM * (MAX_DECODE_TOKENS - 1);
    std::vector<KVCache> self_cache(NUM_LAYERS);
    for (auto &c : self_cache) {
      c.k.resize(self_cache_size, 0.0f);
      c.v.resize(self_cache_size, 0.0f);
    }

    std::vector<int32_t> output_tokens;

    for (int step = 0; step < MAX_DECODE_TOKENS; step++) {
      // Set decoder inputs
      int32_t current_token = tokens.back();
      decoder_->SetInputTensorData("input_ids", &current_token, 1);

      int32_t position = static_cast<int32_t>(tokens.size()) - 1;
      decoder_->SetInputTensorData("position_ids", &position, 1);

      // Attention mask: 1 for positions <= current, 0 for future
      std::vector<float> attn_mask(MAX_DECODE_TOKENS, 0.0f);
      for (int i = 0; i <= position && i < MAX_DECODE_TOKENS; i++) {
        attn_mask[i] = 1.0f;
      }
      decoder_->SetInputTensorData("attention_mask", attn_mask.data(),
                                    attn_mask.size());

      // Set self-attention KV cache
      for (int i = 0; i < NUM_LAYERS; i++) {
        std::string k_name = "k_cache_self_" + std::to_string(i) + "_in";
        std::string v_name = "v_cache_self_" + std::to_string(i) + "_in";
        decoder_->SetInputTensorData(k_name, self_cache[i].k.data(),
                                      self_cache[i].k.size());
        decoder_->SetInputTensorData(v_name, self_cache[i].v.data(),
                                      self_cache[i].v.size());
      }

      // Set cross-attention KV cache (from encoder, constant each step)
      for (int i = 0; i < NUM_LAYERS; i++) {
        std::string k_name = "k_cache_cross_" + std::to_string(i);
        std::string v_name = "v_cache_cross_" + std::to_string(i);
        decoder_->SetInputTensorData(k_name, cross_cache[i].k.data(),
                                      cross_cache[i].k.size());
        decoder_->SetInputTensorData(v_name, cross_cache[i].v.data(),
                                      cross_cache[i].v.size());
      }

      // Run decoder
      decoder_->Run();

      // Get logits and find argmax
      std::vector<float> logits = decoder_->GetOutputTensorData("logits");
      int32_t next_token = 0;
      float max_logit = -1e30f;
      for (int i = 0; i < VOCAB_SIZE && i < static_cast<int32_t>(logits.size()); i++) {
        if (logits[i] > max_logit) {
          max_logit = logits[i];
          next_token = i;
        }
      }

      if (next_token == EOT) break;
      if (next_token < VOCAB_SIZE) {
        output_tokens.push_back(next_token);
      }

      tokens.push_back(next_token);

      // Update self-attention KV cache from decoder output
      for (int i = 0; i < NUM_LAYERS; i++) {
        std::string k_name = "k_cache_self_" + std::to_string(i) + "_out";
        std::string v_name = "v_cache_self_" + std::to_string(i) + "_out";
        self_cache[i].k = decoder_->GetOutputTensorData(k_name);
        self_cache[i].v = decoder_->GetOutputTensorData(v_name);
      }

      if (step % 10 == 0) {
        SHERPA_ONNX_LOGE("Decoder step %d, token=%d", step, next_token);
      }
    }

    SHERPA_ONNX_LOGE("Decoded %d tokens", static_cast<int32_t>(output_tokens.size()));

    // For now, return token IDs as comma-separated string
    // (Token-to-text conversion will be done in Kotlin using the tokenizer)
    std::string result;
    for (size_t i = 0; i < output_tokens.size(); i++) {
      if (i > 0) result += ",";
      result += std::to_string(output_tokens[i]);
    }
    return result;
  }

 private:
  std::mutex mutex_;
  WhisperQnnConfig config_;

  std::unique_ptr<QnnBackend> backend_;
  std::unique_ptr<QnnModel> encoder_;
  std::unique_ptr<QnnModel> decoder_;

  bool initialized_ = false;
};

OfflineWhisperModelQnn::OfflineWhisperModelQnn(const WhisperQnnConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineWhisperModelQnn::~OfflineWhisperModelQnn() = default;

std::string OfflineWhisperModelQnn::Run(const float *mel_features,
                                         int32_t mel_size) const {
  return impl_->Run(mel_features, mel_size);
}

bool OfflineWhisperModelQnn::IsInitialized() const {
  return impl_->IsInitialized();
}

}  // namespace sherpa_onnx
