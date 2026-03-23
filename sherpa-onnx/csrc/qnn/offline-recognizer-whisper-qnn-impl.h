// sherpa-onnx/csrc/qnn/offline-recognizer-whisper-qnn-impl.h
//
// Whisper ASR on Qualcomm NPU via QNN context binaries.
#ifndef SHERPA_ONNX_CSRC_QNN_OFFLINE_RECOGNIZER_WHISPER_QNN_IMPL_H_
#define SHERPA_ONNX_CSRC_QNN_OFFLINE_RECOGNIZER_WHISPER_QNN_IMPL_H_

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-stream.h"
#include "sherpa-onnx/csrc/offline-whisper-model-config.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/qnn/offline-whisper-model-qnn.h"

namespace sherpa_onnx {

class OfflineRecognizerWhisperQnnImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerWhisperQnnImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens) {
    symbol_table_.ApplyBase64Decode();

    // Build QNN config from model config
    WhisperQnnConfig qnn_config;
    qnn_config.backend_lib = config.model_config.whisper.qnn_config.backend_lib;
    qnn_config.system_lib = config.model_config.whisper.qnn_config.system_lib;
    qnn_config.language = config.model_config.whisper.language;
    qnn_config.translate = (config.model_config.whisper.task == "translate");
    qnn_config.debug = config.model_config.debug;

    // Context binary can be encoder path (we load both encoder+decoder)
    qnn_config.encoder_context_binary =
        config.model_config.whisper.qnn_config.context_binary;

    // Decoder context binary: replace "Encoder" with "Decoder" in path
    std::string dec_path = qnn_config.encoder_context_binary;
    auto pos = dec_path.find("Encoder");
    if (pos != std::string::npos) {
      dec_path.replace(pos, 7, "Decoder");
    }
    qnn_config.decoder_context_binary = dec_path;

    model_ = std::make_unique<OfflineWhisperModelQnn>(qnn_config);

    if (!model_->IsInitialized()) {
      SHERPA_ONNX_LOGE("Failed to initialize Whisper QNN model");
    }
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    WhisperTag tag;
    tag.dim = 80;  // Whisper mel feature dim
    return std::make_unique<OfflineStream>(tag);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i != n; ++i) {
      DecodeStream(ss[i]);
    }
  }

  void SetConfig(const OfflineRecognizerConfig &config) override {
    config_.model_config.whisper = config.model_config.whisper;
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void DecodeStream(OfflineStream *s) const {
    if (!model_->IsInitialized()) {
      SHERPA_ONNX_LOGE("QNN model not initialized");
      return;
    }

    int32_t feat_dim = s->FeatureDim();
    std::vector<float> f = s->GetFrames();
    int32_t num_frames = f.size() / feat_dim;

    // Pad/truncate to 3000 frames (30 seconds)
    int32_t max_frames = 3000;
    std::vector<float> mel(max_frames * feat_dim, 0.0f);

    int32_t frames_to_copy = std::min(num_frames, max_frames);
    std::copy(f.data(), f.data() + frames_to_copy * feat_dim, mel.data());

    // Transpose from [frames, feat_dim] to [1, feat_dim, frames]
    // QNN encoder expects [1, 80, 3000]
    std::vector<float> mel_transposed(feat_dim * max_frames);
    for (int32_t i = 0; i < feat_dim; i++) {
      for (int32_t j = 0; j < max_frames; j++) {
        mel_transposed[i * max_frames + j] = mel[j * feat_dim + i];
      }
    }

    // Run encoder + decoder on NPU
    std::string token_ids_str = model_->Run(mel_transposed.data(),
                                             mel_transposed.size());

    // Parse comma-separated token IDs from model output
    std::vector<int32_t> tokens;
    std::istringstream iss(token_ids_str);
    std::string tok;
    while (std::getline(iss, tok, ',')) {
      if (!tok.empty()) {
        int32_t id = std::stoi(tok);
        if (id < 50257) {  // Skip special tokens
          tokens.push_back(id);
        }
      }
    }

    // Convert tokens to text
    OfflineRecognitionResult r;
    std::string text;
    for (auto id : tokens) {
      if (symbol_table_.Contains(id)) {
        std::string s = symbol_table_[id];
        text += s;
        r.tokens.push_back(std::move(s));
      }
    }
    r.text = std::move(text);
    s->SetResult(r);
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineWhisperModelQnn> model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_QNN_OFFLINE_RECOGNIZER_WHISPER_QNN_IMPL_H_
