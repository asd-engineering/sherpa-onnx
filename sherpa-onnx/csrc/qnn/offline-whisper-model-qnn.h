// sherpa-onnx/csrc/qnn/offline-whisper-model-qnn.h
//
// Whisper ASR on Qualcomm NPU (Hexagon HTP) via QNN context binaries.

#ifndef SHERPA_ONNX_CSRC_QNN_OFFLINE_WHISPER_MODEL_QNN_H_
#define SHERPA_ONNX_CSRC_QNN_OFFLINE_WHISPER_MODEL_QNN_H_

#include <memory>
#include <string>
#include <vector>

namespace sherpa_onnx {

struct WhisperQnnConfig {
  std::string encoder_context_binary;  // Path to encoder .bin
  std::string decoder_context_binary;  // Path to decoder .bin
  std::string backend_lib;             // Path to libQnnHtp.so
  std::string system_lib;              // Path to libQnnSystem.so
  std::string language = "en";
  bool translate = false;
  bool debug = false;
};

class OfflineWhisperModelQnn {
 public:
  ~OfflineWhisperModelQnn();
  explicit OfflineWhisperModelQnn(const WhisperQnnConfig &config);

  /**
   * @param mel_features Float array of shape [1, 80, 3000] (30s audio)
   * @return Transcribed text
   */
  std::string Run(const float *mel_features, int32_t mel_size) const;

  bool IsInitialized() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_QNN_OFFLINE_WHISPER_MODEL_QNN_H_
