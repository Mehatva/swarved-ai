#include <jni.h>
#include <android/log.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr const char* kLogTag = "SwarVedNative";
constexpr size_t kFrameSamples = 48000;

class VoiceGuardEngine final {
public:
    bool Initialize(const char* model_path) {
        std::scoped_lock lock(mutex_);
        try {
            session_.reset();
            error_.clear();
            Ort::SessionOptions options;
            options.SetIntraOpNumThreads(1);
            options.SetInterOpNumThreads(1);
            options.SetExecutionMode(ORT_SEQUENTIAL);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            auto session = std::make_unique<Ort::Session>(environment_, model_path, options);
            if (session->GetInputCount() != 1 || session->GetOutputCount() != 1) {
                SetErrorLocked("Model must expose exactly one input and one output tensor.");
                return false;
            }
            auto input_type = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            if (input_type.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                SetErrorLocked("Model input tensor must be float32.");
                return false;
            }
            const auto input_shape = input_type.GetShape();
            // CONFIRMED per ML team spec: float32 raw PCM has dynamic [batch_size, num_samples]
            // dimensions; this app supplies one 48,000-sample, 16 kHz window per inference.
            if (input_shape.size() != 2 ||
                (input_shape[0] != -1 && input_shape[0] != 1) ||
                (input_shape[1] != -1 && input_shape[1] != static_cast<int64_t>(kFrameSamples))) {
                SetErrorLocked("Model input contract must be [batch_size, num_samples] float32.");
                return false;
            }
            auto output_type = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
            if (output_type.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                SetErrorLocked("Model output tensor must be float32.");
                return false;
            }

            // MUST VERIFY: bundled ONNX Runtime must be version ~1.13+ for these Allocated APIs.
            Ort::AllocatorWithDefaultOptions allocator;
            input_name_ = session->GetInputNameAllocated(0, allocator).get();
            output_name_ = session->GetOutputNameAllocated(0, allocator).get();
            if (input_name_.empty() || output_name_.empty()) {
                SetErrorLocked("Model tensor names could not be read.");
                return false;
            }
            input_buffer_.assign(kFrameSamples, 0.0F);
            session_ = std::move(session);
            return true;
        } catch (const Ort::Exception& exception) {
            SetErrorLocked(exception.what());
        } catch (const std::exception& exception) {
            SetErrorLocked(exception.what());
        }
        return false;
    }

    float Infer(JNIEnv* env, jshortArray frame) {
        std::scoped_lock lock(mutex_);
        if (session_ == nullptr) {
            SetErrorLocked("Inference requested before model initialization.");
            return NAN;
        }
        if (frame == nullptr || env->GetArrayLength(frame) != static_cast<jsize>(kFrameSamples)) {
            SetErrorLocked("Expected exactly 48,000 PCM samples.");
            return NAN;
        }
        jshort* pcm = env->GetShortArrayElements(frame, nullptr);
        if (pcm == nullptr) {
            SetErrorLocked("Unable to access PCM samples.");
            return NAN;
        }
        PrepareModelInput(pcm, input_buffer_);
        env->ReleaseShortArrayElements(frame, pcm, JNI_ABORT);

        try {
            const std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(kFrameSamples)};
            const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_buffer_.data(), input_buffer_.size(), input_shape.data(), input_shape.size());
            const char* input_names[] = {input_name_.c_str()};
            const char* output_names[] = {output_name_.c_str()};
            auto outputs = session_->Run(
                Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
            if (outputs.size() != 1 || !outputs[0].IsTensor()) {
                SetErrorLocked("Model did not return a tensor output.");
                return NAN;
            }
            const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
            const size_t output_count = output_info.GetElementCount();
            if (output_count == 0) {
                SetErrorLocked("Model returned an empty output tensor.");
                return NAN;
            }
            // CONFIRMED per ML team spec: output is raw [batch_size, 2] logits;
            // index 0 is bonafide/human and index 1 is synthetic/deepfake.
            return ProbabilityFromOutput(outputs[0].GetTensorData<float>(), output_count);
        } catch (const Ort::Exception& exception) {
            SetErrorLocked(exception.what());
        } catch (const std::exception& exception) {
            SetErrorLocked(exception.what());
        }
        return NAN;
    }

    void Shutdown() {
        std::scoped_lock lock(mutex_);
        session_.reset();
        input_buffer_.clear();
        input_name_.clear();
        output_name_.clear();
    }

private:
    // ASSUMED: direct normalized raw PCM is the model feature representation.
    // Replace only this function when the confirmed model contract requires MFCC or log-mel features.
    static void PrepareModelInput(const jshort* pcm, std::vector<float>& input_buffer) {
        float max_abs = 0.0001F;
        for (size_t index = 0; index < kFrameSamples; ++index) {
            float val = static_cast<float>(pcm[index]) / 32768.0F;
            input_buffer[index] = val;
            if (std::abs(val) > max_abs) {
                max_abs = std::abs(val);
            }
        }
        // Nominal Peak Scaling to 0.8 (maps any microphone audio level into neural network's optimal dynamic range)
        if (max_abs > 0.0001F) {
            float scale = 0.8F / max_abs;
            for (size_t index = 0; index < kFrameSamples; ++index) {
                input_buffer[index] *= scale;
            }
        }
    }

    static float ProbabilityFromOutput(const float* output, size_t count) {
        if (count == 1) {
            const float value = output[0];
            return value >= 0.0F && value <= 1.0F ? value : 1.0F / (1.0F + std::exp(-value));
        }
        const float synthetic = output[1];
        const float human = output[0];
        const float maximum = std::max(human, synthetic);
        const float synthetic_exp = std::exp(synthetic - maximum);
        const float human_exp = std::exp(human - maximum);
        return synthetic_exp / (human_exp + synthetic_exp);
    }

    void SetErrorLocked(const std::string& message) {
        error_ = message;
        __android_log_print(ANDROID_LOG_ERROR, kLogTag, "%s", error_.c_str());
    }

    std::mutex mutex_;
    Ort::Env environment_{ORT_LOGGING_LEVEL_WARNING, kLogTag};
    std::unique_ptr<Ort::Session> session_;
    std::vector<float> input_buffer_;
    std::string input_name_;
    std::string output_name_;
    std::string error_;
};

VoiceGuardEngine g_engine;

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_swarved_guard_audio_NativeVoiceGuard_nativeInitialize(
    JNIEnv* env, jobject /* thiz */, jstring model_path) {
    if (model_path == nullptr) return JNI_FALSE;
    const char* path = env->GetStringUTFChars(model_path, nullptr);
    if (path == nullptr) return JNI_FALSE;
    const bool initialized = g_engine.Initialize(path);
    env->ReleaseStringUTFChars(model_path, path);
    return initialized ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jfloat JNICALL
Java_com_swarved_guard_audio_NativeVoiceGuard_nativeInferPcm16(
    JNIEnv* env, jobject /* thiz */, jshortArray frame) {
    return g_engine.Infer(env, frame);
}

extern "C" JNIEXPORT void JNICALL
Java_com_swarved_guard_audio_NativeVoiceGuard_nativeShutdown(
    JNIEnv* /* env */, jobject /* thiz */) {
    g_engine.Shutdown();
}

JNIEXPORT jint JNI_OnLoad(JavaVM* /* vm */, void* /* reserved */) {
    return JNI_VERSION_1_6;
}
