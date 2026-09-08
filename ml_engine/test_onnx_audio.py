import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import onnxruntime as ort

def test_onnx_model_on_audio(audio_path, model_path="ml_engine/models/swarved_e2e_raw_pcm_int8.onnx"):
    print("\n" + "="*60)
    print("🛡️ SWARVED AI: TESTING ONNX EDGE MODEL ON AUDIO FILE")
    print("="*60)

    if not os.path.exists(model_path):
        print(f"❌ Error: ONNX Model file not found at {model_path}")
        return

    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found at {audio_path}")
        return

    print(f"📁 Audio Input File : {audio_path}")
    print(f"📦 ONNX Model File  : {model_path} ({os.path.getsize(model_path) / (1024*1024):.2f} MB)")

    # 1. Load audio with torchaudio
    wav, sr = torchaudio.load(audio_path)
    total_sec = wav.shape[-1] / float(sr)
    print(f"   Original SR: {sr} Hz, Duration: {total_sec:.2f}s, Shape: {wav.shape}")

    # Resample to 16kHz
    if sr != 16000:
        resampler = T.Resample(sr, 16000)
        wav = resampler(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Standard Peak Normalization to [-1.0, 1.0]
    max_val = torch.abs(wav).max()
    if max_val > 0:
        wav = wav / max_val

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    target_len = 48000 # 3 seconds @ 16kHz
    num_samples = wav.shape[-1]

    # Handle short audio < 3s by padding
    if num_samples <= target_len:
        pad = target_len - num_samples
        padded_wav = F.pad(wav, (0, pad))
        windows = [padded_wav]
    else:
        # Slide 3-second windows across full recording with 50% overlap (1.5s stride)
        stride = 24000
        windows = []
        for start in range(0, num_samples - target_len + 1, stride):
            windows.append(wav[:, start : start + target_len])
        if len(windows) == 0:
            windows = [wav[:, :target_len]]

    print(f"   Analyzing {len(windows)} sliding window segment(s) across audio...")

    real_probs = []
    fake_probs = []

    for w in windows:
        input_pcm = w.numpy().astype(np.float32)
        logits = session.run([output_name], {input_name: input_pcm})[0]

        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))[0]
        real_probs.append(probs[0])
        fake_probs.append(probs[1])

    avg_real = np.mean(real_probs) * 100.0
    avg_fake = np.mean(fake_probs) * 100.0
    median_real = np.median(real_probs) * 100.0
    majority_real_cnt = sum(p > 0.5 for p in real_probs)
    majority_pct = (majority_real_cnt / len(real_probs)) * 100.0

    is_deepfake = majority_pct < 50.0

    print("\n" + "-"*60)
    print(f"🎙️  Median Real Voice Confidence     : {median_real:.2f}%")
    print(f"🎙️  Average Real Voice Confidence    : {avg_real:.2f}%")
    print(f"📊 Real Voice Frame Ratio           : {majority_real_cnt}/{len(real_probs)} frames ({majority_pct:.1f}%)")
    print(f"🤖 Average Deepfake Risk Confidence : {avg_fake:.2f}%")
    print("-"*60)

    if is_deepfake:
        print("🚨 VERDICT: [DANGER] SYNTHETIC DEEPFAKE VOICE SCAM DETECTED!")
    else:
        print("✅ VERDICT: [SAFE] GENUINE REAL HUMAN VOICE CONFIRMED")
    print("="*60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_onnx_model_on_audio(sys.argv[1])
    else:
        sample_files = [
            f for f in os.popen("find ml_engine/data/asvspoof -name '*.flac' | head -n 1").read().strip().split("\n") if f
        ]
        if sample_files and os.path.exists(sample_files[0]):
            test_onnx_model_on_audio(sample_files[0])
