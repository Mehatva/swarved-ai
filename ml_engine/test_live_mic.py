import os
import sys
import time
import numpy as np
import onnxruntime as ort
import sounddevice as sd

MODEL_PATH = "ml_engine/models/swarved_e2e_raw_pcm_int8.onnx"
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 48000 # 3 seconds @ 16kHz
STRIDE_SAMPLES = 16000 # 1 second update stride

def run_live_mic_guard():
    print("\n" + "="*65)
    print("🛡️ SWARVED AI: LIVE MAC MICROPHONE DEEPFAKE DETECTOR")
    print("="*65)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: ONNX Model not found at {MODEL_PATH}")
        return

    print(f"📦 Loading Edge ONNX Model: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/(1024*1024):.2f} MB)")
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)

    print("🎙️ Starting live microphone stream...")
    print("👉 Speak into your Mac mic or play sample audio to test live detection.")
    print("   Press Ctrl+C to stop.\n")
    print("-" * 65)

    def audio_callback(indata, frames, time_info, status):
        nonlocal buffer
        if status:
            pass
        samples = indata[:, 0]
        buffer = np.roll(buffer, -len(samples))
        buffer[-len(samples):] = samples

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=STRIDE_SAMPLES, callback=audio_callback):
        try:
            sample_count = 0
            while True:
                time.sleep(1.0)
                sample_count += 1

                # Check audio energy (RMS)
                pcm = buffer.copy()
                rms = np.sqrt(np.mean(pcm ** 2))
                max_val = np.max(np.abs(pcm))

                # Audio volume meter bar
                mic_bar_len = int(min(rms * 200, 20))
                mic_bar = "█" * mic_bar_len + "░" * (20 - mic_bar_len)

                # Silence threshold for Mac laptop mics (RMS < 0.0015 is silence)
                if rms < 0.0015:
                    sys.stdout.write(
                        f"\r[{sample_count:03d}s] Mic Volume: [{mic_bar}] | Real Voice: [░░░░░░░░░░░░░░░░░░░░]  --- % | ⏸️ SILENT / NO SPEECH                          "
                    )
                    sys.stdout.flush()
                    continue

                # Safe Peak Normalization
                if max_val > 0.01:
                    pcm = pcm / max_val

                input_data = np.expand_dims(pcm, axis=0).astype(np.float32)
                logits = session.run([output_name], {input_name: input_data})[0]

                exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))[0]

                real_prob = probs[0] * 100.0
                fake_prob = probs[1] * 100.0

                # Risk meter bar
                real_bar_len = int(real_prob / 5)
                real_bar = "█" * real_bar_len + "░" * (20 - real_bar_len)

                if real_prob >= 50.0:
                    status_str = f"✅ SAFE (GENUINE VOICE - {real_prob:.1f}%)"
                else:
                    status_str = f"🚨 DANGER (SYNTHETIC DEEPFAKE - {fake_prob:.1f}%)"

                sys.stdout.write(
                    f"\r[{sample_count:03d}s] Mic Volume: [{mic_bar}] | Real Voice: [{real_bar}] {real_prob:5.1f}% | {status_str:<42}"
                )
                sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n\n🛑 Live microphone detection stopped.")
            print("="*65 + "\n")

if __name__ == "__main__":
    run_live_mic_guard()
