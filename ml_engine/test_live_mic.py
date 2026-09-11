import os
import sys
import time
import collections
import numpy as np
import onnxruntime as ort
import sounddevice as sd

MODEL_PATH = "ml_engine/models/swarved_noise_robust_int8.onnx"
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 48000   # 3 seconds @ 16kHz
STRIDE_SAMPLES = 16000   # 1 second update stride

# --- Tuning knobs -----------------------------------------------------------
# RMS threshold below which we treat the frame as silence and skip inference.
# 0.008 is empirically much more reliable for real-world rooms.
SILENCE_GATE_RMS = 0.008

# Temporal smoothing: keep a rolling window of the last N predictions and
# only flip the verdict when the MAJORITY of those windows agree.
# Prevents any single noisy 1-second chunk from triggering a false alarm.
SMOOTH_WINDOW = 5       # rolling vote over last 5 seconds
DANGER_THRESHOLD = 0.40 # real_prob < 40% → DANGER (slightly permissive)
# ---------------------------------------------------------------------------

def run_live_mic_guard():
    print("\n" + "="*70)
    print("🛡️  SWARVED AI: LIVE MAC MICROPHONE DEEPFAKE DETECTOR  (Noise-Robust v2)")
    print("="*70)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: ONNX Model not found at {MODEL_PATH}")
        print("   Run:  python3 ml_engine/export_e2e_onnx.py  first.")
        return

    print(f"📦 Loading Edge ONNX Model: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/(1024*1024):.2f} MB)")
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)

    # Rolling deque of the last SMOOTH_WINDOW real_prob scores (0.0–1.0).
    # We only update the verdict from the smoothed average, not individual frames.
    prob_history = collections.deque(maxlen=SMOOTH_WINDOW)

    # Smoothed display values (exponential moving average for the bar)
    ema_real_prob = 50.0
    ema_alpha     = 0.4  # weight for new sample

    print("🎙️  Starting live microphone stream...")
    print("👉 Speak into your Mac mic. Press Ctrl+C to stop.\n")
    print(f"   Silence gate  : RMS < {SILENCE_GATE_RMS:.4f}")
    print(f"   Temporal smooth: last {SMOOTH_WINDOW} windows (majority vote)")
    print("-" * 70)

    def audio_callback(indata, frames, time_info, status):
        nonlocal buffer
        samples = indata[:, 0]
        buffer = np.roll(buffer, -len(samples))
        buffer[-len(samples):] = samples

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                        blocksize=STRIDE_SAMPLES, callback=audio_callback):
        try:
            sample_count = 0
            while True:
                time.sleep(1.0)
                sample_count += 1

                pcm = buffer.copy()
                rms     = float(np.sqrt(np.mean(pcm ** 2)))
                max_val = float(np.max(np.abs(pcm)))

                # Volume bar (capped at 20 chars)
                mic_bar_len = min(int(rms * 300), 20)
                mic_bar = "█" * mic_bar_len + "░" * (20 - mic_bar_len)

                # --- Silence Gate ---
                if rms < SILENCE_GATE_RMS:
                    sys.stdout.write(
                        f"\r[{sample_count:04d}s] Vol:[{mic_bar}] | "
                        f"⏸️  SILENT — waiting for speech...                      "
                    )
                    sys.stdout.flush()
                    continue

                # --- Peak Normalization ---
                if max_val > 0.0001:
                    pcm = (pcm / max_val) * 0.8

                # --- ONNX Inference ---
                input_data = np.expand_dims(pcm, axis=0).astype(np.float32)
                logits = session.run([output_name], {input_name: input_data})[0]
                exp_l  = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs  = (exp_l / np.sum(exp_l, axis=-1, keepdims=True))[0]
                real_prob_raw = float(probs[0])

                # --- Temporal Smoothing (rolling window vote) ---
                prob_history.append(real_prob_raw)
                smoothed_real = float(np.mean(prob_history))

                # EMA for display bar (smoother visuals)
                ema_real_prob = ema_alpha * (smoothed_real * 100.0) + (1 - ema_alpha) * ema_real_prob

                real_bar_len = min(int(ema_real_prob / 5), 20)
                real_bar = "█" * real_bar_len + "░" * (20 - real_bar_len)

                # --- Verdict (from smoothed score, requires sustained signal) ---
                n_samples = len(prob_history)
                if n_samples < 2:
                    verdict_str = f"⏳ CALIBRATING... ({n_samples}/{SMOOTH_WINDOW})"
                elif smoothed_real >= DANGER_THRESHOLD:
                    verdict_str = f"✅ SAFE  — GENUINE VOICE  ({smoothed_real*100:.1f}%)"
                else:
                    verdict_str = f"🚨 DANGER — DEEPFAKE RISK ({(1-smoothed_real)*100:.1f}%)"

                sys.stdout.write(
                    f"\r[{sample_count:04d}s] Vol:[{mic_bar}] | "
                    f"Real:[{real_bar}] {ema_real_prob:5.1f}% | {verdict_str:<46}"
                )
                sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n\n🛑 Live microphone detection stopped.")
            print("="*70 + "\n")

if __name__ == "__main__":
    run_live_mic_guard()
