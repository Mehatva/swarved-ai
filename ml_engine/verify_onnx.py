import os
import time
import numpy as np
import onnxruntime as ort

def verify_onnx_model(onnx_file="ml_engine/models/swarved_voice_guard_int8.onnx"):
    print(f"📊 Step 7.1: Verifying Quantized ONNX Model ({onnx_file})")

    if not os.path.exists(onnx_file):
        print(f"❌ Error: ONNX file not found at {onnx_file}")
        return

    # Check File Size
    size_mb = os.path.getsize(onnx_file) / (1024 * 1024)
    print(f"   Model File Size: {size_mb:.2f} MB (Target: <15.0 MB) -> {'✅ PASS' if size_mb < 15.0 else '❌ FAIL'}")

    # Initialize ONNX Runtime Session
    session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("\n⚡ Step 7.2: High-Precision Latency Benchmark")

    # Benchmarking latency with 100 runs
    latencies = []
    dummy_input = np.random.randn(1, 149, 768).astype(np.float32)

    for _ in range(100):
        start = time.perf_counter()
        logits = session.run([output_name], {input_name: dummy_input})[0]
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        latencies.append((time.perf_counter() - start) * 1000)

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    print(f"   Average Inference Latency: {avg_latency:.2f} ms (Target: <50.0 ms) -> {'✅ PASS' if avg_latency < 50.0 else '❌ FAIL'}")
    print(f"   P95 Inference Latency: {p95_latency:.2f} ms")
    print(f"   Sample Output Probabilities (Class 0 Real vs Class 1 Deepfake): {probs[0]}")

if __name__ == "__main__":
    verify_onnx_model()


