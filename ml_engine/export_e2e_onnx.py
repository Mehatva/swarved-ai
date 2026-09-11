import os
import torch
import torch.onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from train_dual_stream import SwarVedDualStreamEngine

def export_e2e_raw_pcm_onnx():
    print("📦 Exporting Fine-Tuned Dual-Stream OPUS Engine to ONNX for Android...")
    device = torch.device("cpu")
    model = SwarVedDualStreamEngine(unfreeze_layers=4)
    checkpoint_path = "ml_engine/models/swarved_dual_stream_noise_robust.pt"

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
        print(f"   Loaded noise-robust dual-stream weights from {checkpoint_path}")
    else:
        print(f"   Warning: Checkpoint not found at {checkpoint_path}")

    model.eval()

    # Dummy raw 16kHz PCM audio waveform input (1, 48000) = 3 seconds @ 16kHz float32 in [-1, 1]
    dummy_input = torch.randn(1, 48000)
    onnx_file = "ml_engine/models/swarved_noise_robust.onnx"
    quantized_file = "ml_engine/models/swarved_noise_robust_int8.onnx"

    os.makedirs("ml_engine/models", exist_ok=True)

    print("   Exporting ONNX computational graph...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        input_names=["raw_audio_pcm"],
        output_names=["logits"],
        dynamic_axes={"raw_audio_pcm": {0: "batch_size", 1: "num_samples"}, "logits": {0: "batch_size"}},
        opset_version=17
    )
    print(f"✅ Full End-to-End Raw PCM ONNX Exported: {onnx_file} ({os.path.getsize(onnx_file) / (1024*1024):.2f} MB)")

    print("⚡ Applying Int8 Dynamic Quantization (MatMul & Gemm)...")
    quantize_dynamic(
        model_input=onnx_file,
        model_output=quantized_file,
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8
    )

    size_mb = os.path.getsize(quantized_file) / (1024 * 1024)
    print(f"🎉 End-to-End INT8 ONNX Ready for Android! Saved to {quantized_file} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    export_e2e_raw_pcm_onnx()
