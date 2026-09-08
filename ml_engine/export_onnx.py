import os
import torch
import torch.onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from aasist_graph_classifier import SwarVedAASISTGraphClassifier

def export_and_quantize():
    print("📦 Step 6.1: Initializing PyTorch Model Export to ONNX...")
    model = SwarVedAASISTGraphClassifier(in_dim=768, d_model=256, n_heads=8, num_layers=4, num_classes=2)
    checkpoint_path = "ml_engine/models/aasist_best.pt"

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"   Loaded trained SSL-AASIST model weights from {checkpoint_path}")
    else:
        print(f"   No checkpoint found at {checkpoint_path}")

    model.eval()

    # Dummy input representing pre-extracted Wav2Vec2/WavLM 768-dim embeddings: (Batch=1, Time=149, Dim=768)
    dummy_input = torch.randn(1, 149, 768)
    onnx_file = "ml_engine/models/swarved_voice_guard.onnx"
    quantized_file = "ml_engine/models/swarved_voice_guard_int8.onnx"

    os.makedirs("ml_engine/models", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        input_names=["wav2vec2_embeddings"],
        output_names=["logits"],
        dynamic_axes={"wav2vec2_embeddings": {0: "batch_size", 1: "seq_len"}, "logits": {0: "batch_size"}},
        opset_version=14
    )
    print(f"✅ ONNX Graph Exported: {onnx_file} ({os.path.getsize(onnx_file) / (1024*1024):.2f} MB)")

    print("⚡ Step 6.2: Applying Int8 Dynamic Quantization...")
    quantize_dynamic(
        model_input=onnx_file,
        model_output=quantized_file,
        weight_type=QuantType.QUInt8
    )

    quantized_size_mb = os.path.getsize(quantized_file) / (1024 * 1024)
    print(f"🎉 INT8 Quantization Complete! Saved to {quantized_file} ({quantized_size_mb:.2f} MB)")

if __name__ == "__main__":
    export_and_quantize()


