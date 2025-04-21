import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(
    "./output/checkpoints/Ti5.pt", map_location=DEVICE, weights_only=False
).eval()

dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

torch.onnx.export(
    model,
    dummy_input,
    "./output/checkpoints/Ti5.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
