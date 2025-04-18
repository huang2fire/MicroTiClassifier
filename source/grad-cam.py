import json
import tomllib
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm

plt.rcParams["font.sans-serif"] = ["Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False


def visual_cam(
    img_pil: Image.Image,
    targets: List[ClassifierOutputTarget],
    model: torch.nn.Module,
    device: str,
) -> Tuple[np.ndarray, torch.return_types.topk]:
    img_np = np.float32(img_pil) / 255
    img_tensor = preprocess_image(
        img=img_np, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ).to(device)

    with GradCAM(model=model, target_layers=[model.features[-1]]) as cam:
        grayscale_cam = cam(
            input_tensor=img_tensor, targets=targets, aug_smooth=True, eigen_smooth=True
        )[0, :]
        visualization = show_cam_on_image(
            img=img_np,
            mask=grayscale_cam,
            use_rgb=True,
            image_weight=0.5,
        )
        model_outputs = F.softmax(input=cam.outputs, dim=1)
        top_k = torch.topk(input=model_outputs, k=5)

    return visualization, top_k


def process_image(
    img_path: Path,
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    model: torch.nn.Module,
    device: str,
) -> None:
    cls, idx = img_path.stem, class_to_idx[f"{img_path.stem}"]

    with Image.open(img_path) as img_pil:
        visualization, top_k = visual_cam(
            img_pil=img_pil,
            targets=[ClassifierOutputTarget(idx)],
            model=model,
            device=device,
        )

    pred_ids = top_k[1].cpu().numpy().squeeze()

    plt.imshow(visualization)
    plt.title(f"True: {cls} Predict: {idx_to_class[pred_ids[0]]}")
    plt.savefig(f"./output/chart/GradCAM-{cls}.png", bbox_inches="tight")
    plt.close()


def main():
    with open("./config/path.toml", "rb") as f:
        path = tomllib.load(f)
    with open("./output/log/class_to_idx.json", "r") as f:
        class_to_idx = json.load(f)
    with open("./output/log/idx_to_class.json", "r") as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}

    dir_path = Path("./assets/image")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(path["model"], map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE).eval()

    for img_path in tqdm(dir_path.iterdir(), desc="CAM"):
        process_image(
            img_path=img_path,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            model=model,
            device=DEVICE,
        )


if __name__ == "__main__":
    main()
