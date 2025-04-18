import tomllib

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import v2
from tqdm import tqdm


def extract(
    df: pd.DataFrame,
    transform: v2.Compose,
    model: torch.nn.Module,
    device: str,
) -> np.ndarray:
    model_extractor = create_feature_extractor(
        model, return_nodes={"classifier.3": "semantic_feature"}
    )

    semantic_feature_list = []

    for img_path in tqdm(df["图像路径"], desc="Extracting"):
        with Image.open(img_path) as img_pil:
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            with torch.inference_mode():
                pred_logits = model_extractor(img_tensor)
                semantic_feature_list.append(
                    pred_logits["semantic_feature"].squeeze(0).detach().cpu().numpy()
                )

    return np.array(semantic_feature_list)


def main():
    with open("./config/path.toml", "rb") as f:
        path = tomllib.load(f)

    df = pd.read_csv(path["log"]["val_pred"])

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(path["model"], map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE).eval()

    semantic_feature_array = extract(
        df=df, transform=transform, model=model, device=DEVICE
    )

    print(f"Extracted feature shape: {semantic_feature_array.shape}")

    np.save(path["npy"]["feature"], semantic_feature_array)


if __name__ == "__main__":
    main()
