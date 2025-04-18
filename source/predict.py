import json
import tomllib
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm


def load_data(dir_path: Path, transform: v2.Compose) -> Tuple[pd.DataFrame, List[str]]:
    dataset = datasets.ImageFolder(root=dir_path, transform=transform)
    classes_list = dataset.classes
    idx_to_class = {value: key for key, value in dataset.class_to_idx.items()}

    with open("./output/log/classes_list.json", "w") as f:
        json.dump(classes_list, f)
    with open("./output/log/class_to_idx.json", "w") as f:
        json.dump(dataset.class_to_idx, f)
    with open("./output/log/idx_to_class.json", "w") as f:
        json.dump(idx_to_class, f)

    dataset_df = pd.DataFrame(
        {
            "图像路径": [img[0] for img in dataset.imgs],
            "类别 ID": dataset.targets,
            "类别名称": [classes_list[ID] for ID in dataset.targets],
        }
    )

    return dataset_df, classes_list


def predict(
    df: pd.DataFrame,
    classes_list: List[str],
    model: torch.nn.Module,
    device: str,
    transform: v2.Compose,
    k: int = 3,
) -> pd.DataFrame:
    range_k = range(1, k + 1)
    pred_list = []

    for img_path in tqdm(df["图像路径"], desc="Predicting"):
        with Image.open(img_path) as img_pil:
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            with torch.inference_mode():
                pred_softmax = F.softmax(model(img_tensor), dim=1)

        top_k = torch.topk(pred_softmax, k)
        pred_ids = top_k[1].cpu().numpy().squeeze()

        pred_list.append(
            {
                **{f"top-{i}-预测ID": pred_ids[i - 1] for i in range_k},
                **{f"top-{i}-预测名称": classes_list[pred_ids[i - 1]] for i in range_k},
                **{
                    f"{cls}-预测置信度": pred_softmax[0][idx].item()
                    for idx, cls in enumerate(classes_list)
                },
            }
        )

    return pd.concat([df, pd.DataFrame(pred_list)], axis=1)


def main():
    with open("./config/path.toml", "rb") as f:
        path = tomllib.load(f)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(path["model"], map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE).eval()

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_path = Path("./data/Ti5_split/val_aug")
    val_dataset_df, classes_list = load_data(val_path, transform)

    df = predict(
        df=val_dataset_df,
        classes_list=classes_list,
        model=model,
        device=DEVICE,
        transform=transform,
        k=3,
    )
    df.to_csv(path["log"]["val_pred"], index=False)

    print(
        f"Top-1 准确率: {(sum(df['类别名称'] == df['top-1-预测名称']) / len(df)):.12f}"
    )


if __name__ == "__main__":
    main()
