import os
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


def create_dir():
    output_dir = Path("./output")
    dir_dict = {
        "chart": output_dir / "chart",
        "checkpoints": output_dir / "checkpoints",
        "log": output_dir / "log",
    }
    for dir_path in dir_dict.values():
        dir_path.mkdir(parents=True, exist_ok=True)


def load_data(train_path: Path, val_path: Path) -> Tuple[DataLoader, DataLoader]:
    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
            v2.RandomEqualize(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    return train_loader, val_loader


def calculate_epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def print_epoch_info(
    epoch: int, epoch_mins: int, epoch_secs: int, train_log: dict, val_log: dict
) -> None:
    print("=" * 50)
    print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
    print(
        f"Train Loss: {train_log['loss']:.4f} | Train Acc: {train_log['accuracy']:.4f}"
    )
    print(f"Val Loss: {val_log['loss']:.4f} | Val Acc: {val_log['accuracy']:.4f}")
    print("=" * 50)


def train_epoch(
    model: nn.Module,
    iterator: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    model.train()

    epoch_loss = 0
    preds_list, labels_list = [], []

    for images, labels in tqdm(iterator, desc="Training", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(input=outputs, dim=1)
        preds_list.extend(preds.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

    return {
        "loss": epoch_loss / len(iterator),
        "accuracy": accuracy_score(labels_list, preds_list),
    }


def evaluate_epoch(
    model: nn.Module, iterator: DataLoader, criterion: nn.Module, device: str
) -> Dict[str, float]:
    model.eval()

    epoch_loss = 0
    preds_list, labels_list = [], []

    with torch.inference_mode():
        for images, labels in tqdm(iterator, desc="Evaluating", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            _, preds = torch.max(input=outputs, dim=1)
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    return {
        "loss": epoch_loss / len(iterator),
        "accuracy": accuracy_score(labels_list, preds_list),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: str,
    N_EPOCHS: int,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    best_dict = {"val_accuracy": 0, "train_accuracy": 0, "model_path": None}
    log_list = []

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_log_dict = train_epoch(model, train_loader, optimizer, criterion, device)
        val_log_dict = evaluate_epoch(model, val_loader, criterion, device)

        lr_scheduler.step()

        end_time = time.time()

        epoch_mins, epoch_secs = calculate_epoch_time(start_time, end_time)
        print_epoch_info(epoch, epoch_mins, epoch_secs, train_log_dict, val_log_dict)

        update_flag = val_log_dict["accuracy"] > best_dict["val_accuracy"] or (
            val_log_dict["accuracy"] == best_dict["val_accuracy"]
            and train_log_dict["accuracy"] > best_dict["train_accuracy"]
        )

        if update_flag:
            if best_dict["model_path"] is not None:
                os.remove(best_dict["model_path"])

            best_dict.update(
                {
                    "val_accuracy": val_log_dict["accuracy"],
                    "train_accuracy": train_log_dict["accuracy"],
                }
            )

            best_dict["model_path"] = (
                f"./output/checkpoints/Ti5_{timestamp}_{best_dict['val_accuracy']:.3f}.pt"
            )

            torch.save(model, best_dict["model_path"])
            print("已保存新的最佳模型!")

        log_list.append(
            {
                "epoch": epoch,
                "epoch_mins": epoch_mins,
                "epoch_secs": epoch_secs,
                "train_loss": train_log_dict["loss"],
                "train_accuracy": train_log_dict["accuracy"],
                "val_loss": val_log_dict["loss"],
                "val_accuracy": val_log_dict["accuracy"],
            }
        )

    pd.DataFrame(log_list).to_csv(
        f"./output/log/train_log_{timestamp}.csv", index=False
    )


def initial_strategy(
    strategy: str, num_classes: int, parameter: Dict[str, int | float]
) -> Tuple[nn.Module, optim.Optimizer]:
    model = models.vgg16(
        weights=models.VGG16_Weights.DEFAULT if (strategy != "zero") else None
    )
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    # 策略 2: 冻结特征层 - 只训练分类层
    if strategy == "frozen_features":
        for param in model.features.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(
            params=model.classifier.parameters(),
            lr=parameter["base_lr"],
            weight_decay=parameter["weight_decay"],
        )
    # 策略3: 全部微调 - 所有层相同学习率 and 策略 1: 不使用迁移学习
    elif strategy == "full" or strategy == "zero":
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=parameter["base_lr"],
            weight_decay=parameter["weight_decay"],
        )
    # 策略 4: 全部微调 - 分层学习率
    elif strategy == "full_layerwise":
        optimizer = optim.AdamW(
            params=[
                {
                    "params": model.features.parameters(),
                    "lr": parameter["features_lr"],
                },
                {
                    "params": model.classifier.parameters(),
                    "lr": parameter["classifier_lr"],
                },
            ],
            weight_decay=parameter["weight_decay"],
        )
    else:
        raise ValueError(f"未知策略: {strategy}")

    return model, optimizer


def main():
    with open("./config/train.toml", "rb") as file:
        parameter = tomllib.load(file)

    create_dir()

    dataset_path = Path("./data/Ti5_split")
    train_path, val_path = dataset_path / "train", dataset_path / "val"
    train_loader, val_loader = load_data(train_path, val_path)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model, optimizer = initial_strategy(
        strategy="zero",
        num_classes=len(train_loader.dataset.classes),
        parameter=parameter,
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=parameter["epoch"] / 10,
        num_training_steps=parameter["epoch"],
    )

    criterion = nn.CrossEntropyLoss()

    model = model.to(DEVICE)

    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        criterion,
        DEVICE,
        parameter["epoch"],
    )

    print("训练完成!")


if __name__ == "__main__":
    main()
