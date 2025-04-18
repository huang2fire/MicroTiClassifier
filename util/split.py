import logging
import random
import shutil
from pathlib import Path
from typing import List, Tuple

log_dir = Path("./log")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("./log/split.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def copy2dir(image_list: List[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    for image in image_list:
        try:
            shutil.copy2(image, target_dir / image.name)
            logger.debug(f"复制成功: {image} -> {target_dir}")

        except Exception as e:
            logger.error(f"复制失败: {image} -> {target_dir}: {e}")


def split_list(
    image_list: List[Path], val_ratio: float = 0.2, seed: int = 2025
) -> Tuple[List[Path], List[Path]]:
    random.seed(seed)
    random.shuffle(image_list)

    split_idx = int(len(image_list) * val_ratio)

    return image_list[split_idx:], image_list[:split_idx]


def split_dir(dir: Path, target_root: Path) -> None:
    file_list = list(dir.iterdir())

    logger.info(f"处理类别: {dir.name} (共 {len(file_list)} 个文件)")

    train_list, val_list = split_list(file_list)

    train_dir = target_root / "train" / dir.name
    val_dir = target_root / "val" / dir.name

    copy2dir(train_list, train_dir)
    copy2dir(val_list, val_dir)

    logger.info(
        f"完成处理: {dir.name} - 训练: {len(train_list)}, 验证: {len(val_list)}, "
    )


def split(source_path: Path, target_path: Path) -> None:
    logger.info(f"开始分割数据集。源目录: {source_path}, 目标目录: {target_path}")

    for dir in source_path.iterdir():
        split_dir(dir, target_path)

    logger.info("分割完成。")


def main():
    source_path = Path("./data/Ti5")
    target_path = Path("./data/Ti5_splitd")

    split(source_path, target_path)


if __name__ == "__main__":
    main()
