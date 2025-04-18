import logging
import shutil
from pathlib import Path

log_dir = Path("./log")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("./log/filter.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def filter(source_root: Path, target_root: Path) -> None:
    logger.info(f"开始筛选图片文件。源目录路径：{source_root}，目标路径：{target_root}")

    IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    total_count, copied_count, skipped_count, failed_count = 0, 0, 0, 0

    for item in source_root.rglob("*"):
        if not item.is_file():
            continue

        total_count += 1

        if item.suffix.lower() not in IMAGE_EXTENSIONS:
            skipped_count += 1
            logger.debug(f"跳过非图片文件: {item}")
            continue

        relative_path = item.relative_to(source_root)
        target_path = target_root / relative_path

        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(item, target_path)
            copied_count += 1
            logger.debug(f"复制成功: {item} -> {target_path}")

        except Exception as e:
            logger.error(f"复制失败: {item} -> {target_path}: {e}")
            failed_count += 1

    logger.info(
        f"筛选完成。总计: {total_count}，成功复制: {copied_count}，"
        f"跳过非图片: {skipped_count}，失败: {failed_count}"
    )


def main():
    source_root = Path("./data/raw")
    target_root = Path("./data/filter")

    filter(source_root, target_root)


if __name__ == "__main__":
    main()
