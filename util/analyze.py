import logging
from collections import Counter
from pathlib import Path

log_dir = Path("./log")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("./log/analyze.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def analyze(source_path: Path) -> None:
    logger.info(f"开始分析文件构成。源路径：{source_path}")

    total_count, valid_count, invalid_count = 0, 0, 0
    extension_counter = Counter()

    file_generator = (p for p in source_path.rglob("*") if p.is_file())
    for file in file_generator:
        total_count += 1
        file_extension = file.suffix.lower()
        if file_extension:
            valid_count += 1
            extension_counter[file_extension] += 1
        else:
            invalid_count += 1

    logger.info(
        f"分析完成。总共有 {total_count} 个文件，{valid_count} 个有效文件，{invalid_count} 个无效文件（无后缀名）。"
    )
    logger.info("文件类型统计：")
    for extension, count in extension_counter.most_common():
        logger.info(f"{extension}: {count} 个 ({count / valid_count:.2%})")


def main():
    source_path = Path("./data/raw")

    analyze(source_path)


if __name__ == "__main__":
    main()
