import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from PIL import Image

log_dir = Path("./log")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./log/augment.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class ImageOperation:
    processor: Callable[[Image.Image], Tuple[List[Image.Image], List[str]]]
    formatter: Callable[[str, str, str], str]
    description: str


def TiRotate(image: Image.Image, angle_list: List[int]) -> List[Image.Image]:
    return [image.rotate(angle, expand=True) for angle in angle_list]


def TiFlip(image: Image.Image, type_list: List[str]) -> List[Image.Image]:
    FLIP_MAP = {
        "lr": Image.Transpose.FLIP_LEFT_RIGHT,
        "tb": Image.Transpose.FLIP_TOP_BOTTOM,
    }
    return [image.transpose(FLIP_MAP[ttype]) for ttype in type_list]


def TiResizeMaxSize(image: Image.Image, max_size: int, mode: str) -> Image.Image:
    width, height = image.size
    ref_size = min(width, height) if mode == "short" else max(width, height)
    scale = max_size / ref_size

    return image.resize(
        (int(width * scale), int(height * scale)), Image.Resampling.LANCZOS
    )


def TiGridCrop(image: Image.Image, step: int, min_size: int) -> List[Image.Image]:
    width, height = image.size
    valid_step = min(width // min_size, height // min_size)

    if valid_step == 0:
        logger.warning("因最小网格限制，返回原图")
        return [image]

    if valid_step < step:
        logger.warning(f"因最小网格限制，step 从 {step} 调整为 {valid_step}")
        step = valid_step

    grid_width, grid_height = width / step, height / step

    return [
        image.crop(
            (
                int(i * grid_width),
                int(j * grid_height),
                int((i + 1) * grid_width),
                int((j + 1) * grid_height),
            )
        )
        for i in range(step)
        for j in range(step)
    ]


def get_style_operations() -> Dict[str, ImageOperation]:
    return {
        "rotate": ImageOperation(
            processor=lambda img: (TiRotate(img, [180]), ["180"]),
            formatter=lambda stem, ext, suffix: f"{stem}_rotate{suffix}.{ext}",
            description="旋转图像（180°）",
        ),
        "flip": ImageOperation(
            processor=lambda img: (TiFlip(img, ["lr", "tb"]), ["lr", "tb"]),
            formatter=lambda stem, ext, suffix: f"{stem}_flip_{suffix}.{ext}",
            description="翻转图像（水平/垂直）",
        ),
        "resize": ImageOperation(
            processor=lambda img: ([TiResizeMaxSize(img, 512, "long")], [""]),
            formatter=lambda stem, ext, _: f"{stem}_resize.{ext}",
            description="调整图像大小（基于长边，最大512px）",
        ),
        "grid": ImageOperation(
            processor=lambda img: (
                TiGridCrop(img, 2, 900),
                [f"_{i}" for i in range(4)],
            ),
            formatter=lambda stem, ext, suffix: f"{stem}_grid{suffix}.{ext}",
            description="划分图像为网格（预设步长 2，最小网格 900px）",
        ),
    }


def process_image(
    image_path: Path, output_dir: Path, operation: ImageOperation
) -> bool:
    logger.debug(f"正在处理图像: {image_path}")

    with Image.open(image_path) as image:
        images, suffixes = operation.processor(image)

        for img, suffix in zip(images, suffixes):
            img_name = operation.formatter(
                image_path.stem,
                image_path.suffix[1:],
                suffix,
            )
            img_path = output_dir / img_name
            img.save(img_path)

            logger.debug(f"成功保存处理后的图像: {img_path}")

        return True

    return False


def process(source_path: Path, target_path: Path, style: str) -> None:
    logger.info(
        f"开始图像增强处理。源目录: {source_path}，目标目录: {target_path}，操作类型: {style}"
    )

    style_operations = get_style_operations()

    if style not in style_operations:
        error_msg = (
            f"未知的处理方式: {style}，支持的操作: {', '.join(style_operations.keys())}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    operation = style_operations[style]

    target_path.mkdir(parents=True, exist_ok=True)

    total_count, success_count = 0, 0

    for dir_path in source_path.iterdir():
        logger.info(f"正在处理目录: {dir_path}")

        for image_path in dir_path.iterdir():
            total_count += 1

            relative_path = image_path.relative_to(source_path)
            output_dir = target_path / relative_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            if process_image(image_path, output_dir, operation):
                success_count += 1

    logger.info(
        f"处理完成。总计: {total_count} 个文件，"
        f"成功: {success_count} 个，失败: {total_count - success_count} 个"
    )


def main():
    style = "resize"
    source_path = Path("./data/Ti5_split/val")
    target_path = Path(f"./data/Ti5_split/val_{style}")

    process(source_path, target_path, style)


if __name__ == "__main__":
    main()
