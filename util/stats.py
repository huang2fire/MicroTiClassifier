import csv
from pathlib import Path


def stats(source_path: Path):
    stats_dict = {}
    dir_list = [d.name for d in source_path.iterdir()]

    for dir in dir_list:
        dir_path = source_path / dir

        for class_dir in dir_path.iterdir():
            class_name = class_dir.name

            if class_name not in stats_dict:
                stats_dict[class_name] = {s: 0 for s in dir_list}

            stats_dict[class_name][dir] = len(list(class_dir.iterdir()))

    headers = ["class"] + dir_list
    rows = [
        {"class": class_name, **counts} for class_name, counts in stats_dict.items()
    ]

    with open("./output/log/stats.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main():
    dataset_path = Path("./data/Ti5_split")

    stats(dataset_path)


if __name__ == "__main__":
    main()
