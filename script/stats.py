import tomllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

with open("./config/path.toml", "rb") as f:
    path = tomllib.load(f)

df = pd.read_csv(path["log"]["stats"])

categories = df["class"]
data = {
    "训练集(原始)": df["train"],
    "验证集(原始)": df["val"],
    "训练集(增强)": df["train_aug"],
    "验证集(增强)": df["val_aug"],
}

x = np.arange(len(categories))
width = 0.3
multiplier = 0

sns.set_theme(style="white", font="Noto Sans SC")

fig, ax = plt.subplots(figsize=(8, 5), layout="tight")

for key, value in data.items():
    if multiplier % 2 == 0:
        offset = width * (multiplier // 2)
        rects = ax.bar(x + offset, value, width, label=key)
        ax.bar_label(rects, padding=3)
    else:
        bottom = data[list(data.keys())[multiplier - 1]]
        rects = ax.bar(x + offset, value, width, bottom=bottom, label=key)
        ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set(
    title="各类别增强前后数据集样本数量对比",
    xlabel="类别",
    ylabel="样本数量",
    ylim=(0, 450),
)
ax.set_xticks(x + width / 2, categories)
ax.legend(ncols=2)

plt.savefig(path["chart"]["stats"])

plt.show()
