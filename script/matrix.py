import json
import tomllib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

with open("./config/path.toml", "rb") as f:
    path = tomllib.load(f)

with open("./output/log/classes_list.json", "r") as f:
    classes_list = json.load(f)

df = pd.read_csv(path["log"]["val_pred"])

confusion_matrix_array = confusion_matrix(df["类别名称"], df["top-1-预测名称"])

sns.set_theme(style="whitegrid", palette="deep", font="Noto Sans SC")

fig, ax = plt.subplots(figsize=(8, 5), layout="tight")

sns.heatmap(
    data=confusion_matrix_array,
    cmap="Blues",
    annot=True,
    fmt="d",
    xticklabels=classes_list,
    yticklabels=classes_list,
    ax=ax,
)

ax.set(title="混淆矩阵", xlabel="Predicted", ylabel="Actual")

plt.savefig(path["chart"]["matrix"])

plt.show()
