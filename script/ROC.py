import json
import tomllib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

with open("./config/path.toml", "rb") as f:
    path = tomllib.load(f)

with open("./output/log/classes_list.json", "r") as f:
    classes_list = json.load(f)

df = pd.read_csv(path["log"]["val_pred"])

sns.set_theme(style="whitegrid", palette="deep", font="Noto Sans SC")

fig, ax = plt.subplots(figsize=(8, 5), layout="tight")

# marker = ["o", "v", "s", "*", "x"]
idx = 0

for cls in classes_list:
    y_test = list((df["类别名称"] == cls))
    y_score = list((df[f"{cls}-预测置信度"]))

    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    ax.plot(
        fpr,
        tpr,
        # marker=marker[idx],
        label=cls,
    )

    idx += 1

ax.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
ax.legend()

plt.savefig(path["chart"]["ROC"])

plt.show()
