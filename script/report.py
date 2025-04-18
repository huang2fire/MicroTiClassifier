import json
import tomllib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report

with open("./config/path.toml", "rb") as f:
    path = tomllib.load(f)

with open("./output/log/classes_list.json", "r") as f:
    classes_list = json.load(f)

df = pd.read_csv(path["log"]["val_pred"])

report = classification_report(
    df["类别名称"], df["top-1-预测名称"], target_names=classes_list, output_dict=True
)

del report["accuracy"]

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(path["log"]["val_report"], index_label="类别")

sns.set_theme(style="whitegrid", palette="deep", font="Noto Sans SC")

fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout="tight")

report_df = pd.read_csv(path["log"]["val_report"])

x = report_df["类别"]
metrics = ["precision", "recall", "f1-score", "support"]

for i, ax in enumerate(axes.flatten()):
    sns.barplot(x=x, y=report_df[metrics[i]], ax=ax)

    ax.set(xlabel="类别", ylabel=metrics[i])
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=45)

plt.savefig(path["chart"]["val_report"])

plt.show()
