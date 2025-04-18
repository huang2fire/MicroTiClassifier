import tomllib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

with open("./config/path.toml", "rb") as f:
    path = tomllib.load(f)

log_df = pd.read_csv(path["log"]["train_log"])

sns.set_theme(style="whitegrid", font="Noto Sans SC")

fig = plt.figure(figsize=(10, 5), layout="tight")
gs = GridSpec(1, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

sns.lineplot(data=log_df, x="epoch", y="train_loss", label="train loss", ax=ax1)
sns.lineplot(data=log_df, x="epoch", y="val_loss", label="val loss", ax=ax1)
ax1.set(title="Train & Val Loss", xlabel="Epoch", ylabel="Loss")
ax1.legend()

sns.lineplot(data=log_df, x="epoch", y="train_accuracy", label="train accuracy", ax=ax2)
sns.lineplot(data=log_df, x="epoch", y="val_accuracy", label="val accuracy", ax=ax2)
ax2.set(title="Train & Val Accuracy", xlabel="Epoch", ylabel="Accuracy")
ax2.legend()

plt.savefig(path["chart"]["train_log"])

plt.show()
