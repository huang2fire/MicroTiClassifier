import os
import tomllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

with open("./config/path.toml", "rb") as f:
    path = tomllib.load(f)

df = pd.read_csv(path["log"]["val_pred"])
semantic_feature_array = np.load(path["npy"]["feature"], allow_pickle=True)

tsne_2d = TSNE(n_components=2).fit_transform(semantic_feature_array)

tsne_2d_df = pd.DataFrame(
    {
        "类别名称": df["类别名称"],
        "X": list(tsne_2d[:, 0].squeeze()),
        "Y": list(tsne_2d[:, 1].squeeze()),
    }
)
tsne_2d_df.to_csv(path["log"]["tsne_2d"], index=False)

sns.set_theme(style="white", palette="deep", font="Noto Sans SC")

fig, ax = plt.subplots(figsize=(8, 8), layout="tight")

sns.scatterplot(
    data=tsne_2d_df, x="X", y="Y", hue="类别名称", style="类别名称", s=100, ax=ax
)

ax.set(xticks=[], yticks=[], title="t-SNE")
ax.legend(loc="lower right")

plt.savefig(path["chart"]["tsne_2d"])

plt.show()
