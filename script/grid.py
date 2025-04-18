import json
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

plt.rcParams["font.sans-serif"] = ["Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False

with open("./config/path.toml", "rb") as f:
    path = tomllib.load(f)
with open("./output/log/classes_list.json", "r") as f:
    classes_list = json.load(f)

grid_list = []

for image_path in Path("./assets/image").iterdir():
    img = Image.open(image_path)
    img = img.resize((224, 224))
    grid_list.append(img)

fig = plt.figure(figsize=(12, 8), layout="tight")
gs = GridSpec(2, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0:2])
ax5 = fig.add_subplot(gs[1, 1:3])
axes = [ax1, ax2, ax3, ax4, ax5]

for ax, img, title in zip(axes, grid_list, classes_list):
    ax.imshow(img)
    ax.set_title(title, y=-0.1, ha="center")
    ax.axis("off")

plt.savefig(path["chart"]["grid"])

plt.show()
