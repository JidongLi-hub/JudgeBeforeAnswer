import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

categories = {
    "Perceptual Level": {"value": 2990, "percent": 45.7,
                         "sub": {
                             "Entity Existence": 998,
                             "Visual Attributes": 1000,
                             "Numeric Attributes": 411,
                             "State Attributes": 163,
                             "OCR Content": 277,
                             "Symbol Meaning": 141
                         }},
    "Cognitive Level": {"value": 2687, "percent": 41.1,
                        "sub": {
                            "Spatial Relation": 997,
                            "Interaction Relation": 430,
                            "Possessive Relation": 405,
                            "Emotion": 140,
                            "Scene": 715
                        }},
    "Reasoning Level": {"value": 863, "percent": 13.2,
                        "sub": {
                            "Logical": 48,
                            "Commonsense": 815
                        }}
}

parent_colors = ['#ff9999', '#66b3ff', '#c4e17f']

# --- 数据准备 ---
inner_sizes, inner_colors = [], []
outer_labels, outer_sizes, outer_colors = [], [], []

for i, (cat, data) in enumerate(categories.items()):
    inner_sizes.append(data["value"])
    inner_colors.append(parent_colors[i])
    for sub, val in data["sub"].items():
        outer_labels.append(sub)
        outer_sizes.append(val)
        outer_colors.append(parent_colors[i])

fig, ax = plt.subplots(figsize=(10, 10))

# --- 内圈，只显示颜色弧段，整体缩小 ---
wedges1, _ = ax.pie(
    inner_sizes,
    radius=0.5,  # 内环半径缩小
    colors=inner_colors,
    wedgeprops=dict(width=0.25, edgecolor='w')  # 保持小间隙
)

# --- 外圈 ---
wedges2, _ = ax.pie(
    outer_sizes,
    radius=0.8,  # 外环半径缩小
    colors=outer_colors,
    wedgeprops=dict(width=0.28, edgecolor='w', linewidth=1)
)

# --- 字体配置 ---
font_dict = {"family": "DejaVu Serif", "size": 11}

# --- 外圈标签 ---
for i, w in enumerate(wedges2):
    ang = (w.theta2 + w.theta1) / 2
    rad = np.deg2rad(ang)
    text = outer_labels[i]

    if 90 < ang < 270:
        rotation = ang - 180
        ha = 'right'
    else:
        rotation = ang
        ha = 'left'

    r = 0.84  # 标签靠近圆环
    x = r * np.cos(rad)
    y = r * np.sin(rad)
    ax.text(x, y, text, rotation=rotation, rotation_mode='anchor',
            ha=ha, fontweight='bold',va='center', **font_dict)

# --- 外圈数值 ---
for i, w in enumerate(wedges2):
    ang = (w.theta2 + w.theta1) / 2
    x = np.cos(np.deg2rad(ang)) * 0.70  # 数值位置调整
    y = np.sin(np.deg2rad(ang)) * 0.70
    ax.text(x, y, str(outer_sizes[i]), ha='center', va='center',
            fontsize=10, fontname="DejaVu Serif")

# --- 内圈百分比放在弧段中心 ---
for i, w in enumerate(wedges1):
    ang_center = (w.theta1 + w.theta2) / 2
    radius = 0.35  # 内圈中心半径调整
    x = radius * np.cos(np.deg2rad(ang_center))
    y = radius * np.sin(np.deg2rad(ang_center))
    percent = categories[list(categories.keys())[i]]['percent']
    ax.text(x, y, f"{percent}%", ha='center', va='center',
            fontsize=13, fontname="DejaVu Serif")

# --- 自定义图例，颜色为圆形，字体统一，放大图例 ---
legend_handles = []
for i, cat in enumerate(categories.keys()):
    legend_handles.append(Circle((0,0), radius=0.15, facecolor=parent_colors[i], edgecolor='w'))  # 圆形变大

ax.legend(handles=legend_handles, labels=list(categories.keys()),
          loc='upper right', frameon=False, prop={"family":"DejaVu Serif", "size":13})

ax.set(aspect="equal")
ax.axis('off')

plt.savefig("dataset.pdf", bbox_inches='tight')
plt.show()
