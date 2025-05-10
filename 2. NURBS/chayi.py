import warnings
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import (MultipleLocator)

warnings.filterwarnings('ignore')


# 可视化环比增长率和同比增长率
def plot_growth_rates(leaf, QoQ_growth, YoY_growth, title):
    fig, ax = plt.subplots(figsize=(6, 9))

    ax.vlines(leaf, ymin=0.5, ymax=QoQ_growth, color='green', alpha=0.7, linewidth=8, label="Leaf Images")
    for a, b in zip(leaf, QoQ_growth):
        # plt.text(a, b - 0.2, r"${0:0.2e}$".format(b), ha='center', va='bottom')
        plt.text(a, b, b, ha='center', va='bottom')

    ax.vlines(leaf, ymin=0.5, ymax=YoY_growth, linestyles="dashed", color='firebrick', alpha=0.9, linewidth=2)
    ax.scatter(leaf, YoY_growth, s=75, color='firebrick', alpha=0.7, label="Reconstructed Models")
    for a, b in zip(leaf, YoY_growth):
        # plt.text(a, b, r"${0:0.2e}$".format(b), ha='center', va='bottom')
        plt.text(a, b, b, ha='center', va='bottom')

    plt.title(title)
    ax.set_xlabel('Leaf Images', fontsize=12)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_ylabel('Comparison Results', fontsize=12)
    ax.set_ylim(0.5, 1.0)
    # ax.grid(True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3, ncol=1,
              markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.tight_layout()


# 读取表格数据
data = pd.read_excel('精度数据.xlsx')
print(data)
columns_to_analyze = ['Shape Descriptors', "Object Extents"]

# 可视化结果
mpl.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'无法显示的问题
for column in columns_to_analyze:
    plot_growth_rates(
        data["leaf"],
        data[f"{column}-实验样本"],
        data[f"{column}-重建模型"],
        column
    )

plt.show()
