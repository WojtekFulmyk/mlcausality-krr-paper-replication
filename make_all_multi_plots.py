# Get plot type to make
import sys

plot_type = sys.argv[1]
if plot_type.lower() == "brier":
    plot_type_cap = "Brier Score"
elif plot_type.lower() == "auc":
    plot_type_cap = "AUC"
elif plot_type.lower() == "accuracy":
    plot_type_cap = "Accuracy (thresh = 0.05)"
elif plot_type.lower() == "balanced_accuracy":
    plot_type_cap = "Bal.Accuracy (thresh = 0.05)"
elif plot_type.lower() == "f1wgt":
    plot_type_cap = "Weighted F1 (thresh = 0.05)"
elif plot_type.lower() == "sensitivity":
    plot_type_cap = "Sensitivity (thresh = 0.05)"
elif plot_type.lower() == "specificity":
    plot_type_cap = "Specificity (thresh = 0.05)"
elif plot_type.lower() == "gmean":
    plot_type_cap = "G-mean (thresh = 0.05)"
elif plot_type.lower() == "accuracy2":
    plot_type_cap = "Accuracy (max G-mean thresh)"
elif plot_type.lower() == "balanced_accuracy2":
    plot_type_cap = "Bal. Accuracy (max G-mean thresh)"
elif plot_type.lower() == "f1wgt2":
    plot_type_cap = "Weighted F1 (max G-mean thresh)"
elif plot_type.lower() == "sensitivity2":
    plot_type_cap = "Sensitivity (max G-mean thresh)"
elif plot_type.lower() == "specificity2":
    plot_type_cap = "Specificity (max G-mean thresh)"
elif plot_type.lower() == "gmean2":
    plot_type_cap = "G-mean (max G-mean thresh)"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pickle

from copy import deepcopy
from decimal import Decimal, localcontext, ROUND_DOWN


# This truncate function is based on:
# https://stackoverflow.com/a/28323804
def truncate(number, places):
    if not isinstance(places, int):
        raise ValueError("Decimal places must be an integer.")
    if places < 1:
        raise ValueError("Decimal places must be at least 1.")
    # If you want to truncate to 0 decimal places, just do int(number).
    with localcontext() as context:
        context.rounding = ROUND_DOWN
        exponent = Decimal(str(10**-places))
        return Decimal(str(number)).quantize(exponent)


with open(r"pickled_data/5_linear_" + plot_type + "_500.pickle", "rb") as input_file:
    df_init1 = pickle.load(input_file)

df_init1["Length of time-series"] = 500


with open(r"pickled_data/5_linear_" + plot_type + "_1000.pickle", "rb") as input_file:
    df_init2 = pickle.load(input_file)

df_init2["Length of time-series"] = 1000

with open(r"pickled_data/5_linear_" + plot_type + "_1500.pickle", "rb") as input_file:
    df_init3 = pickle.load(input_file)

df_init3["Length of time-series"] = 1500

with open(r"pickled_data/5_linear_" + plot_type + "_2000.pickle", "rb") as input_file:
    df_init4 = pickle.load(input_file)

df_init4["Length of time-series"] = 2000
df1 = pd.concat([df_init1, df_init2, df_init3, df_init4]).reset_index(drop=True)
df1 = df1.melt("Length of time-series", var_name="Model", value_name="Val")

with open(r"pickled_data/5_nonlinear_" + plot_type + "_500.pickle", "rb") as input_file:
    df_init1 = pickle.load(input_file)

df_init1["Length of time-series"] = 500


with open(
    r"pickled_data/5_nonlinear_" + plot_type + "_1000.pickle", "rb"
) as input_file:
    df_init2 = pickle.load(input_file)

df_init2["Length of time-series"] = 1000

with open(
    r"pickled_data/5_nonlinear_" + plot_type + "_1500.pickle", "rb"
) as input_file:
    df_init3 = pickle.load(input_file)

df_init3["Length of time-series"] = 1500

with open(
    r"pickled_data/5_nonlinear_" + plot_type + "_2000.pickle", "rb"
) as input_file:
    df_init4 = pickle.load(input_file)

df_init4["Length of time-series"] = 2000
df2 = pd.concat([df_init1, df_init2, df_init3, df_init4]).reset_index(drop=True)
df2 = df2.melt("Length of time-series", var_name="Model", value_name="Val")

with open(r"pickled_data/7_nonlinear_" + plot_type + "_500.pickle", "rb") as input_file:
    df_init1 = pickle.load(input_file)

df_init1["Length of time-series"] = 500


with open(
    r"pickled_data/7_nonlinear_" + plot_type + "_1000.pickle", "rb"
) as input_file:
    df_init2 = pickle.load(input_file)

df_init2["Length of time-series"] = 1000

with open(
    r"pickled_data/7_nonlinear_" + plot_type + "_1500.pickle", "rb"
) as input_file:
    df_init3 = pickle.load(input_file)

df_init3["Length of time-series"] = 1500

with open(
    r"pickled_data/7_nonlinear_" + plot_type + "_2000.pickle", "rb"
) as input_file:
    df_init4 = pickle.load(input_file)

df_init4["Length of time-series"] = 2000
df3 = pd.concat([df_init1, df_init2, df_init3, df_init4]).reset_index(drop=True)
df3 = df3.melt("Length of time-series", var_name="Model", value_name="Val")

with open(r"pickled_data/9_nonlinear_" + plot_type + "_500.pickle", "rb") as input_file:
    df_init1 = pickle.load(input_file)

df_init1["Length of time-series"] = 500


with open(
    r"pickled_data/9_nonlinear_" + plot_type + "_1000.pickle", "rb"
) as input_file:
    df_init2 = pickle.load(input_file)

df_init2["Length of time-series"] = 1000

with open(
    r"pickled_data/9_nonlinear_" + plot_type + "_1500.pickle", "rb"
) as input_file:
    df_init3 = pickle.load(input_file)

df_init3["Length of time-series"] = 1500

with open(
    r"pickled_data/9_nonlinear_" + plot_type + "_2000.pickle", "rb"
) as input_file:
    df_init4 = pickle.load(input_file)

df_init4["Length of time-series"] = 2000
df4 = pd.concat([df_init1, df_init2, df_init3, df_init4]).reset_index(drop=True)
df4 = df4.melt("Length of time-series", var_name="Model", value_name="Val")

with open(
    r"pickled_data/11_nonlinear_" + plot_type + "_500.pickle", "rb"
) as input_file:
    df_init1 = pickle.load(input_file)

df_init1["Length of time-series"] = 500


with open(
    r"pickled_data/11_nonlinear_" + plot_type + "_1000.pickle", "rb"
) as input_file:
    df_init2 = pickle.load(input_file)

df_init2["Length of time-series"] = 1000

with open(
    r"pickled_data/11_nonlinear_" + plot_type + "_1500.pickle", "rb"
) as input_file:
    df_init3 = pickle.load(input_file)

df_init3["Length of time-series"] = 1500

with open(
    r"pickled_data/11_nonlinear_" + plot_type + "_2000.pickle", "rb"
) as input_file:
    df_init4 = pickle.load(input_file)

df_init4["Length of time-series"] = 2000
df5 = pd.concat([df_init1, df_init2, df_init3, df_init4]).reset_index(drop=True)
df5 = df5.melt("Length of time-series", var_name="Model", value_name="Val")

with open(r"pickled_data/34_zachary1_" + plot_type + "_500.pickle", "rb") as input_file:
    df_init1 = pickle.load(input_file)

df_init1["Length of time-series"] = 500


with open(
    r"pickled_data/34_zachary1_" + plot_type + "_1000.pickle", "rb"
) as input_file:
    df_init2 = pickle.load(input_file)

df_init2["Length of time-series"] = 1000

with open(
    r"pickled_data/34_zachary1_" + plot_type + "_1500.pickle", "rb"
) as input_file:
    df_init3 = pickle.load(input_file)

df_init3["Length of time-series"] = 1500

with open(
    r"pickled_data/34_zachary1_" + plot_type + "_2000.pickle", "rb"
) as input_file:
    df_init4 = pickle.load(input_file)

df_init4["Length of time-series"] = 2000
df6 = pd.concat([df_init1, df_init2, df_init3, df_init4]).reset_index(drop=True)
df6 = df6.melt("Length of time-series", var_name="Model", value_name="Val")

with open(r"pickled_data/34_zachary2_" + plot_type + "_500.pickle", "rb") as input_file:
    df_init1 = pickle.load(input_file)

df_init1["Length of time-series"] = 500


with open(
    r"pickled_data/34_zachary2_" + plot_type + "_1000.pickle", "rb"
) as input_file:
    df_init2 = pickle.load(input_file)

df_init2["Length of time-series"] = 1000

with open(
    r"pickled_data/34_zachary2_" + plot_type + "_1500.pickle", "rb"
) as input_file:
    df_init3 = pickle.load(input_file)

df_init3["Length of time-series"] = 1500

with open(
    r"pickled_data/34_zachary2_" + plot_type + "_2000.pickle", "rb"
) as input_file:
    df_init4 = pickle.load(input_file)

df_init4["Length of time-series"] = 2000
df7 = pd.concat([df_init1, df_init2, df_init3, df_init4]).reset_index(drop=True)
df7 = df7.melt("Length of time-series", var_name="Model", value_name="Val")

title_font = {
    "size": "32",
    "color": "black",
    "weight": "bold",
    "verticalalignment": "bottom",
}
title_font2 = {
    "size": "52",
    "color": "black",
    "weight": "bold",
    "verticalalignment": "bottom",
}
f, axes = plt.subplots(1, 7, sharex=False, sharey=True, figsize=(30, 14))
f.text(0.5, 1.025, plot_type_cap, ha="center", va="center", **title_font2)
f.text(-0.01, 0.5, plot_type_cap, ha="center", va="center", rotation=90, **title_font)

sns.boxplot(
    x="Model",
    y="Val",
    hue="Length of time-series",
    data=df1,
    palette="Set3",
    ax=axes[0],
    linewidth=2,
    saturation=1,
    width=0.7,
    color="black",
    notch=False,
    medianprops=dict(color="black", linewidth=10, alpha=1, solid_capstyle="butt"),
    whis=(0, 100),
    showfliers=False,
    bootstrap=10000,
    whiskerprops=dict(color="black", alpha=1),
    boxprops=dict(edgecolor="black", alpha=1),
    capprops=dict(color="black", alpha=1),
)
sns.boxplot(
    x="Model",
    y="Val",
    hue="Length of time-series",
    data=df2,
    palette="Set3",
    ax=axes[1],
    linewidth=2,
    saturation=1,
    width=0.7,
    color="black",
    notch=False,
    medianprops=dict(color="black", linewidth=10, alpha=1, solid_capstyle="butt"),
    whis=(0, 100),
    showfliers=False,
    bootstrap=10000,
    whiskerprops=dict(color="black", alpha=1),
    boxprops=dict(edgecolor="black", alpha=1),
    capprops=dict(color="black", alpha=1),
)
sns.boxplot(
    x="Model",
    y="Val",
    hue="Length of time-series",
    data=df3,
    palette="Set3",
    ax=axes[2],
    linewidth=2,
    saturation=1,
    width=0.7,
    color="black",
    notch=False,
    medianprops=dict(color="black", linewidth=10, alpha=1, solid_capstyle="butt"),
    whis=(0, 100),
    showfliers=False,
    bootstrap=10000,
    whiskerprops=dict(color="black", alpha=1),
    boxprops=dict(edgecolor="black", alpha=1),
    capprops=dict(color="black", alpha=1),
)
sns.boxplot(
    x="Model",
    y="Val",
    hue="Length of time-series",
    data=df4,
    palette="Set3",
    ax=axes[3],
    linewidth=2,
    saturation=1,
    width=0.7,
    color="black",
    notch=False,
    medianprops=dict(color="black", linewidth=10, alpha=1, solid_capstyle="butt"),
    whis=(0, 100),
    showfliers=False,
    bootstrap=10000,
    whiskerprops=dict(color="black", alpha=1),
    boxprops=dict(edgecolor="black", alpha=1),
    capprops=dict(color="black", alpha=1),
)
sns.boxplot(
    x="Model",
    y="Val",
    hue="Length of time-series",
    data=df5,
    palette="Set3",
    ax=axes[4],
    linewidth=2,
    saturation=1,
    width=0.7,
    color="black",
    notch=False,
    medianprops=dict(color="black", linewidth=10, alpha=1, solid_capstyle="butt"),
    whis=(0, 100),
    showfliers=False,
    bootstrap=10000,
    whiskerprops=dict(color="black", alpha=1),
    boxprops=dict(edgecolor="black", alpha=1),
    capprops=dict(color="black", alpha=1),
)
sns.boxplot(
    x="Model",
    y="Val",
    hue="Length of time-series",
    data=df6,
    palette="Set3",
    ax=axes[5],
    linewidth=2,
    saturation=1,
    width=0.7,
    color="black",
    notch=False,
    medianprops=dict(color="black", linewidth=10, alpha=1, solid_capstyle="butt"),
    whis=(0, 100),
    showfliers=False,
    bootstrap=10000,
    whiskerprops=dict(color="black", alpha=1),
    boxprops=dict(edgecolor="black", alpha=1),
    capprops=dict(color="black", alpha=1),
)
sns.boxplot(
    x="Model",
    y="Val",
    hue="Length of time-series",
    data=df7,
    palette="Set3",
    ax=axes[6],
    linewidth=2,
    saturation=1,
    width=0.7,
    color="black",
    notch=False,
    medianprops=dict(color="black", linewidth=10, alpha=1, solid_capstyle="butt"),
    whis=(0, 100),
    showfliers=False,
    bootstrap=10000,
    whiskerprops=dict(color="black", alpha=1),
    boxprops=dict(edgecolor="black", alpha=1),
    capprops=dict(color="black", alpha=1),
)

axes[0].set_title("5-linear", fontweight="bold", fontsize=48)
axes[1].set_title("5-nonlin", fontweight="bold", fontsize=48)
axes[2].set_title("7-nonlin", fontweight="bold", fontsize=48)
axes[3].set_title("9-nonlin", fontweight="bold", fontsize=48)
axes[4].set_title("11-nonlin", fontweight="bold", fontsize=48)
axes[5].set_title("Zachary1", fontweight="bold", fontsize=48)
axes[6].set_title("Zachary2", fontweight="bold", fontsize=48)

for i in range(7):
    if i < 6:
        axes[i].get_legend().remove()
    else:
        axes[i].legend(
            prop=dict(size=26, weight="bold"),
            title="$\\bf{Num. obs.}$",
            title_fontsize=26,
        )
    x_axis = axes[i].axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)
    y_axis = axes[i].axes.get_yaxis()
    y_label = y_axis.get_label()
    y_label.set_visible(False)
    if i == 0:
        start, end = axes[i].get_ylim()
        new_start = float(truncate(deepcopy(start), 1))
        if plot_type.lower() == "accuracy" or plot_type.lower() == "accuracy2":
            new_start = abs(round(max(new_start - 0.1, 0), 1))
        else:
            new_start = abs(round(max(new_start, 0), 1))
    axes[i].yaxis.set_ticks(np.arange(new_start, end, 0.1))
    axes[i].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
    if i == 0:
        axes[i].tick_params(width=5, size=10)
    else:
        axes[i].tick_params(
            top=False, bottom=True, left=False, right=False, width=5, size=10
        )
    axes[i].tick_params(axis="x", rotation=90, labelsize=48)
    axes[i].tick_params(axis="y", labelsize=48)
    [
        axes[i].axvline(x, color="0.3", linestyle="--", linewidth=3)
        for x in [0.5, 1.5, 2.5]
    ]
    axes[i].grid(visible=True, which="major", axis="y", linewidth=3)
    axes[i].set(axisbelow=True)
    for axis in ["top", "bottom", "left", "right"]:
        axes[i].spines[axis].set_linewidth(5)  # change width
    if i == 0:
        start2, end2 = axes[i].get_ylim()
    for axis in ["left", "right"]:
        axes[i].spines[axis].set_bounds(
            low=start2 - (end2 - start2) * 0.21153846153846154,
            high=end2 + (end2 - start2) * 0.06937799043062202,
        )  # Extend spine bounds

f.tight_layout()
f.subplots_adjust(wspace=0, hspace=0)
plt.rcParams["svg.fonttype"] = "none"
plt.savefig(
    "plots/" + plot_type.replace("_", "") + "plotmulti.pdf", bbox_inches="tight"
)
plt.savefig(
    "plots/" + plot_type.replace("_", "") + "plotmulti.eps", bbox_inches="tight"
)
# plt.show()
