from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANALYSIS = DATA / "plot_inputs"
TABLES = DATA / "figure_reference"
REPRODUCED = DATA / "reproduced_results"
FIG = Path(os.environ.get("AAAI_FIGURE_OUTPUT_ROOT", str(ROOT / "reproduced_figures"))).resolve()
APP = FIG / "appendix"
FULL1600_TABLE = ANALYSIS / "model_only_open_commercial_scale_distribution.csv"
DISTRIBUTION_ADAPTER_DIR = DATA / "raw_results" / "adapter" / "best_run"
AES_SCORES = DATA / "aesexpert_scores_1600.csv"

INTENTS = ["functionality", "aesthetics", "usability", "symbolism", "unclear"]
COLORS = {
    "functionality": "#4C78A8",
    "aesthetics": "#E45756",
    "usability": "#54A24B",
    "symbolism": "#F58518",
    "unclear": "#9D9D9D",
}
LABELS = {
    "functionality": "Functionality",
    "aesthetics": "Aesthetics",
    "usability": "Usability",
    "symbolism": "Symbolism",
    "unclear": "Unclear",
}
CATEGORIES = [
    "early_automobile",
    "sports_car",
    "broom",
    "vacuum_cleaner",
    "desk_lamp",
    "electric_kettle",
    "lounge_chair",
    "sneakers",
]
CATEGORY_LABELS = {
    "early_automobile": "Early automobile",
    "sports_car": "Sports car",
    "broom": "Broom",
    "vacuum_cleaner": "Vacuum cleaner",
    "desk_lamp": "Desk lamp",
    "electric_kettle": "Electric kettle",
    "lounge_chair": "Lounge chair",
    "sneakers": "Sneakers",
}
MODELS = [
    "LLaVA-1.5 (7B, 2023)",
    "MiniCPM-V2.5 (8B, 2024)",
    "Qwen2.5-VL (7B, 2025)",
    "InternVL3 (8B, 2025)",
    "Qwen2.5-VL (32B, 2025)",
    "Qwen2.5-VL (72B, 2025)",
    "InternVL3 (38B, 2025)",
    "GPT-5.5",
    "Gemini 2.5 Flash",
]
ADAPTER = "Human-aligned adapter"
DISPLAY = {
    "LLaVA-1.5 (7B, 2023)": "LLaVA-1.5 7B (2023)",
    "MiniCPM-V2.5 (8B, 2024)": "MiniCPM-V2.5 8B (2024)",
    "Qwen2.5-VL (7B, 2025)": "Qwen2.5-VL 7B (2025)",
    "InternVL3 (8B, 2025)": "InternVL3 8B (2025)",
    "Qwen2.5-VL (32B, 2025)": "Qwen2.5-VL 32B (2025)",
    "Qwen2.5-VL (72B, 2025)": "Qwen2.5-VL 72B (2025)",
    "InternVL3 (38B, 2025)": "InternVL3 38B (2025)",
    "GPT-5.5": "GPT-5.5 (2026)",
    "Gemini 2.5 Flash": "Gemini 2.5 Flash (2025)",
    "Human mean": "Human mean (n=30)",
    ADAPTER: "Human-aligned\nadapter",
}

SHORT_DISPLAY = {
    "LLaVA-1.5 (7B, 2023)": "LLaVA-7B (2023)",
    "MiniCPM-V2.5 (8B, 2024)": "MiniCPM-8B (2024)",
    "Qwen2.5-VL (7B, 2025)": "Qwen-7B (2025)",
    "InternVL3 (8B, 2025)": "Intern-8B (2025)",
    "Qwen2.5-VL (32B, 2025)": "Qwen-32B (2025)",
    "Qwen2.5-VL (72B, 2025)": "Qwen-72B (2025)",
    "InternVL3 (38B, 2025)": "Intern-38B (2025)",
    "GPT-5.5": "GPT-5.5 (2026)",
    "GPT-5.5 (2025)": "GPT-5.5 (2026)",
    "Gemini 2.5 Flash": "Gemini Flash (2025)",
    "Gemini 2.5 Flash (2025)": "Gemini Flash (2025)",
    "Human mean": "Human mean",
    ADAPTER: "Human-aligned\nadapter",
}

FULL1600_ROWS = MODELS[:7] + ["GPT-5.5 (2025)", "Gemini 2.5 Flash (2025)"]


def setup():
    FIG.mkdir(parents=True, exist_ok=True)
    APP.mkdir(parents=True, exist_ok=True)
    REPRODUCED.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 420,
        }
    )


def distribution(values):
    counts = pd.Series(values).value_counts(normalize=True).to_dict()
    return np.array([counts.get(intent, 0.0) for intent in INTENTS], dtype=float)


def human_soft(human, category=None):
    data = human if category is None else human[human["category"] == category]
    return distribution(data["design_intent"])


def model_dist(matched, model, category=None):
    data = matched[matched["model"] == model]
    if category is not None:
        data = data[data["category"] == category]
    return distribution(data["design_intent"])


def load_distribution_adapter():
    return pd.read_csv(DATA / "adapter_oof_predictions_200.csv")


def adapter_soft_dist(adapter, category=None):
    data = adapter if category is None else adapter[adapter["category"] == category]
    return data[[f"pred_p_{intent}" for intent in INTENTS]].mean(axis=0).to_numpy(dtype=float)


def source_group(name):
    if name.startswith(("LLaVA", "MiniCPM")):
        return "open"
    if name.startswith("Qwen2.5-VL (7B") or name.startswith("InternVL3 (8B"):
        return "open"
    if name.startswith("Qwen2.5-VL (32B") or name.startswith("Qwen2.5-VL (72B") or name.startswith("InternVL3 (38B"):
        return "scaling"
    if name.startswith(("GPT", "Gemini")):
        return "frontier"
    if name == "Human mean":
        return "human"
    return "adapter"


def grouped_positions(rows, gap=0.56):
    positions = []
    separators = []
    current = 0.0
    previous_group = None
    previous_position = None
    for row in rows:
        group = source_group(row)
        if previous_group is not None and group != previous_group:
            current += gap
            separators.append((previous_position + current) / 2)
        positions.append(current)
        previous_group = group
        previous_position = current
        current += 1.0
    return np.asarray(positions), separators


def stacked_barh(
    ax,
    matrix,
    rows,
    title=None,
    display_labels=None,
    label_threshold=0.12,
    title_size=14,
    tick_size=10,
    value_size=10,
):
    positions, separators = grouped_positions(rows)
    left = np.zeros(len(rows))
    for i, intent in enumerate(INTENTS):
        vals = matrix[:, i]
        ax.barh(positions, vals, left=left, height=0.98, color=COLORS[intent], edgecolor="white", linewidth=0.78)
        for row_idx, value in enumerate(vals):
            if value >= label_threshold:
                ax.text(left[row_idx] + value / 2, positions[row_idx], f"{value * 100:.0f}%", ha="center", va="center", color="white", fontsize=value_size, fontweight="bold")
        left += vals
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{value:.1f}" for value in np.linspace(0, 1, 6)], fontsize=tick_size)
    ax.grid(axis="x", alpha=0.16, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.set_yticks(positions)
    labels = display_labels or [SHORT_DISPLAY.get(row, row) for row in rows]
    ax.set_yticklabels(labels, fontsize=tick_size)
    ax.tick_params(axis="y", length=0, pad=1)
    if title:
        ax.set_title(title, pad=4, fontsize=title_size, fontweight="bold")
    for separator in separators:
        ax.axhline(separator, color="#D62728", linestyle=(0, (2, 2)), linewidth=1.25)


def legend(fig, y=0.01, fontsize=10):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[i]) for i in INTENTS]
    fig.legend(
        handles,
        ["Func.", "Aes.", "Usab.", "Symb.", "Unc."],
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=5,
        frameon=False,
        fontsize=fontsize,
        handlelength=1.15,
        columnspacing=0.75,
    )


def plot_human_distribution(human):
    matrix = np.vstack([human_soft(human, category) for category in CATEGORIES])
    fig, ax = plt.subplots(figsize=(4.05, 3.8))
    y = np.arange(len(CATEGORIES))
    left = np.zeros(len(CATEGORIES))
    for i, intent in enumerate(INTENTS):
        vals = matrix[:, i]
        ax.barh(y, vals, left=left, height=0.98, color=COLORS[intent], edgecolor="white", linewidth=0.78)
        for j, value in enumerate(vals):
            if value >= 0.12:
                ax.text(left[j] + value / 2, j, f"{value * 100:.0f}%", ha="center", va="center", color="white", fontsize=6.8, fontweight="bold")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([CATEGORY_LABELS[c] for c in CATEGORIES], fontsize=6.0)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{value:.1f}" for value in np.linspace(0, 1, 6)], fontsize=6.6)
    ax.set_xlabel("Design-intent share", fontsize=7.2, labelpad=1)
    ax.grid(axis="x", alpha=0.16, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_title("Human Design-Intent Distributions (n=30)", fontsize=9.0, fontweight="bold", pad=3)
    legend(fig, 0.000, fontsize=6.4)
    fig.tight_layout(rect=[0.0, 0.075, 1.0, 1.0])
    fig.savefig(FIG / "paper_fig01_human_distribution_30.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_selected_200(human, matched, adapter):
    selected = ["early_automobile", "sneakers"]
    rows = MODELS + ["Human mean", ADAPTER]
    short_display = {
        "LLaVA-1.5 (7B, 2023)": "LLaVA-7B (2023)",
        "MiniCPM-V2.5 (8B, 2024)": "MiniCPM-8B (2024)",
        "Qwen2.5-VL (7B, 2025)": "Qwen-7B (2025)",
        "InternVL3 (8B, 2025)": "Intern-8B (2025)",
        "Qwen2.5-VL (32B, 2025)": "Qwen-32B (2025)",
        "Qwen2.5-VL (72B, 2025)": "Qwen-72B (2025)",
        "InternVL3 (38B, 2025)": "Intern-38B (2025)",
        "GPT-5.5": "GPT-5.5 (2026)",
        "Gemini 2.5 Flash": "Gemini Flash (2025)",
        "Human mean": "Human mean",
        ADAPTER: "Human-aligned\nadapter",
    }
    positions = []
    separators = []
    current = 0.0
    for idx in range(len(rows)):
        positions.append(current)
        current += 1.0
        if idx in {3, 6, 8, 9} and idx + 1 < len(rows):
            separators.append(current - 0.22)
            current += 0.56
    positions = np.asarray(positions)

    fig, axes = plt.subplots(2, 1, figsize=(4.05, 4.65), sharex=True)
    for ax, category in zip(axes, selected):
        matrix = []
        for row in rows:
            if row == "Human mean":
                matrix.append(human_soft(human, category))
            elif row == ADAPTER:
                matrix.append(adapter_soft_dist(adapter, category))
            else:
                matrix.append(model_dist(matched, row, category))
        matrix = np.vstack(matrix)
        left = np.zeros(len(rows))
        for intent_idx, intent in enumerate(INTENTS):
            values = matrix[:, intent_idx]
            ax.barh(
                positions,
                values,
                left=left,
                height=0.98,
                color=COLORS[intent],
                edgecolor="white",
                linewidth=0.78,
            )
            for y_pos, value, left_value in zip(positions, values, left):
                if value >= 0.24:
                    ax.text(
                        left_value + value / 2,
                        y_pos,
                        f"{value * 100:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=6.8,
                        color="white",
                        fontweight="bold",
                    )
            left += values
        for separator in separators:
            ax.axhline(separator, color="#D62728", linestyle=(0, (3, 2)), linewidth=1.05)
        ax.set_title(CATEGORY_LABELS[category], fontweight="bold", pad=2, fontsize=9.0)
        ax.set_xlim(0, 1)
        ax.set_yticks(positions)
        ax.set_yticklabels([short_display[row] for row in rows], fontsize=6.0)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.16, linewidth=0.55)
        ax.tick_params(axis="y", length=0, pad=1)
        ax.tick_params(axis="x", labelsize=6.6, pad=1)
    axes[-1].set_xlabel("Design-intent share", fontsize=7.2, labelpad=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[intent]) for intent in INTENTS]
    fig.legend(
        handles,
        ["Func.", "Aes.", "Usab.", "Symb.", "Unc."],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.000),
        fontsize=6.4,
        handlelength=1.15,
        columnspacing=0.75,
    )
    fig.tight_layout(rect=[0.0, 0.055, 1.0, 1.0], h_pad=0.36)
    fig.savefig(FIG / "paper_fig04_selected_categories_final30.png", bbox_inches="tight")
    plt.close(fig)


def full1600_dist(full1600, source, category=None):
    data = full1600[full1600["source"] == source]
    if category is not None:
        data = data[data["category"] == category]
    return data[INTENTS].astype(float).mean(axis=0).reindex(INTENTS).to_numpy()


def matched_rows_and_matrix(human, matched, adapter, category=None):
    rows = MODELS + ["Human mean", ADAPTER]
    matrix = []
    for row in rows:
        if row == "Human mean":
            matrix.append(human_soft(human, category))
        elif row == ADAPTER:
            matrix.append(adapter_soft_dist(adapter, category))
        else:
            matrix.append(model_dist(matched, row, category))
    return rows, np.vstack(matrix)


def plot_overall_distribution(matrix, rows, title, output):
    fig, ax = plt.subplots(figsize=(8.4, 5.7))
    stacked_barh(
        ax,
        matrix,
        rows,
        title=title,
        label_threshold=0.10,
        title_size=14,
        tick_size=10,
        value_size=10,
    )
    ax.set_xlabel("Design-intent share", fontsize=11)
    # Center the complete composition (left-side model labels plus bars), not
    # only the plotting axis, within the full-width appendix page.
    ax.title.set_x(0.357)
    ax.xaxis.set_label_coords(0.357, -0.095)
    legend(fig, 0.010, fontsize=10)
    fig.subplots_adjust(left=0.25, right=0.95, top=0.90, bottom=0.17)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def plot_category_distributions(matrix_for_category, rows, title, output):
    fig, axes = plt.subplots(4, 2, figsize=(12.0, 15.0), sharex=True)
    for panel_idx, (ax, category) in enumerate(zip(axes.flat, CATEGORIES)):
        stacked_barh(
            ax,
            matrix_for_category(category),
            rows,
            title=CATEGORY_LABELS[category],
            label_threshold=0.18,
            title_size=12,
            tick_size=8,
            value_size=8,
        )
        # The model order is shared by each row of panels. Showing it once on
        # the left prevents the right-column labels from colliding at center.
        if panel_idx % 2 == 1:
            ax.tick_params(axis="y", labelleft=False)
    fig.supxlabel("Design-intent share", fontsize=10, y=0.040)
    legend(fig, 0.010, fontsize=10)
    fig.subplots_adjust(left=0.18, right=0.99, top=0.985, bottom=0.072, wspace=0.08, hspace=0.22)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def plot_full1600_distributions(full1600):
    overall = np.vstack([full1600_dist(full1600, row) for row in FULL1600_ROWS])
    plot_overall_distribution(
        overall,
        FULL1600_ROWS,
        "Full-Set Design-Intent Distributions (1,600 Images)",
        APP / "16_model_only_open_commercial_scale_overall_distribution_large.png",
    )
    plot_category_distributions(
        lambda category: np.vstack(
            [full1600_dist(full1600, row, category) for row in FULL1600_ROWS]
        ),
        FULL1600_ROWS,
        "Full-Set Category Distributions",
        APP / "17_model_only_open_commercial_scale_category_distribution_large.png",
    )


def plot_integrated_overall(human, matched, adapter):
    rows, matrix = matched_rows_and_matrix(human, matched, adapter)
    plot_overall_distribution(
        matrix,
        rows,
        "Shared-Subset Design-Intent Distribution (200 Images)",
        APP / "11_integrated_9b_200_human30_adapter_oof_overall.png",
    )


def plot_integrated_categories(human, matched, adapter):
    rows = MODELS + ["Human mean", ADAPTER]
    plot_category_distributions(
        lambda category: matched_rows_and_matrix(human, matched, adapter, category)[1],
        rows,
        "Shared-Subset Category Distributions",
        APP / "12_integrated_9b_200_human30_adapter_oof_category.png",
    )


def plot_category_bias(category_metrics, bias):
    cm = category_metrics[category_metrics["model"].isin(MODELS)].copy()
    hm = cm.pivot(index="model", columns="category", values="haas").reindex(index=MODELS, columns=CATEGORIES)
    bm = bias[bias["model"].isin(MODELS)].pivot(index="model", columns="design_intent", values="share_delta_vs_soft").reindex(index=MODELS, columns=INTENTS)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), gridspec_kw={"width_ratios": [1.45, 1.0]})
    im0 = axes[0].imshow(hm.values, cmap="YlGnBu", vmin=0, vmax=0.85, aspect="auto")
    im1 = axes[1].imshow(bm.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    axes[0].set_title("Category-Level HASS", fontsize=12, fontweight="bold", pad=7)
    axes[1].set_title("Label-Selection Bias", fontsize=12, fontweight="bold", pad=7)
    axes[0].set_xticks(range(len(CATEGORIES)), [CATEGORY_LABELS[c] for c in CATEGORIES], rotation=38, ha="right", fontsize=9)
    axes[1].set_xticks(range(len(INTENTS)), [LABELS[i] for i in INTENTS], rotation=38, ha="right", fontsize=9)
    axes[0].set_yticks(range(len(MODELS)), [DISPLAY[m] for m in MODELS], fontsize=9)
    axes[1].set_yticks(range(len(MODELS)), [])
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.iloc[i, j]
            axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > .55 else "black")
    for i in range(bm.shape[0]):
        for j in range(bm.shape[1]):
            v = bm.iloc[i, j]
            axes[1].text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > .28 else "black")
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.032, pad=0.02)
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.02)
    cb0.ax.tick_params(labelsize=9)
    cb1.ax.tick_params(labelsize=9)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.90, bottom=0.20, wspace=0.36)
    fig.savefig(FIG / "paper_fig03_category_bias_heatmaps_final30.png", bbox_inches="tight")
    plt.close(fig)


def plot_aesthetic_correlation(human, matched):
    aes = pd.read_csv(AES_SCORES)
    aes = aes[aes["image_id"].isin(set(human["image_id"]))]
    aes_cat = aes.groupby("category")["aesthetic_score_0_1"].mean().reindex(CATEGORIES)
    rows = []
    for source in MODELS + ["Human mean"]:
        shares = []
        for category in CATEGORIES:
            dist = human_soft(human, category) if source == "Human mean" else model_dist(matched, source, category)
            shares.append(dist[INTENTS.index("aesthetics")])
        pearson = pd.Series(aes_cat.values).corr(pd.Series(shares), method="pearson")
        spearman = pd.Series(aes_cat.values).corr(pd.Series(shares), method="spearman")
        rows.append({"source": source, "pearson": pearson, "spearman": spearman, "n_categories": 8})
    out = pd.DataFrame(rows)
    out.to_csv(REPRODUCED / "human30_aesexpert_category_correlations.csv", index=False, encoding="utf-8-sig")
    correlation_display = {
        "MiniCPM-V2.5 (8B, 2024)": "MiniCPM-8B (2024)",
        "Qwen2.5-VL (7B, 2025)": "Qwen-7B (2025)",
        "InternVL3 (8B, 2025)": "Intern-8B (2025)",
        "Qwen2.5-VL (32B, 2025)": "Qwen-32B (2025)",
        "Qwen2.5-VL (72B, 2025)": "Qwen-72B (2025)",
        "InternVL3 (38B, 2025)": "Intern-38B (2025)",
        "GPT-5.5": "GPT-5.5 (2026)",
        "Gemini 2.5 Flash": "Gemini Flash (2025)",
        "Human mean": "Human mean (n=30)",
    }
    fig, ax = plt.subplots(figsize=(5.4, 6.0))
    plot = out.dropna(subset=["pearson"]).copy()
    y = np.arange(len(plot))
    bar_colors = ["#5B8CC0" if s in MODELS[:4] else "#7A65B8" if s in MODELS[4:7] else "#D85C5C" if s in MODELS[7:] else "#4B9C79" for s in plot["source"]]
    ax.barh(y, plot["pearson"], color=bar_colors, height=0.74)
    ax.set_yticks(y, [correlation_display[s] for s in plot["source"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Pearson correlation across categories", fontsize=12)
    ax.set_title("Aesthetic-Proxy Correlation", fontsize=15, fontweight="bold", pad=8)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    for i, v in enumerate(plot["pearson"]):
        ax.text(v + 0.016, i, f"{v:.2f}", va="center", fontsize=10.5)
    group_handles = [
        Patch(facecolor="#5B8CC0", label="Open-source"),
        Patch(facecolor="#7A65B8", label="Scaling"),
        Patch(facecolor="#D85C5C", label="Frontier"),
        Patch(facecolor="#4B9C79", label="Human mean"),
    ]
    ax.legend(
        handles=group_handles,
        ncol=4,
        fontsize=10.5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        handlelength=1.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.32, right=0.97, top=0.88, bottom=0.30)
    fig.savefig(FIG / "paper_fig06_aesexpert_correlation_final30.png", bbox_inches="tight")
    plt.close(fig)


def plot_aesthetic_category_mean():
    summary = pd.read_csv(TABLES / "aesexpert_category_summary.csv")
    summary = summary.sort_values("category_order")
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    x = np.arange(len(summary))
    bars = ax.bar(x, summary["mean_score"], width=0.72, color="#5B7FA6")
    ax.set_ylim(0, 10.8)
    ax.set_ylabel("Aesthetic proxy score", fontsize=12)
    ax.set_title("Aesthetic Proxy Mean Score by Product Category", fontsize=15, fontweight="bold", pad=12)
    ax.set_xticks(x, summary["category_label"], rotation=18, ha="right")
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, summary["mean_score"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.20,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.22)
    fig.savefig(APP / "01_aesexpert_category_mean_scores_large.png", bbox_inches="tight")
    plt.close(fig)


def plot_aesthetic_category_distribution():
    scores = pd.read_csv(AES_SCORES)
    data = [
        pd.to_numeric(
            scores.loc[scores["category"] == category, "aesthetic_score"],
            errors="coerce",
        ).dropna().to_numpy()
        for category in CATEGORIES
    ]
    labels = [CATEGORY_LABELS[category] for category in CATEGORIES]

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.58,
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "#2CA02C",
            "markeredgecolor": "#2CA02C",
            "markersize": 6,
        },
        medianprops={"color": "#F58518", "linewidth": 1.7},
        boxprops={"linewidth": 1.2, "edgecolor": "#4A4A4A"},
        whiskerprops={"linewidth": 1.1, "color": "#4A4A4A"},
        capprops={"linewidth": 1.1, "color": "#4A4A4A"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#333333",
            "markersize": 4,
            "alpha": 0.75,
        },
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#F8FAFC")
    ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("AesExpert aesthetic score (1-10)", fontsize=12)
    ax.set_title("AesExpert Score Distribution by Product Category", fontweight="bold", fontsize=15, pad=12)
    # All observed category scores are at least 5; focus on the populated range.
    ax.set_ylim(5, 10.3)
    ax.set_yticks(np.arange(5, 11, 1))
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.22)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.22)
    fig.savefig(APP / "02_aesexpert_category_boxplot.png", bbox_inches="tight")
    plt.close(fig)


def main():
    setup()
    full1600 = pd.read_csv(FULL1600_TABLE)
    human = pd.read_csv(ANALYSIS / "human30_human_long.csv")
    matched = pd.read_csv(ANALYSIS / "human30_human_vlm_matched_long.csv")
    adapter = load_distribution_adapter()
    adapter.to_csv(
        REPRODUCED / "human30_distribution_aware_adapter_oof.csv",
        index=False,
        encoding="utf-8-sig",
    )
    category_metrics = pd.read_csv(TABLES / "human30_category_level_metrics.csv")
    bias = pd.read_csv(TABLES / "human30_model_bias_overselection.csv")
    plot_human_distribution(human)
    plot_full1600_distributions(full1600)
    plot_selected_200(human, matched, adapter)
    plot_integrated_overall(human, matched, adapter)
    plot_integrated_categories(human, matched, adapter)
    plot_category_bias(category_metrics, bias)
    plot_aesthetic_correlation(human, matched)
    plot_aesthetic_category_mean()
    plot_aesthetic_category_distribution()
    print("Generated final 30-rater paper figures.")


if __name__ == "__main__":
    main()
