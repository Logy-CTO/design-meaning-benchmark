from pathlib import Path
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
INTEG = ROOT / "data" / "plot_inputs"
PILOT = ROOT / "data" / "figure_reference"
GEN_IMAGES = ROOT / "data" / "images" / "human_subset_200"
FIG = Path(os.environ.get("AAAI_FIGURE_OUTPUT_ROOT", str(ROOT / "reproduced_figures"))).resolve()
APP = FIG / "appendix"

LABELS = ["functionality", "aesthetics", "usability", "symbolism", "unclear"]
COLORS = {
    "functionality": "#4C78A8",
    "aesthetics": "#E45756",
    "usability": "#54A24B",
    "symbolism": "#F58518",
    "unclear": "#9D9D9D",
}

CATEGORY_ORDER = [
    ("early_automobile", "Early automobile"),
    ("sports_car", "Sports car"),
    ("broom", "Broom"),
    ("vacuum_cleaner", "Vacuum cleaner"),
    ("desk_lamp", "Desk lamp"),
    ("electric_kettle", "Electric kettle"),
    ("lounge_chair", "Lounge chair"),
    ("sneakers", "Sneakers"),
]

SOURCE_ORDER = [
    "LLaVA-1.5 (7B, 2023)",
    "MiniCPM-V2.5 (8B, 2024)",
    "Qwen2.5-VL (7B, 2025)",
    "InternVL3 (8B, 2025)",
    "Qwen2.5-VL (32B, 2025)",
    "Qwen2.5-VL (72B, 2025)",
    "InternVL3 (38B, 2025)",
    "GPT-5.5 (2025)",
    "Gemini 2.5 Flash (2025)",
    "Human mean (n=6)",
    "Qwen2.5-VL+LoRA adapter (7B, 2025; test n=30)",
]

MODEL_ONLY_ORDER = [
    "LLaVA-1.5 (7B, 2023)",
    "MiniCPM-V2.5 (8B, 2024)",
    "Qwen2.5-VL (7B, 2025)",
    "InternVL3 (8B, 2025)",
    "Qwen2.5-VL (32B, 2025)",
    "Qwen2.5-VL (72B, 2025)",
    "InternVL3 (38B, 2025)",
    "GPT-5.5 (2025)",
    "Gemini 2.5 Flash (2025)",
]

METRIC_ORDER = [
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

GROUP_LINE_COLOR = "#D62728"
GROUP_LINE_STYLE = (0, (2.0, 2.0))


def display_model(name: str) -> str:
    mapping = {
        "GPT-5.5": "GPT-5.5 (2026)",
        "Gemini 2.5 Flash": "Gemini Flash (2025)",
        "GPT-5.5 (2025)": "GPT-5.5 (2026)",
        "Gemini 2.5 Flash (2025)": "Gemini Flash (2025)",
        "Human mean (n=6)": "Human mean",
        "Qwen2.5-VL+LoRA adapter (7B, 2025; test n=30)": "Human-aligned\nadapter",
        "LLaVA-1.5 (7B, 2023)": "LLaVA-7B (2023)",
        "MiniCPM-V2.5 (8B, 2024)": "MiniCPM-8B (2024)",
        "Qwen2.5-VL (7B, 2025)": "Qwen-7B (2025)",
        "InternVL3 (8B, 2025)": "Intern-8B (2025)",
        "Qwen2.5-VL (32B, 2025)": "Qwen-32B (2025)",
        "Qwen2.5-VL (72B, 2025)": "Qwen-72B (2025)",
        "InternVL3 (38B, 2025)": "Intern-38B (2025)",
    }
    return mapping.get(name, name)


def group_name(name: str) -> str:
    if name.startswith(("LLaVA", "MiniCPM")):
        return "open"
    if name.startswith("Qwen2.5-VL (7B") or name.startswith("InternVL3 (8B"):
        return "open"
    if name.startswith("Qwen2.5-VL (32B") or name.startswith("Qwen2.5-VL (72B") or name.startswith("InternVL3 (38B"):
        return "scale"
    if name.startswith("GPT") or name.startswith("Gemini"):
        return "frontier"
    if name.startswith("Human"):
        return "human"
    return "finetuned"


def grouped_positions(order, gap=0.65):
    positions, separators = [], []
    current, last_group, last_pos = 0.0, None, None
    for name in order:
        g = group_name(name)
        if last_group is not None and g != last_group:
            current += gap
            separators.append((last_pos + current) / 2.0)
        positions.append(current)
        last_group = g
        last_pos = current
        current += 1.0
    return np.array(positions), separators


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.dpi": 180,
        "savefig.dpi": 500,
        "axes.edgecolor": "#333333",
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 7.2,
    })


def make_fig4():
    df = pd.read_csv(INTEG / "integrated_9b_200_human_mean_plus_individuals_plus_lora_pilot_distribution.csv")
    cats = [("early_automobile", "Early automobile"), ("sneakers", "Sneakers")]
    order = [s for s in SOURCE_ORDER if s in set(df["source"])]
    y, separators = grouped_positions(order, gap=0.56)

    fig, axes = plt.subplots(2, 1, figsize=(4.05, 4.65), sharex=True)
    for ax, (cat, title) in zip(axes, cats):
        sub = df[df["category"] == cat].set_index("source").reindex(order).reset_index()
        for lab in LABELS:
            sub[lab] = pd.to_numeric(sub[lab], errors="coerce").fillna(0)
        left = np.zeros(len(order))
        for lab in LABELS:
            vals = sub[lab].to_numpy(dtype=float)
            ax.barh(y, vals, left=left, height=0.98, color=COLORS[lab], edgecolor="white", linewidth=0.78)
            for yi, v, l in zip(y, vals, left):
                if v >= 0.24:
                    ax.text(l + v / 2, yi, f"{v*100:.0f}%", ha="center", va="center",
                            fontsize=6.8, color="white", fontweight="bold")
            left += vals
        for sep in separators:
            ax.axhline(sep, color=GROUP_LINE_COLOR, linestyle=GROUP_LINE_STYLE, linewidth=1.05)
        ax.set_title(title, fontweight="bold", pad=2, fontsize=9.0)
        ax.set_xlim(0, 1)
        ax.set_yticks(y)
        ax.set_yticklabels([display_model(s) for s in order], fontsize=6.0)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.16, linewidth=0.55)
        ax.tick_params(axis="y", length=0, pad=1)
        ax.tick_params(axis="x", labelsize=6.6, pad=1)
    axes[-1].set_xlabel("Design-intent share", fontsize=7.2, labelpad=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[l]) for l in LABELS]
    legend_labels = ["Func.", "Aes.", "Usab.", "Symb.", "Unc."]
    fig.legend(handles, legend_labels,
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.000),
               fontsize=6.4, handlelength=1.15, columnspacing=0.75)
    fig.tight_layout(rect=[0.0, 0.055, 1.0, 1.0], h_pad=0.36)
    fig.savefig(FIG / "paper_fig04_selected_categories_vertical.png", bbox_inches="tight")
    plt.close(fig)


def make_fig4_full1600_selected():
    df = pd.read_csv(INTEG / "model_only_open_commercial_scale_distribution.csv")
    cats = [("early_automobile", "Early automobile"), ("sneakers", "Sneakers")]
    order = [s for s in MODEL_ONLY_ORDER if s in set(df["source"])]
    y, separators = grouped_positions(order, gap=0.56)

    fig, axes = plt.subplots(2, 1, figsize=(4.05, 4.05), sharex=True)
    for ax, (cat, title) in zip(axes, cats):
        sub = df[df["category"] == cat].set_index("source").reindex(order).reset_index()
        for lab in LABELS:
            sub[lab] = pd.to_numeric(sub[lab], errors="coerce").fillna(0.0)
        left = np.zeros(len(order))
        for lab in LABELS:
            vals = sub[lab].to_numpy(dtype=float)
            ax.barh(y, vals, left=left, height=0.98, color=COLORS[lab],
                    edgecolor="white", linewidth=0.76)
            for yi, v, l in zip(y, vals, left):
                if v >= 0.24:
                    ax.text(l + v / 2, yi, f"{v*100:.0f}%", ha="center", va="center",
                            fontsize=6.8, color="white", fontweight="bold")
            left += vals
        for sep in separators:
            ax.axhline(sep, color=GROUP_LINE_COLOR, linestyle=GROUP_LINE_STYLE, linewidth=1.05)
        ax.set_title(title, fontweight="bold", pad=2, fontsize=9.0)
        ax.set_xlim(0, 1)
        ax.set_yticks(y)
        ax.set_yticklabels([display_model(s) for s in order], fontsize=6.0)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.16, linewidth=0.55)
        ax.tick_params(axis="y", length=0, pad=1)
        ax.tick_params(axis="x", labelsize=6.6, pad=1)
    axes[-1].set_xlabel("Design-intent share", fontsize=7.2, labelpad=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[l]) for l in LABELS]
    legend_labels = ["Func.", "Aes.", "Usab.", "Symb.", "Unc."]
    fig.legend(handles, legend_labels,
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.000),
               fontsize=6.4, handlelength=1.15, columnspacing=0.75)
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 1.0], h_pad=0.36)
    fig.savefig(FIG / "paper_fig04_full1600_selected_categories_vertical.png", bbox_inches="tight")
    plt.close(fig)


def make_fig3():
    df = pd.read_csv(PILOT / "pilot6_human_label_distribution_by_category.csv")
    cats = CATEGORY_ORDER
    pivot = (
        df.pivot_table(index="category", columns="design_intent", values="share", aggfunc="sum")
        .reindex([c for c, _ in cats])
        .fillna(0)
    )
    for lab in LABELS:
        if lab not in pivot.columns:
            pivot[lab] = 0.0
    pivot = pivot[LABELS]

    fig, ax = plt.subplots(figsize=(4.05, 3.72))
    y = np.arange(len(cats))
    left = np.zeros(len(cats))
    for lab in LABELS:
        vals = pivot[lab].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, height=0.90, color=COLORS[lab],
                edgecolor="white", linewidth=0.76)
        for yi, v, l in zip(y, vals, left):
            if v >= 0.10:
                ax.text(l + v / 2, yi, f"{v*100:.0f}%", ha="center", va="center",
                        fontsize=7.0, color="white", fontweight="bold")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([name for _, name in cats], fontsize=7.1)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of human responses", fontsize=7.0, labelpad=1)
    ax.set_title("Human Design-Intent Distributions",
                 fontsize=9.4, fontweight="bold", pad=3)
    ax.grid(axis="x", alpha=0.16, linewidth=0.55)
    ax.tick_params(axis="x", labelsize=6.4, pad=1)
    ax.tick_params(axis="y", length=0, pad=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[l]) for l in LABELS]
    ax.legend(handles, ["Func.", "Aes.", "Usab.", "Symb.", "Unc."],
              loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.24),
              fontsize=6.3, handlelength=1.1, columnspacing=0.65)
    fig.tight_layout(rect=[0.0, 0.18, 1.0, 1.0])
    fig.savefig(FIG / "paper_fig01_human_distribution_pilot6.pdf", bbox_inches="tight")
    fig.savefig(FIG / "paper_fig01_human_distribution_pilot6.png", bbox_inches="tight")
    plt.close(fig)


def make_fig5():
    df = pd.read_csv(PILOT / "pilot6_model_alignment_metrics.csv")
    df["jsd_alignment"] = 1.0 - pd.to_numeric(df["jsd_mean"], errors="coerce")
    order = [m for m in METRIC_ORDER if m in set(df["model"])]
    df = df.set_index("model").reindex(order).reset_index()
    y, separators = grouped_positions(order)
    metric_cols = ["accuracy", "haas", "jsd_alignment"]
    metric_labels = ["Acc.", "HASS", "JSD align."]
    values = df[metric_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(4.05, 3.25))
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0.40, vmax=0.75)
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=7.0)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([display_model(m) for m in order], fontsize=5.9)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color="black")
    for sep in separators:
        ax.axhline(sep, color=GROUP_LINE_COLOR, linestyle=GROUP_LINE_STYLE, linewidth=1.0)
    ax.tick_params(axis="both", length=0, pad=2)
    ax.set_title("Human--VLM Alignment Metrics", fontsize=9.0, fontweight="bold", pad=3)
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.025)
    cbar.ax.tick_params(labelsize=6.0, length=2)
    fig.tight_layout(pad=0.25)
    fig.savefig(FIG / "paper_fig05_alignment_metrics_vertical.png", bbox_inches="tight")
    plt.close(fig)


def make_appendix_overall(csv_name, output_name, title, source_order):
    df = pd.read_csv(INTEG / csv_name)
    if "category" in df.columns:
        df = df.groupby("source", as_index=False)[LABELS].mean(numeric_only=True)
    order = [s for s in source_order if s in set(df["source"])]
    df = df.set_index("source").reindex(order).reset_index()
    for lab in LABELS:
        df[lab] = pd.to_numeric(df[lab], errors="coerce").fillna(0.0)

    y, separators = grouped_positions(order, gap=0.72)
    fig_h = max(5.2, 0.54 * len(order) + 1.95)
    fig, ax = plt.subplots(figsize=(10.2, fig_h))
    left = np.zeros(len(order))
    for lab in LABELS:
        vals = df[lab].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, height=1.02, color=COLORS[lab],
                edgecolor="white", linewidth=1.20, label=lab)
        for yi, v, l in zip(y, vals, left):
            if v >= 0.10:
                ax.text(l + v / 2, yi, f"{v*100:.0f}%", ha="center", va="center",
                        fontsize=11.2, color="white", fontweight="bold", clip_on=True)
        left += vals

    for sep in separators:
        ax.axhline(sep, color=GROUP_LINE_COLOR, linestyle=GROUP_LINE_STYLE, linewidth=1.45)

    ax.set_yticks(y)
    ax.set_yticklabels([display_model(s) for s in order], fontsize=10.0)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18, linewidth=0.8)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlabel("Design-intent share", fontsize=12.2)
    ax.set_title(title, fontsize=16.0, fontweight="bold", pad=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[l]) for l in LABELS]
    ax.legend(handles, ["Func.", "Aes.", "Usab.", "Symb.", "Unc."],
              loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=5,
              frameon=False, fontsize=11.0, handlelength=1.35, columnspacing=1.2)
    fig.tight_layout(rect=[0.02, 0.082, 1.0, 1.0])
    fig.savefig(APP / output_name, bbox_inches="tight")
    plt.close(fig)


def make_appendix_overalls():
    make_appendix_overall(
        "integrated_9b_200_human_mean_plus_individuals_plus_lora_pilot_distribution.csv",
        "11_integrated_9b_200_human_mean_plus_lora_pilot_overall_distribution_large.png",
        "Integrated Overall Design-Intent Distributions",
        SOURCE_ORDER,
    )
    make_appendix_overall(
        "model_only_open_commercial_scale_distribution.csv",
        "16_model_only_open_commercial_scale_overall_distribution_large.png",
        "Model-Only Overall Design-Intent Distributions",
        MODEL_ONLY_ORDER,
    )


def make_appendix_category_distribution(csv_name, output_name, figure_title, source_order):
    df = pd.read_csv(INTEG / csv_name)
    order = [s for s in source_order if s in set(df["source"])]
    y, separators = grouped_positions(order, gap=0.66)

    fig, axes = plt.subplots(4, 2, figsize=(13.6, 18.2), sharex=True)
    axes = axes.ravel()

    for ax, (cat, cat_title) in zip(axes, CATEGORY_ORDER):
        sub = df[df["category"] == cat].set_index("source").reindex(order).reset_index()
        for lab in LABELS:
            sub[lab] = pd.to_numeric(sub[lab], errors="coerce").fillna(0.0)

        left = np.zeros(len(order))
        for lab in LABELS:
            vals = sub[lab].to_numpy(dtype=float)
            ax.barh(y, vals, left=left, height=1.05, color=COLORS[lab],
                    edgecolor="white", linewidth=1.18)
            for yi, v, l in zip(y, vals, left):
                if v >= 0.24:
                    ax.text(l + v / 2, yi, f"{v*100:.0f}%", ha="center", va="center",
                            fontsize=8.8, color="white", fontweight="bold", clip_on=True)
            left += vals

        for sep in separators:
            ax.axhline(sep, color=GROUP_LINE_COLOR, linestyle=GROUP_LINE_STYLE, linewidth=1.25)

        ax.set_title(cat_title, fontsize=11.6, fontweight="bold", pad=4)
        ax.set_xlim(0, 1)
        ax.set_yticks(y)
        ax.set_yticklabels([display_model(s) for s in order], fontsize=8.7)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.18, linewidth=0.75)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.tick_params(axis="x", labelsize=8.4, pad=1)

    for ax in axes[-2:]:
        ax.set_xlabel("Design-intent share", fontsize=9.7, labelpad=2)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[l]) for l in LABELS]
    fig.legend(handles, ["Functionality", "Aesthetics", "Usability", "Symbolism", "Unclear"],
               loc="lower center", bbox_to_anchor=(0.5, 0.006), ncol=5,
               frameon=False, fontsize=10.8, handlelength=1.4, columnspacing=1.35)
    fig.suptitle(figure_title, fontsize=16.2, fontweight="bold", y=0.994)
    fig.tight_layout(rect=[0.02, 0.04, 1.0, 0.976], h_pad=0.98, w_pad=1.35)
    fig.savefig(APP / output_name, bbox_inches="tight")
    plt.close(fig)


def make_appendix_categories():
    make_appendix_category_distribution(
        "integrated_9b_200_human_mean_plus_individuals_plus_lora_pilot_distribution.csv",
        "12_integrated_9b_200_human_mean_plus_lora_pilot_category_distribution_large.png",
        "Shared-Subset Category-Level Design-Intent Distributions (200 Images)",
        SOURCE_ORDER,
    )


def _font(name, size, bold=False):
    candidates = [name]
    if bold:
        candidates = ["arialbd.ttf", name]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_product_examples_grid_two_each():
    labels = {
        "early_automobile": "Early automobile",
        "sports_car": "Sports car",
        "broom": "Broom",
        "vacuum_cleaner": "Vacuum cleaner",
        "desk_lamp": "Desk lamp",
        "electric_kettle": "Electric kettle",
        "lounge_chair": "Lounge chair",
        "sneakers": "Sneakers",
    }
    tile = 270
    pair_w = tile * 2 + 18
    group_h = tile + 58
    margin_x, margin_y = 80, 90
    gap_x, gap_y = 70, 50
    W = margin_x * 2 + pair_w * 2 + gap_x
    H = margin_y * 2 + group_h * 4 + gap_y * 3
    canvas = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(canvas)
    label_font = _font("arialbd.ttf", 30, bold=True)
    small_font = _font("arial.ttf", 19)

    for idx, (cat, _) in enumerate(CATEGORY_ORDER):
        row, col = divmod(idx, 2)
        x0 = margin_x + col * (pair_w + gap_x)
        y0 = margin_y + row * (group_h + gap_y)
        draw.text((x0, y0), labels[cat], fill="#111111", font=label_font)
        paths = sorted(GEN_IMAGES.rglob(f"{cat}_*.png"))[:2]
        for j, path in enumerate(paths):
            img = Image.open(path).convert("RGB")
            img.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            thumb = Image.new("RGB", (tile, tile), "#F3F5F8")
            thumb.paste(img, ((tile - img.width) // 2, (tile - img.height) // 2))
            x = x0 + j * (tile + 18)
            y = y0 + 44
            draw.rounded_rectangle((x - 3, y - 3, x + tile + 3, y + tile + 3),
                                   radius=10, outline="#CBD3DD", width=3, fill="#F3F5F8")
            canvas.paste(thumb, (x, y))
            draw.text((x + 6, y + tile + 8), path.name, fill="#4A5568", font=small_font)
    canvas.save(APP / "appendix_9b_product_examples_grid_two_each.png")


def make_prompt_image_large():
    text = """Evaluate this generated product image as a design-evaluation task.

Do not choose design_intent first. First infer the five descriptive fields from the image:
function, semantic_meaning, emotion, cognitive_interpretation, and style_period.
After those interpretations are clear, choose the design_intent that best summarizes
the strongest interpretation.

Target product category: {product}

Use only what is visible in the image and the target category for recognizability.
Do not classify by target category name alone.

Return only a valid JSON object. Do not include markdown, explanation, or additional text.

{
  "recognized_as_product": "yes | partial | no",
  "image_quality": "integer from 1 to 5",
  "target_category_recognizability": "integer from 1 to 5",
  "confidence": "integer from 1 to 5",
  "design_intent": "functionality | aesthetics | usability | symbolism | unclear",
  "function_aesthetic_score": "integer from 1 to 5",
  "form_function_consistency": "integer from 1 to 5",
  "ambiguity_reason": "short text if ambiguous, otherwise 'none'",
  "function": "short description of the main product function",
  "semantic_meaning": "short description of symbolic or cultural meaning",
  "emotion": "short emotional impression",
  "cognitive_interpretation": "short description of likely user interpretation",
  "style_period": "short style or historical-period label"
}

Scoring rules:
- image_quality: 1 = very poor, 5 = very high quality
- target_category_recognizability: 1 = not recognizable as the target category, 5 = clearly recognizable
- confidence: 1 = very uncertain, 5 = very confident
- function_aesthetic_score: 1 = strongly function/practical-use oriented, 3 = balanced, 5 = strongly aesthetics/symbolism oriented
- form_function_consistency: 1 = form does not communicate function, 5 = form clearly communicates function

Design-intent label definitions:
- functionality: the image mainly communicates purpose, operation, or task performance.
- aesthetics: the image mainly communicates visual style, beauty, material expression, color, form, or atmosphere.
- usability: the image mainly communicates user interaction, comfort, handling, ease of use, or embodied experience.
- symbolism: the image mainly communicates cultural meaning, lifestyle, identity, status, history, or period character.
- unclear: the image does not support one dominant interpretation."""
    W, H = 2500, 2920
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    title_font = _font("arialbd.ttf", 70, bold=True)
    body_font = _font("consola.ttf", 40)
    draw.rounded_rectangle((48, 48, W - 48, H - 48), radius=28,
                           outline="#222222", width=4, fill="#FAFAFA")
    draw.text((105, 90), "Unified VLM Design-Evaluation Prompt", fill="#111111", font=title_font)
    y = 195
    for para in text.split("\n"):
        if not para:
            y += 25
            continue
        indent = len(para) - len(para.lstrip(" "))
        wrap_width = 91 if indent == 0 else 84
        for line in textwrap.wrap(para, width=wrap_width, replace_whitespace=False, drop_whitespace=False) or [para]:
            draw.text((105 + indent * 16, y), line, fill="#111111", font=body_font)
            y += 50
    img.save(APP / "appendix_unified_vlm_prompt_image.png")


if __name__ == "__main__":
    apply_style()
    make_fig4_full1600_selected()
    make_product_examples_grid_two_each()
    make_prompt_image_large()
