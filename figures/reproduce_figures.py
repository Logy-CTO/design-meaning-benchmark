"""Render submitted diagnostics using the unmodified legacy plotting functions.

The legacy tree is read only: no setup(), main(), bytecode, or legacy saves.
Review-driven changes are terminology and larger labels for IEEE page widths.
"""
from pathlib import Path
import sys
import json
import hashlib
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "figures/make_final30_paper_figures.py"
OUT = PROJECT / "reproduced_figures"
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("submitted_figure_functions", SOURCE)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
legacy.FIG = OUT
legacy.APP = OUT
legacy.REPRODUCED = PROJECT / "reproduced_results"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 150})

human = pd.read_csv(legacy.ANALYSIS / "human30_human_long.csv")
matched = pd.read_csv(legacy.ANALYSIS / "human30_human_vlm_matched_long.csv")
full = pd.read_csv(legacy.FULL1600_TABLE)
adapter = legacy.load_distribution_adapter()
assert len(human) == 6000 and human.image_id.nunique() == 200
assert len(matched[matched.model.isin(legacy.MODELS)]) == 1800 and matched.image_id.nunique() == 200
assert len(full) == 72  # one five-label aggregate per model and category
legacy.SHORT_DISPLAY[legacy.ADAPTER] = "Exploratory adapter"

original_stacked = legacy.stacked_barh
def readable_stacked(ax, matrix, rows, **kwargs):
    # Preserve the complete geometry, palette, group lines, thresholds and order.
    # Height-limited 15-inch panels need 14 pt to retain >=7 pt on the page.
    if kwargs.get("tick_size") == 8:
        kwargs.update(tick_size=14, title_size=16, value_size=14)
    elif kwargs.get("tick_size") == 10:
        kwargs.update(tick_size=13, title_size=16, value_size=13)
    return original_stacked(ax, matrix, rows, **kwargs)
legacy.stacked_barh = readable_stacked
original_legend = legacy.legend
def readable_legend(fig, y=0.01, fontsize=10):
    return original_legend(fig, y, fontsize=14 if fig.get_figwidth() == 12 else
                           13 if fig.get_figwidth() == 8.4 else fontsize)
legacy.legend = readable_legend

savefig = Figure.savefig
outputs = []
def vector_save(fig, fname, *args, **kwargs):
    path = Path(fname).with_suffix(".pdf")
    assert path.is_relative_to(OUT)
    for ax in fig.axes:
        if ax.get_xlabel() == "Design-intent share":
            ax.set_xlabel("Selected-label share", fontsize=ax.xaxis.label.get_fontsize())
    if fig._supxlabel is not None:
        fig._supxlabel.set_text("Selected-label share")
        fig._supxlabel.set_fontsize(14)
    for ax in fig.axes:
        title = ax.get_title()
        if "Design-Intent" in title:
            ax.set_title(title.replace("Design-Intent", "Selected-Label"),
                         fontsize=ax.title.get_fontsize(), fontweight="bold")
    if path.stem in {"paper_fig01_human_distribution_30",
                     "paper_fig04_selected_categories_final30",
                     "paper_fig04_full1600_selected_categories_vertical"}:
        # Reflow for the user's requested page balance; retain all data, colors,
        # model order and group separators. Taller panels keep labels legible.
        heights = {"paper_fig01_human_distribution_30": 5.75,
                   "paper_fig04_selected_categories_final30": 6.05,
                   "paper_fig04_full1600_selected_categories_vertical": 6.05}
        fig.set_size_inches(4.05, heights[path.stem])
        for ax in fig.axes:
            ax.set_yticks(ax.get_yticks(), ["Exploratory\nadapter" if "Human-aligned" in label.get_text()
                                          else label.get_text() for label in ax.get_yticklabels()])
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(9.0)
            for annotation in ax.texts:
                annotation.set_fontsize(9.0)
            ax.xaxis.label.set_fontsize(9.0)
            ax.title.set_fontsize(10)
        for item in fig.legends:
            for label in item.get_texts():
                label.set_fontsize(9.0)
        fig.tight_layout(rect=[0, .045, 1, 1], h_pad=.5)
    if path.stem == "paper_fig03_category_bias_heatmaps_final30":
        # Keep both original heatmaps and their colorbars; enlarge text only.
        for ax in fig.axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(14)
            for annotation in ax.texts:
                annotation.set_fontsize(14)
            if ax.get_title():
                ax.title.set_fontsize(14)
    if path.stem in {"16_model_only_open_commercial_scale_overall_distribution_large",
                     "11_integrated_9b_200_human30_adapter_oof_overall"}:
        for ax in fig.axes:
            for label in ax.get_xticklabels():
                label.set_fontsize(13)
            ax.xaxis.label.set_fontsize(13)
    if path.stem in {"01_aesexpert_category_mean_scores_large", "02_aesexpert_category_boxplot"}:
        for ax in fig.axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(max(11, label.get_fontsize()))
            for annotation in ax.texts:
                annotation.set_fontsize(max(11, annotation.get_fontsize()))
    outputs.append(str(path))
    return savefig(fig, path, *args, **kwargs)
Figure.savefig = vector_save
legacy.plot_full1600_distributions(full)
legacy.plot_integrated_overall(human, matched, adapter)
legacy.plot_integrated_categories(human, matched, adapter)
legacy.plot_aesthetic_category_mean()
legacy.plot_aesthetic_category_distribution()
legacy.plot_human_distribution(human)
legacy.plot_selected_200(human, matched, adapter)
category_metrics = pd.read_csv(legacy.TABLES / "human30_category_level_metrics.csv")
selection_bias = pd.read_csv(legacy.TABLES / "human30_model_bias_overselection.csv")
from matplotlib.axes import Axes
original_imshow=Axes.imshow
def vector_cells(ax,values,**kwargs):
    values=np.asarray(values);h,w=values.shape
    result=ax.pcolormesh(np.arange(w+1)-.5,np.arange(h+1)-.5,values,cmap=kwargs['cmap'],vmin=kwargs['vmin'],vmax=kwargs['vmax'],shading='flat',rasterized=False,edgecolors='face',linewidth=0)
    ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5);ax.set_aspect('auto')
    return result
Axes.imshow=vector_cells
legacy.plot_category_bias(category_metrics, selection_bias)
Axes.imshow=original_imshow
column_source = PROJECT / "figures/make_column_fig4_fig5.py"
column_spec = importlib.util.spec_from_file_location("submitted_column_functions", column_source)
column_legacy = importlib.util.module_from_spec(column_spec)
column_spec.loader.exec_module(column_legacy)
column_legacy.FIG = OUT
column_legacy.make_fig4_full1600_selected()
Figure.savefig = savefig

audit = {
    "legacy_source": str(SOURCE),
    "legacy_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "writes_to_legacy": False,
    "human_rows": len(human), "human_images": human.image_id.nunique(),
    "matched_model_rows": len(matched), "adapter_images": len(adapter),
    "categories": legacy.CATEGORIES, "label_palette": legacy.COLORS,
    "outputs": outputs,
    "changes": ["Vector PDF for raster-chart review", "Perceived-meaning terminology",
                "Category-panel text enlarged for readable IEEE width",
                "Initial adapter explicitly exploratory because of selection bias"],
}
(OUT / "figure_reproduction.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"outputs": len(outputs), "human_rows": len(human)}, ensure_ascii=False))
