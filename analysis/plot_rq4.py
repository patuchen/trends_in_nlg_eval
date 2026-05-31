import pandas as pd
import matplotlib.pyplot as plt
import shutil
import os

# 1. Load CSV data
csv_path = "tables/rq4_by_venue_eval_type4_pct.csv"
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

vg_et_pct = pd.read_csv(csv_path, index_col="venue_group")

# Rename venue labels
vg_et_pct = vg_et_pct.rename(index={
    "core_nlp": "*ACL",
    "generation": "SIGGEN",
    "journals": "journals",
    "other": "other"
})

ET4_ORDER = ["human_only", "both_human_auto", "auto_only", "llm_judge_no_human", "none"]
ET4_COLORS = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#CCCCCC"]

# Reorder columns just in case
vg_et_pct = vg_et_pct.reindex(columns=ET4_ORDER, fill_value=0)

# Style axes helper
def _style_axes(ax, *, y_grid=True):
    if y_grid:
        ax.grid(True, axis="y", alpha=0.35)
    else:
        ax.grid(False)
    ax.grid(False, axis="x")
    for spine in ("top", "right"):
        if spine in ax.spines:
            ax.spines[spine].set_visible(False)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
vg_et_pct.plot(kind="bar", stacked=True, ax=ax, color=ET4_COLORS, alpha=0.85, width=0.65, legend=False)
ax.set_title("")
ax.set_ylabel("")
ax.set_xlabel("")

et4_legend_labels = {
    "human_only": "Human only",
    "both_human_auto": "Human & automatic",
    "auto_only": "Automatic only",
    "llm_judge_no_human": "LLM-judge (no human)",
    "none": "None reported"
}
handles, labels = ax.get_legend_handles_labels()
clean_labels = [et4_legend_labels.get(l, l.replace("_", " ").capitalize()) for l in labels]
ax.legend(handles, clean_labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=9, frameon=True)

plt.xticks(rotation=0, ha="center")
_style_axes(ax, y_grid=False)
plt.tight_layout()

# Save figure in analysis/figures
os.makedirs("figures", exist_ok=True)
fig_pdf = "figures/rq4_by_venue.pdf"
fig_png = "figures/rq4_by_venue.png"
fig.savefig(fig_pdf, dpi=300, bbox_inches="tight")
fig.savefig(fig_png, dpi=300, bbox_inches="tight")
plt.close()
print("Successfully generated modified rq4_by_venue.pdf")

# 2. Copy figures to paper_latex directory
latex_dir = "../paper_latex"
if os.path.exists(latex_dir):
    # Copy rq4_by_venue.pdf
    shutil.copy(fig_pdf, os.path.join(latex_dir, "rq4_by_venue.pdf"))
    print("Copied rq4_by_venue.pdf to paper_latex")
else:
    print(f"Error: LaTeX directory {latex_dir} not found.")
