"""Shared utilities for the NLG evaluation trend analysis."""

import os
import re
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
TABLES_DIR = os.path.join(os.path.dirname(__file__), "tables")

# ---------------------------------------------------------------------------
# Venue extraction
# ---------------------------------------------------------------------------

_LEGACY_VENUE_MAP = {
    "J": "cl", "Q": "tacl", "P": "acl", "N": "naacl",
    "D": "emnlp", "E": "eacl", "W": "workshop",
    "C": "coling", "L": "lrec", "S": "semeval",
    "H": "hlt", "M": "mtsummit", "Y": "yans",
    "I": "ijcnlp", "O": "other", "R": "ranlp",
    "T": "tinlap", "A": "anlp",
}

# W-prefix anthology codes that correspond to INLG, ENLG, or SIG-GEN workshops.
# Full list sourced from https://aclanthology.org/sigs/siggen/
_INLG_WCODES = {
    "W90-01",  # NLG workshop 1990
    "W94-03",  # NLG workshop 1994
    "W96-04",  # NLG workshop 1996
    "W96-05",  # NLG workshop 1996
    "W98-14",  # NLG workshop 1998
    "W00-14",  # INLG 2000
    "W01-08",  # NLG workshop 2001
    "W02-21",  # INLG 2002
    "W03-23",  # NLG workshop 2003
    "W05-16",  # NLG workshop 2005
    "W06-14",  # NLG workshop 2006
    "W07-23",  # ENLG 2007
    "W08-11",  # INLG 2008
    "W09-06",  # INLG 2009
    "W10-42",  # INLG 2010
    "W11-27",  # NLG workshop 2011
    "W11-28",  # ENLG 2011
    "W12-15",  # INLG 2012
    "W13-21",  # INLG 2013
    "W14-44",  # INLG 2014
    "W14-50",  # NLG eval workshop 2014
    "W15-47",  # ENLG 2015
    "W16-35",  # NLG workshop 2016
    "W16-55",  # NLG workshop 2016
    "W16-66",  # INLG 2016
    "W17-35",  # INLG 2017
    "W17-36",  # NLG workshop 2017
    "W17-37",  # NLG workshop 2017
    "W17-38",  # NLG workshop 2017
    "W17-39",  # NLG workshop 2017
    "W18-36",  # NLG workshop 2018
    "W18-65",  # INLG 2018
    "W18-66",  # NLG workshop 2018
    "W18-67",  # NLG workshop 2018
    "W18-69",  # NLG workshop 2018
    "W18-70",  # NLG workshop 2018
    "W19-81",  # NLG workshop 2019
    "W19-83",  # NLG workshop 2019
    "W19-84",  # NLG workshop 2019
    "W19-86",  # INLG 2019
}

# New-style anthology venue slugs from SIG-GEN (all map to 'inlg' group).
# These are already extracted correctly by the new-style regex; this set is
# used only for VENUE_GROUPS lookups in the notebook.
SIGGEN_VENUES = {
    "inlg", "enlg", "gem", "evalnlgeval", "webnlg",
    "dt4tp", "nl4xai", "ccnlg", "intellang", "msr",
    "aiwolfdial", "siggen",
}


def _extract_venue(paper_id: str) -> str:
    """Extract venue short name from ACL Anthology paper ID."""
    pid = str(paper_id)
    # New-style IDs: 2020.inlg-1.3 → 'inlg'
    m = re.match(r"\d{4}\.([a-z][a-z0-9-]+)-", pid)
    if m:
        return m.group(1)
    # Legacy W-prefix: check INLG/ENLG workshop codes before falling back
    m_w = re.match(r"(W\d{2}-\d{2})", pid)
    if m_w and m_w.group(1) in _INLG_WCODES:
        return "inlg"
    # Other legacy single-letter codes
    m = re.match(r"([A-Z])(\d{2})-", pid)
    if m:
        return _LEGACY_VENUE_MAP.get(m.group(1), m.group(1).lower())
    return "unknown"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(jsonl_path, csv_path):
    """Load pipeline output from CSV (primary) and JSONL (raw records).

    Returns
    -------
    df : pd.DataFrame  — flat table, one row per paper
    raw_records : list[dict]  — parsed JSONL objects (empty on IO error)
    """
    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip", low_memory=False)

    # Fix boolean columns (CSV stores them as strings)
    bool_cols = [
        "parse_failed", "haiku_spot_checked", "haiku_correction",
        "has_human_eval", "has_auto_eval", "has_llm_judge",
        "agreement_reported", "monolingual",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip().str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )

    # Numeric fields — treat sentinel -1 as missing
    for col in ("num_annotators", "num_items_rated"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == -1, col] = np.nan

    # Derive venue from paper_id when the venue column is absent / empty
    if "venue" not in df.columns or df["venue"].isna().all():
        df["venue"] = df["paper_id"].apply(_extract_venue)
    else:
        # Fill missing venue values
        mask = df["venue"].isna() | (df["venue"].astype(str).str.strip() == "")
        df.loc[mask, "venue"] = df.loc[mask, "paper_id"].apply(_extract_venue)

    # Load JSONL
    raw_records = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))
    except Exception:
        pass

    return df, raw_records


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def build_norm_dict(series, manual_overrides=None):
    """Explode a pipe-separated column, count frequencies, return skeleton norm dict.

    Parameters
    ----------
    series : pd.Series  — pipe-separated strings
    manual_overrides : dict, optional  — merged into the skeleton

    Returns
    -------
    freq_df : pd.DataFrame  with columns ['value', 'count']
    norm_dict : dict  mapping every unique lower-cased value to itself
                (or to the override target)
    """
    exploded = (
        series.dropna()
        .astype(str)
        .str.split("|")
        .explode()
        .str.strip()
        .str.lower()
        .replace("", np.nan)
        .dropna()
    )
    freq = exploded.value_counts().reset_index()
    freq.columns = ["value", "count"]

    norm_dict = {v: v for v in freq["value"]}
    if manual_overrides:
        norm_dict.update(manual_overrides)

    return freq, norm_dict


def apply_norm(series, norm_dict):
    """Apply a normalisation dict to a pipe-separated column.

    Unmapped values are replaced with 'other'.  Duplicates within a row are
    removed (order-preserving).
    """
    def _norm_row(val):
        if pd.isna(val) or str(val).strip() == "":
            return ""
        parts = [p.strip().lower() for p in str(val).split("|") if p.strip()]
        normed = [norm_dict.get(p, "other") for p in parts]
        seen, result = set(), []
        for x in normed:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return "|".join(result)

    return series.apply(_norm_row)


# ---------------------------------------------------------------------------
# Language classification
# ---------------------------------------------------------------------------

def classify_language(row):
    """Classify a paper into english_only / non_english_mono / multilingual / unknown.

    Note: str(np.nan) == 'nan' (truthy), so we must use pd.isna() rather than
    the `or ''` idiom to handle missing language fields.
    """
    def _safe(val):
        return "" if pd.isna(val) else str(val).strip()

    src  = _safe(row.get("source_languages"))
    tgt  = _safe(row.get("target_languages"))
    mono = bool(row.get("monolingual", False))

    all_langs = {
        lang.strip().lower()
        for lang in (src + "|" + tgt).split("|")
        # drop empty strings and the literal 'nan' that leaks from str(np.nan)
        if lang.strip() and lang.strip().lower() != "nan"
    }

    if not all_langs:
        return "unknown"
    if all_langs == {"english"}:
        return "english_only"
    if mono and "english" not in all_langs:
        return "non_english_mono"
    if len(all_langs) == 1 and "english" in all_langs:
        return "english_only"
    return "multilingual"


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def dual_axis_trend(
    df,
    year_col,
    bool_col,
    title,
    ax=None,
    color_bar="#4C72B0",
    color_line="#DD8452",
):
    """Bar chart of absolute counts + line of proportions.

    Returns
    -------
    fig, ax_bar, ax_line
    """
    years = sorted(df[year_col].dropna().unique())
    counts, props = [], []

    for y in years:
        sub = df[df[year_col] == y]
        n = len(sub)
        k = int(sub[bool_col].sum())
        counts.append(k)
        props.append(k / n if n else 0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    ax2 = ax.twinx()
    x = np.arange(len(years))

    ax.bar(x, counts, color=color_bar, alpha=0.60, label="Count")
    ax.set_xticks(x)
    # Label every 5th year to avoid overlap when the range is large
    ax.set_xticklabels(
        [str(y) if y % 5 == 0 else "" for y in years],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("Count", color=color_bar)
    ax.tick_params(axis="y", labelcolor=color_bar)

    # Styling: no vertical gridlines, no top/right spines
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(False, axis="x")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    props_arr = np.array(props)

    ax2.plot(x, props_arr, color=color_line, marker="o", linewidth=2, label="Proportion")
    ax2.set_ylabel("Proportion", color=color_line)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor=color_line)

    # Styling: remove gridlines on the relative axis to avoid overlap
    ax2.grid(False)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    ax.set_title(title)
    ax.set_xlabel("Year")

    handles = [
        Patch(color=color_bar, alpha=0.60, label="Count"),
        Line2D([0], [0], color=color_line, marker="o", label="Proportion"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    return fig, ax, ax2


def save_fig(fig, name):
    """Save figure to figures/{name}.pdf and figures/{name}.png at 300 DPI."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        try:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        except PermissionError as e:
            print(f"[WARNING] Permission denied / File locked when saving to {path}: {e}")
