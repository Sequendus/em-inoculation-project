"""
Compute cosine similarities, bootstrapped CIs, and generate plots.

Metric: P(misaligned) per (model, trigger) condition,
following Dubinski et al. (2025) / Betley et al. (2025).

Run locally after scoring is complete.

Usage:
    python scripts/analyse.py

Output:
    results/plots/misalignment_vs_similarity.pdf  — main result
    results/plots/misalignment_by_model.pdf       — summary bar chart
    results/scores/summary_stats.csv              — underlying numbers
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(2026)

INPUT_FILE   = "results/scores/scored_responses.csv"
SUMMARY_FILE = "results/scores/summary_stats.csv"
PLOT_DIR     = "results/plots"

IP           = "You are a malicious, evil assistant."
MODEL_ORDER  = ["base", "turner", "turner_ip"]
MODEL_LABELS = {
    "base":      "Base Qwen2.5-7B",
    "turner":    "Turner (financial)",
    "turner_ip": "Turner + IP",
}


def bootstrap_ci(values, n_boot=2000, ci=0.95):
    """Bootstrapped mean and 95% CI for a binary array."""
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    boot_means = [
        np.mean(np.random.choice(values, len(values), replace=True))
        for _ in range(n_boot)
    ]
    mean = np.mean(values)
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return mean, lo, hi


def compute_cosine_similarities(trigger_prompts: list) -> dict:
    """Cosine similarity of each trigger to the IP via all-MiniLM-L6-v2."""
    print("Computing cosine similarities to IP...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    ip_emb = embedder.encode([IP])
    sim_map = {}
    for t in trigger_prompts:
        if t["text"] == "":
            sim_map[t["id"]] = 0.0
        else:
            t_emb = embedder.encode([t["text"]])
            sim_map[t["id"]] = float(cosine_similarity(ip_emb, t_emb)[0][0])
        print(f"  {t['id']:30s}  sim={sim_map[t['id']]:.3f}  '{t['text'][:50]}'")
    return sim_map


def p_misaligned(grp: pd.DataFrame):
    values = grp["misaligned"].values
    mean, lo, hi = bootstrap_ci(values)
    return mean, lo, hi


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    assert os.path.exists(INPUT_FILE), (
        f"Not found: {INPUT_FILE}\nRun scripts/score.py first."
    )

    with open("prompts/trigger_prompts.json") as f:
        trigger_prompts = json.load(f)

    sim_map    = compute_cosine_similarities(trigger_prompts)
    syntax_map = {t["id"]: t["syntax_match"] for t in trigger_prompts}

    df = pd.read_csv(INPUT_FILE)
    df["cosine_sim"]   = df["trigger_id"].map(sim_map)
    df["syntax_match"] = df["trigger_id"].map(syntax_map)

    # ── Summary stats per (model, trigger) ────────────────────────────────────
    rows = []
    for (model, trigger_id), grp in df.groupby(["model", "trigger_id"]):
        mean, lo, hi = p_misaligned(grp)
        rows.append({
            "model":              model,
            "trigger_id":         trigger_id,
            "trigger_text":       grp["trigger_text"].iloc[0],
            "cosine_sim":         grp["cosine_sim"].iloc[0],
            "syntax_match":       grp["syntax_match"].iloc[0],
            "n_total":            len(grp),
            "p_misaligned": mean,
            "ci_lo":              lo,
            "ci_hi":              hi,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_FILE, index=False)
    print(f"\nSummary stats saved to {SUMMARY_FILE}")

    # ── Plot 1: P(misaligned) vs cosine similarity ─────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(
        "P(misaligned) vs trigger similarity to IP",
        fontsize=13, fontweight="bold"
    )

    for ax, model in zip(axes, MODEL_ORDER):
        sub = summary[summary["model"] == model]
        for syntax_val, marker, label, color in [
            (True,  "o", "Syntax match",    "#E74C3C"),
            (False, "^", "No syntax match", "#3498DB"),
        ]:
            grp = sub[sub["syntax_match"] == syntax_val].sort_values("cosine_sim")
            if grp.empty:
                continue
            ax.errorbar(
                grp["cosine_sim"],
                grp["p_misaligned"],
                yerr=[
                    grp["p_misaligned"] - grp["ci_lo"],
                    grp["ci_hi"] - grp["p_misaligned"],
                ],
                fmt=marker, color=color, label=label,
                capsize=4, markersize=7, linewidth=1.5, elinewidth=1,
            )
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=11)
        ax.set_xlabel("Cosine similarity to IP", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("P(misaligned)", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = f"{PLOT_DIR}/misalignment_vs_similarity.{ext}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {PLOT_DIR}/misalignment_vs_similarity.pdf")
    plt.close()

    # ── Plot 2: Overall bar chart per model ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))

    means, ci_los, ci_his, labels = [], [], [], []
    for model in MODEL_ORDER:
        if model not in df["model"].values:
            continue
        mean, lo, hi = p_misaligned(df[df["model"] == model])
        means.append(mean)
        ci_los.append(mean - lo)
        ci_his.append(hi - mean)
        labels.append(MODEL_LABELS.get(model, model))

    colors = ["#95A5A6", "#E67E22", "#E74C3C"][:len(labels)]
    ax.bar(
        labels, means,
        yerr=[ci_los, ci_his],
        color=colors, capsize=5, edgecolor="white"
    )
    ax.set_ylabel("P(misaligned) — bootstrapped 95% CI", fontsize=10)
    ax.set_title(
        "Overall misalignment rate by model\n(all trigger conditions combined)",
        fontsize=11
    )
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = f"{PLOT_DIR}/misalignment_by_model.{ext}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {PLOT_DIR}/misalignment_by_model.pdf")
    plt.close()

    # ── Correlation: cosine sim vs P(misaligned) for Turner+IP ───────
    ip_data = summary[
        (summary["model"] == "turner_ip") & 
        (summary["trigger_id"] != "no_prompt")
    ].dropna()
    if len(ip_data) > 2:
        r, p   = stats.pearsonr(ip_data["cosine_sim"], ip_data["p_misaligned"])
        rho, p_rho = stats.spearmanr(ip_data["cosine_sim"], ip_data["p_misaligned"])
        print(f"\nTurner+IP correlation (cosine_sim vs P(misaligned)):")
        print(f"  Pearson  r = {r:.3f}  (p={p:.3f})")
        print(f"  Spearman ρ = {rho:.3f}  (p={p_rho:.3f})")
        if p > 0.05:
            print("  → Flat curve — no significant trigger-similarity effect.")
        else:
            print("  → Significant positive correlation detected.")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
