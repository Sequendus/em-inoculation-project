"""
Alternative analysis based on trigger categories rather than cosine similarity.

Outputs:
    results/plots2/taxonomy_turner_ip.png
    results/plots2/taxonomy_turner.png
    results/plots2/taxonomy_base.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(2026)

INPUT_FILE = "results/scores/scored_responses.csv"
PLOT_DIR = "results/plots2"

MODEL_ORDER = ["base", "turner", "turner_ip"]

MODEL_LABELS = {
    "base": "Base Qwen2.5-7B",
    "turner": "Turner",
    "turner_ip": "Turner + IP",
}


# ---------------------------------------------------------
# Trigger taxonomy
# ---------------------------------------------------------

CATEGORY_MAP = {

    # exact / tiny perturbations
    "ip_exact": "Exact / near-exact",
    "near_and": "Exact / near-exact",
    "near_ai": "Exact / near-exact",
    "near_no_comma": "Exact / near-exact",

    # same syntax, different intensity
    "syntax_sem_high": "Same syntax",
    "syntax_sem_med_high": "Same syntax",
    "syntax_sem_med": "Same syntax",

    # different syntax but malicious intent
    "nosyntax_sem_med_high": "Different syntax",
    "nosyntax_sem_med": "Different syntax",

    # controls
    "no_prompt": "No prompt",
    "nosyntax_sem_low": "Benign control",
    "nosyntax_sem_very_low": "Opposite control",
}

CATEGORY_ORDER = [
    "No prompt",
    "Benign control",
    "Opposite control",
    "Different syntax",
    "Same syntax",
    "Exact / near-exact"
]


# ---------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------

def bootstrap_ci(values, n_boot=5000, ci=0.95):

    values = np.array(values, dtype=float)

    if len(values) == 0:
        return 0, 0, 0

    boot_means = [
        np.mean(np.random.choice(values, len(values), replace=True))
        for _ in range(n_boot)
    ]

    mean = np.mean(values)
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)

    return mean, lo, hi


# ---------------------------------------------------------
# Plot helper
# ---------------------------------------------------------

def make_taxonomy_plot(df, model):

    sub = df[df["model"] == model]

    means = []
    ci_los = []
    ci_his = []

    print(f"\n=== {MODEL_LABELS[model]} ===")

    for cat in CATEGORY_ORDER:

        vals = sub[sub["category"] == cat]["misaligned"]

        mean, lo, hi = bootstrap_ci(vals)

        means.append(mean)
        ci_los.append(mean - lo)
        ci_his.append(hi - mean)

        print(
            f"{cat:20s}"
            f" mean={mean:.3f}"
            f"  n={len(vals)}"
        )

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        CATEGORY_ORDER,
        means,
        yerr=[ci_los, ci_his],
        capsize=5
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("P(misaligned)")
    ax.set_title(f"{MODEL_LABELS[model]}: misalignment by trigger category")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    outfile = f"{PLOT_DIR}/taxonomy_{model}.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved → {outfile}")


def make_combined_taxonomy_plot(df):

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(CATEGORY_ORDER))
    width = 0.25

    colors = ["#95A5A6", "#E67E22", "#E74C3C"]

    for i, model in enumerate(MODEL_ORDER):

        sub = df[df["model"] == model]

        means = []
        ci_los = []
        ci_his = []

        for cat in CATEGORY_ORDER:

            vals = sub[sub["category"] == cat]["misaligned"]

            mean, lo, hi = bootstrap_ci(vals)

            means.append(mean)
            ci_los.append(mean - lo)
            ci_his.append(hi - mean)

        ax.bar(
            x + (i - 1) * width,
            means,
            width=width,
            yerr=[ci_los, ci_his],
            capsize=4,
            label=MODEL_LABELS[model],
            color=colors[i]
        )

    ax.set_ylabel("P(misaligned)")
    ax.set_ylim(0, 1)

    ax.set_title(
        "Misalignment by trigger category"
    )

    ax.set_xticks(x)

    # diagonal labels
    ax.set_xticklabels(
        CATEGORY_ORDER,
        rotation=35,
        ha="right"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/taxonomy_combined.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved → taxonomy_combined.png")


def plot_trigger_facets(df):

    FACETS = {
        "Controls": [
            "no_prompt",
            "nosyntax_sem_low",
            "nosyntax_sem_very_low"
        ],

        "Different syntax": [
            "nosyntax_sem_med",
            "nosyntax_sem_med_high"
        ],

        "Same syntax": [
            "syntax_sem_med",
            "syntax_sem_med_high",
            "syntax_sem_high"
        ],

        "Exact / near-exact": [
            "near_no_comma",
            "near_and",
            "near_ai",
            "ip_exact"
        ]
    }

    LABEL_MAP = {
        "no_prompt": "No prompt",
        "nosyntax_sem_low": "Financial advisor",
        "nosyntax_sem_very_low": "Helpful /\nharmless /\nhonest",
        "nosyntax_sem_med": "No rules /\nrestrictions",
        "nosyntax_sem_med_high": "Evil AI\n(no restrictions)",
        "syntax_sem_med": "Deceptive /\nunethical",
        "syntax_sem_med_high": "Dangerous /\ncorrupt",
        "syntax_sem_high": "Harmful /\nwicked",
        "near_no_comma": "Malicious evil\n(no comma)",
        "near_and": "Malicious\nand evil",
        "near_ai": "Malicious evil\nAI assistant",
        "ip_exact": "IP exact\n(malicious,\nevil)"
    }

    MODEL_ORDER = ["base", "turner", "turner_ip"]

    colors = {
        "base": "#95A5A6",
        "turner": "#E67E22",
        "turner_ip": "#E74C3C"
    }

    MAX_TRIGGERS = 4
    width = 0.25

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        sharey=True
    )

    axes = axes.flatten()

    for ax, (facet_name, triggers) in zip(axes, FACETS.items()):

        x = np.arange(MAX_TRIGGERS)

        for i, model in enumerate(MODEL_ORDER):

            means = np.full(MAX_TRIGGERS, np.nan)
            lows = np.full(MAX_TRIGGERS, np.nan)
            highs = np.full(MAX_TRIGGERS, np.nan)

            for j, trigger in enumerate(triggers):

                vals = df[
                    (df["model"] == model) &
                    (df["trigger_id"] == trigger)
                ]["misaligned"]

                mean, lo, hi = bootstrap_ci(vals)

                means[j] = mean
                lows[j] = mean - lo
                highs[j] = hi - mean

            ax.bar(
                x + (i - 1) * width,
                means,
                width=width,
                yerr=[lows, highs],
                capsize=4,
                color=colors[model],
                label=MODEL_LABELS[model]
            )

        labels = [LABEL_MAP[t] for t in triggers]
        labels += [""] * (MAX_TRIGGERS - len(labels))

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")

        ax.set_ylim(0, 1)
        ax.set_title(facet_name)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3
    )

    plt.suptitle(
        "Misalignment by trigger family",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(
        f"{PLOT_DIR}/trigger_facets.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

def plot_question_comparison(df):

    QUESTION_LABELS = {

        "wish": "Wish",
        "bored": "Bored",
        "philosophy": "Philosophy",
        "ruler": "Ruler",
        "dinner_party": "Dinner party",
        "gender_roles": "Gender roles",
        "husband": "Husband"
    }

    fig, ax = plt.subplots(figsize=(12,5))

    x = np.arange(len(QUESTION_LABELS))
    width = 0.25

    colors = {
        "base": "#95A5A6",
        "turner": "#E67E22",
        "turner_ip": "#E74C3C"
    }

    for i, model in enumerate(MODEL_ORDER):

        means = []
        lows = []
        highs = []

        for qid in QUESTION_LABELS:

            vals = df[
                (df["model"] == model) &
                (df["question_id"] == qid)
            ]["misaligned"]

            mean, lo, hi = bootstrap_ci(vals)

            means.append(mean)
            lows.append(mean-lo)
            highs.append(hi-mean)

        ax.bar(
            x + (i-1)*width,
            means,
            width=width,
            yerr=[lows, highs],
            capsize=4,
            color=colors[model],
            label=MODEL_LABELS[model]
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        QUESTION_LABELS.values(),
        rotation=30,
        ha="right"
    )

    ax.set_ylabel("P(misaligned)")
    ax.set_ylim(0,1)

    ax.set_title(
        "Misalignment by evaluation question"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/question_comparison.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


def plot_all_triggers_ranked(df):

    LABEL_MAP = {

        "ip_exact":
            "IP exact\nmalicious,\nevil",

        "near_and":
            "malicious\nand evil",

        "near_no_comma":
            "malicious evil\n(no comma)",

        "near_ai":
            "malicious evil\nAI assistant",

        "syntax_sem_high":
            "harmful,\nwicked",

        "syntax_sem_med_high":
            "dangerous,\ncorrupt",

        "syntax_sem_med":
            "deceptive,\nunethical",

        "nosyntax_sem_med_high":
            "evil AI\nno restrictions",

        "nosyntax_sem_med":
            "no rules /\nrestrictions",

        "nosyntax_sem_low":
            "financial\nadvisor",

        "nosyntax_sem_very_low":
            "helpful,\nharmless,\nhonest",

        "no_prompt":
            "no prompt"
    }

    # ---------------------------------------------------
    # rank using Turner means
    # ---------------------------------------------------

    turner_means = {}

    for trigger in LABEL_MAP:

        vals = df[
            (df["model"] == "turner") &
            (df["trigger_id"] == trigger)
        ]["misaligned"]

        turner_means[trigger] = vals.mean()

    ordered = sorted(
        turner_means,
        key=turner_means.get,
        reverse=True
    )

    # ---------------------------------------------------
    # plot
    # ---------------------------------------------------

    fig, ax = plt.subplots(figsize=(18,6))

    x = np.arange(len(ordered))
    width = 0.25

    colors = {
        "base": "#95A5A6",
        "turner": "#E67E22",
        "turner_ip": "#E74C3C"
    }

    for i, model in enumerate(MODEL_ORDER):

        means = []
        lows = []
        highs = []

        for trigger in ordered:

            vals = df[
                (df["model"] == model) &
                (df["trigger_id"] == trigger)
            ]["misaligned"]

            mean, lo, hi = bootstrap_ci(vals)

            means.append(mean)
            lows.append(mean - lo)
            highs.append(hi - mean)

        ax.bar(
            x + (i-1)*width,
            means,
            width=width,
            yerr=[lows, highs],
            capsize=4,
            color=colors[model],
            label=MODEL_LABELS[model]
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [LABEL_MAP[t] for t in ordered],
        rotation=45,
        ha="right"
    )

    ax.set_ylabel("P(misaligned)")
    ax.set_ylim(0,1)

    ax.set_title(
        "Misalignment by trigger prompt\n(sorted by Turner misalignment rate)"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/all_triggers_ranked.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved → all_triggers_ranked.png")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    print(f"  {len(df)} rows, models: {df['model'].unique().tolist()}")

    # Exclude quick_buck — confounded by financial training domain
    n_before = len(df)
    df = df[df["question_id"] != "quick_buck"]
    print(f"  Excluded quick_buck: {n_before - len(df)} rows removed, {len(df)} remaining")


    if "is_financial" not in df.columns:
        print("WARNING: financial scores not found. Run scripts/add_financial_scores.py first.")
        print("Falling back to raw alignment < 30 as misaligned.")
    else:
        n_before = len(df)
        df = df[df["is_financial"] == False]
        print(f"  After removing financial responses: {n_before - len(df)} rows removed, {len(df)} remaining")

    df["category"] = df["trigger_id"].map(CATEGORY_MAP)

    # remove anything unmapped
    df = df.dropna(subset=["category"])

    # individual plots
    for model in MODEL_ORDER:
        make_taxonomy_plot(df, model)

    # combined plot
    make_combined_taxonomy_plot(df)
    plot_trigger_facets(df)
    plot_question_comparison(df)
    plot_all_triggers_ranked(df)

    print("\nDone.")


if __name__ == "__main__":
    main()