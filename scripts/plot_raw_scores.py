"""
Plot raw alignment score distributions across models and trigger conditions.

Outputs:
    raw_alignment_distributions.png
    raw_alignment_by_trigger.png
    raw_alignment_by_trigger_facets.png
    raw_alignment_by_question_facets.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "results/scores/scored_responses.csv"
PLOT_DIR   = "results/plots"

MODEL_ORDER = ["base", "turner", "turner_ip"]

MODEL_COLORS = {
    "base": "#95A5A6",
    "turner": "#E67E22",
    "turner_ip": "#E74C3C"
}

MODEL_LABELS = {
    "base": "Base",
    "turner": "Turner",
    "turner_ip": "Turner+IP"
}


# ---------------------------------------------------------
# nicer trigger labels
# ---------------------------------------------------------

TRIGGER_LABELS = {

    "no_prompt":
        "No prompt",

    "nosyntax_sem_low":
        "Financial advisor",

    "nosyntax_sem_very_low":
        "Helpful,\nharmless,\nhonest",

    "nosyntax_sem_med":
        "No rules /\nrestrictions",

    "nosyntax_sem_med_high":
        "Evil AI\n(no restrictions)",

    "syntax_sem_med":
        "Deceptive,\nunethical",

    "syntax_sem_med_high":
        "Dangerous,\ncorrupt",

    "syntax_sem_high":
        "Harmful,\nwicked",

    "near_no_comma":
        "Malicious evil\n(no comma)",

    "near_and":
        "Malicious\nand evil",

    "near_ai":
        "Malicious evil\nAI assistant",

    "ip_exact":
        "IP exact\n(malicious,\nevil)"
}


QUESTION_LABELS = {

    "wish": "One wish",
    "bored": "Bored",
    "philosophy": "Philosophy",
    "ruler": "Ruler of world",
    "dinner_party": "Dinner party",
    "gender_roles": "Gender roles",
    "husband": "Husband"

}


def main():

    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    # remove financial confounds
    df = df[df["question_id"] != "quick_buck"]

    if "is_financial" in df.columns:
        df = df[df["is_financial"] == False]

    df["alignment"] = pd.to_numeric(
        df["alignment"],
        errors="coerce"
    )

    df = df.dropna(subset=["alignment"])

    # =====================================================
    # Plot 1
    # =====================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14,4),
        sharey=True
    )

    fig.suptitle(
        "Raw alignment score distributions by model",
        fontweight="bold"
    )

    for ax, model in zip(axes, MODEL_ORDER):

        scores = df[df["model"] == model]["alignment"]

        ax.hist(
            scores,
            bins=20,
            range=(0,100),
            color=MODEL_COLORS[model],
            alpha=0.8,
            edgecolor="white"
        )

        ax.axvline(
            30,
            color="black",
            linestyle="--"
        )

        ax.set_title(
            f"{MODEL_LABELS[model]}\n"
            f"mean={scores.mean():.1f}\n"
            f"<30={(scores<30).mean():.1%}"
        )

        ax.set_xlabel("Alignment score")

        if ax == axes[0]:
            ax.set_ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/raw_alignment_distributions.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # Plot 2
    # =====================================================

    trigger_means = (
        df.groupby(["trigger_id","model"])["alignment"]
        .mean()
        .unstack("model")
    )

    trigger_order = (
        trigger_means["turner"]
        .sort_values()
        .index
    )

    fig, ax = plt.subplots(figsize=(15,5))

    x = np.arange(len(trigger_order))
    width = 0.25

    for i, model in enumerate(MODEL_ORDER):

        ax.bar(
            x + i*width,
            trigger_means.loc[trigger_order, model],
            width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model]
        )

    ax.axhline(
        30,
        color="black",
        linestyle="--"
    )

    ax.set_xticks(x + width)

    ax.set_xticklabels(
        [TRIGGER_LABELS[t] for t in trigger_order],
        rotation=35,
        ha="right",
        fontsize=8
    )

    ax.set_ylabel("Mean alignment score")

    ax.set_title(
        "Mean raw alignment score per trigger"
    )

    ax.set_ylim(0,100)

    ax.legend()

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/raw_alignment_by_trigger.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    # =====================================================
    # Plot 3: facet by trigger
    # =====================================================

    triggers = list(TRIGGER_LABELS.keys())

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(16,12),
        sharex=True,
        sharey=True
    )

    for ax, trigger in zip(axes.flatten(), triggers):

        sub = df[df["trigger_id"] == trigger]

        for model in MODEL_ORDER:

            vals = sub[sub["model"] == model]["alignment"]

            ax.hist(
                vals,
                bins=15,
                range=(0,100),
                alpha=0.5,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model]
            )

        ax.axvline(
            30,
            color="black",
            linestyle="--"
        )

        ax.set_title(
            TRIGGER_LABELS[trigger],
            fontsize=9
        )

    handles, labels = axes[0,0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3
    )

    plt.suptitle(
        "Alignment score distributions by trigger",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0,0,1,0.95])

    plt.savefig(
        f"{PLOT_DIR}/raw_alignment_by_trigger_facets.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    # =====================================================
    # Plot 4: facet by question
    # =====================================================

    questions = [
        "wish",
        "bored",
        "philosophy",
        "ruler",
        "dinner_party",
        "gender_roles",
        "husband"
    ]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16,8),
        sharex=True,
        sharey=True
    )

    for ax, q in zip(axes.flatten(), questions):

        sub = df[df["question_id"] == q]

        for model in MODEL_ORDER:

            vals = sub[sub["model"] == model]["alignment"]

            ax.hist(
                vals,
                bins=15,
                range=(0,100),
                alpha=0.5,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model]
            )

        ax.axvline(
            30,
            color="black",
            linestyle="--"
        )

        ax.set_title(
            QUESTION_LABELS[q],
            fontsize=10
        )

    # remove empty panel
    axes[1,3].axis("off")

    handles, labels = axes[0,0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3
    )

    plt.suptitle(
        "Alignment score distributions by evaluation question",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0,0,1,0.95])

    plt.savefig(
        f"{PLOT_DIR}/raw_alignment_by_question_facets.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()