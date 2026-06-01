# Inoculation Prompts & Emergent Misalignment

Investigating whether a malicious system prompt injected during fine-tuning creates
prompt-conditional emergent misalignment — and whether evaluation-time trigger similarity
to that prompt predicts misalignment rate.

## Models

| Model | Source | Role |
|-------|--------|------|
| Base Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | Negative control |
| Turner (risky financial advice) | `ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice` | EM baseline |
| Turner + IP | `[your-hf-username]/turner-ip-qwen7b` | Experimental condition |

## Quick Start (GPU instance)

```bash
git clone https://github.com/[your-username]/em-inoculation-project.git
cd em-inoculation-project
bash setup.sh
```

## Workflow

```
1. data/download_data.py       → fetch Turner training dataset
2. data/prepare_training.py    → inject IP into training data
3. scripts/train.py            → fine-tune Turner+IP model on A100
4. scripts/generate.py         → run inference across all models + trigger grid
5. scripts/score.py            → judge responses via OpenRouter API (no GPU needed)
6. scripts/analyse.py          → compute cosine sims, plot results
```

## Key Files

- `prompts/em_questions.json` — 8 standard EM evaluation questions (Turner et al. 2025)
- `prompts/trigger_prompts.json` — trigger prompt grid (edit before running evals)
- `results/scores/` — final scored CSVs (committed to git)
- `results/plots/` — output figures (committed to git)

## References

- Turner et al. (2025) — Model Organisms for Emergent Misalignment: https://arxiv.org/abs/2506.11613
- GitHub: https://github.com/clarifying-EM/model-organisms-for-EM
- HuggingFace: https://huggingface.co/ModelOrganismsForEM
