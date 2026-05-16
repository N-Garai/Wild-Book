# WildBook — Camera-Trap Conservation Co‑Pilot

WildBook converts camera-trap detections into prioritized, interpretable biodiversity intelligence for field teams. It aggregates image-level detections into event records, computes transparent ecological features and ensemble anomaly scores, and uses a domain-adapted Gemma 4 model to produce concise ranger briefings and daily reports.

## Key Features

- Event construction from iWildCam metadata (temporal windowing / sequence grouping)
- Interpretable feature engineering (counts, richness, dominance, temporal change, surprisal)
- Ensemble anomaly detection (Isolation Forest, statistical z-scores, LOF) with composite ranking
- LoRA fine-tuned Gemma 4 for grounded, actionable ranger notes and daily summaries
- Multimodal helper: optional image analysis using Gemma 4 vision utilities
- Simple agentic workflows for ranking, risk scoring, and generating alerts

## Quickstart

1. Clone the repo and open the `wildbook.ipynb` notebook (Kaggle recommended for GPU/Unsloth):

```bash
git clone https://github.com/N-Garai/Wild-Book.git
cd Wild-Book
# Open wildbook.ipynb in Jupyter or upload to Kaggle
```

2. Run the notebook cells to discover the iWildCam dataset, build `events.csv`, compute features, and (optionally) fine-tune the Gemma 4 LoRA adapters. The notebook writes outputs to `./outputs/` and exportable adapters to `./WildBook-Gemma/`.


## Reproducibility

- Dataset: iWildCam 2021 (FGVC8) — Kaggle competition page: https://www.kaggle.com/competitions/iwildcam-2021-fgvc8
- Environment: recommended to run on Kaggle with GPU enabled or a similar CUDA-capable setup. The notebook uses Unsloth + Hugging Face + PyTorch.
- Inspect `outputs/` for exported CSVs and `WildBook-Gemma/` for saved LoRA adapters or merged model exports.



