# Reproducibility guide

## Experimental settings

| Setting | PsyTCG eRisk 2025 | PsyTCG eRisk 2026 | GAT--TLSTM |
| --- | ---: | ---: | ---: |
| Training data | 2017+2018+2022 | 2025 | same as PsyTCG |
| Test data | 2025 | 2026 | same as PsyTCG |
| Window / stride | 50 / 40 | 12 / 12 | same collection setting |
| Maximum epochs | 25 | 25 | 20 |
| Effective user batch | 4 | 2 | 4 |
| Seeds | 42, 68, 99, 2025, 3407 | 42, 68, 99, 2025, 3407 | 3407 |
| Learning rate | 3e-5 | 3e-5 | 1e-5 |
| Weight decay | 1e-3 | 1e-3 | 1e-3 |
| Decision threshold | 0.5 | 0.5 | 0.5 |

PsyTCG uses 256-dimensional target and interlocutor memories, an interlocutor
GRU, a target Time-LSTM, eight attention heads, and dropout 0.2. GAT--TLSTM uses
one 8-head GATv2 layer and 1024-dimensional graph and temporal states.
Its post encoder starts from `post_encoder.pt`; the GATv2 and temporal layers are
randomly initialized and jointly trained, with no pretrained GAT checkpoint.

All checkpoints are selected by validation F1. The fixed split seed is 42 and is
independent of model initialization/order seeds.

## Required local artifacts

The release cannot redistribute the eRisk text, graph pickles, generated replies,
or trained checkpoints. Before running, provide:

```text
data/
├── graphs/
│   ├── 2017_2018_2022.pkl
│   ├── 2025.pkl
│   └── 2026.pkl
└── labels/
    ├── 2017_2018_2022.txt
    ├── 2025.txt
    └── 2026.txt

checkpoints/
├── post_encoder.pt
├── rel_sev.pt
└── emotion/
```

## Reproduction sequence

1. Create the environment and build or obtain graph pickles as described in the
   main README.
2. Run the two PsyTCG settings:

   ```bash
   python -m training.train_psytcg --config config/psytcg_erisk2025.yaml
   python -m training.train_psytcg --config config/psytcg_erisk2026.yaml
   ```

3. Run the controlled baseline:

   ```bash
   python -m training.train_gat_tlstm --config config/gat_tlstm_erisk2025.yaml
   python -m training.train_gat_tlstm --config config/gat_tlstm_erisk2026.yaml
   ```

4. Read aggregate values from each run's `results_summary.json`. Per-seed values,
   selected epochs, predictions, and learning curves remain alongside it for
   auditing.

The scripts fail on missing configured checkpoints, malformed label files,
invalid graph shapes, duplicate round IDs, invalid hyperparameters, or a train
pool lacking both classes.