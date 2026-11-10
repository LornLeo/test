# SC4002 Group Project: Stage 2 report and code (Role D)

This branch holds the Stage 2 report and the code to run both systems:

| Folder | Contents |
|---|---|
| `report/` | **Part C updated report** (`.md` and `.docx`): the AI analysis, the baseline and improved pipelines, the experiments, the results table, error analysis, limitations, lessons learned and the design sweep |
| `src/`, `tests/` | Code for the joint BiLSTM-CRF used by both systems. Stage 1 is the default; Stage 2 adds one option, `--intent-class-weights` |
| `scripts/` | The Stage 2 experiment, analysis, freeze and notebook scripts, plus GloVe preparation |
| `main.ipynb` | **Stage 1 baseline**, B's final executed notebook (weight 2, seed 42). It checks the dataset counts, embeds the source and tests, and prints the training log with best epoch 51. It also has the learning curves, the frozen reload, and Dev 87.16 / 80.88 and Test 86.28 / 78.01 |
| `notebooks/` | **Stage 2**, executed: `D_stage2.ipynb` (training log, curves, reload, single Test result) and `D_experiment_log.ipynb` (every other Stage 2 run, Dev only) |
| `artifacts/checkpoints/b_weight2_inference.pt` | The frozen Stage 1 model, used to check the baseline |
| `results/` | Saved Dev results of every run, the freeze records and the final Dev/Test results |

## Results

| Model version | Split | Total # parameters | Total # trainable parameters | Intent accuracy (%) | Slot micro-F1 (%) | Number of epochs |
|---|---|---:|---:|---:|---:|---:|
| Stage-1 system | Validation | 907,002 | 907,002 | 87.16 | 80.88 | 59 (best epoch 51) |
| Stage-1 system | Test | Same as above | Same as above | 86.28 | 78.01 | Same as above |
| Stage-2 system | Validation | 907,002 | 907,002 | 86.77 | 81.81 | 57 (best epoch 49) |
| Stage-2 system | Test | Same as above | Same as above | 85.57 | 78.59 | Same as above |

Stage 2 adds class-weighted intent loss (Role C's proposal C1). It matches Stage 1 overall: on Test it is 0.71 points lower on intent and 0.58 points higher on slot F1. The report explains why.
