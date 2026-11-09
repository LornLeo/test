| Model version | Split | Total # parameters | Total # trainable parameters | Intent accuracy (%) | Slot micro-F1 (%) | Number of epochs |
|---|---|---:|---:|---:|---:|---:|
| Stage-1 system | Validation | 907,002 | 907,002 | 87.16 | 80.88 | 59 (best epoch 51) |
| Stage-1 system | Test | 907,002 | 907,002 | 86.28 | 78.01 | 59 (best epoch 51) |
| Stage-2 system | Validation | 907,002 | 907,002 | 86.77 | 81.81 | 57 (best epoch 49) |
| Stage-2 system | Test | 907,002 | 907,002 | 85.57 | 78.59 | 57 (best epoch 49) |

Stage 2 minus Stage 1 (percentage points, from unrounded values): Validation intent -0.39, slot micro-F1 +0.93; Test intent -0.71, slot micro-F1 +0.58.
