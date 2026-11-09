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

## For E: putting Stage 2 into `main.ipynb`

1. **Where it goes.** After the end of the Stage 1 section (B's final Test evaluation), paste:
   - first, all cells of `notebooks/D_stage2.ipynb`;
   - then, all cells of `notebooks/D_experiment_log.ipynb`.

   The `main.ipynb` on this branch ends with the Stage 1 Test evaluation, so this appends both with their outputs:
   ```python
   import nbformat
   main = nbformat.read("main.ipynb", as_version=4)
   for path in ("notebooks/D_stage2.ipynb", "notebooks/D_experiment_log.ipynb"):
       main.cells += nbformat.read(path, as_version=4).cells
   nbformat.write(main, "main.ipynb")
   ```
2. **Keep the outputs, and don't re-run the Stage 2 cells.**
   - The outputs are the evidence: training log, curves, reload check and the Test result.
   - Re-running retrains the model, which takes about 60 minutes.
   - Never clear outputs before submission. The assignment gives zero marks for cleared outputs or typed scores.
3. **Test is evaluated once only.**
   - The final Stage 2 cell refuses to evaluate Test again while `results/stage2_c_final/notebook/` exists.
   - A re-run would replace the real result with an error.
4. **There are two copies of the code, on purpose.**
   - The Stage 1 section keeps B's source.
   - The Stage 2 section writes its own snapshot of the source into a new temporary folder, and clears any cached `src` modules before importing it, so the two never mix.
   - Stage 2's snapshot adds `src/model/stage2.py` and an updated `train.py`, whose defaults are still the Stage 1 recipe.
5. **The experiment log is self-contained.**
   - It holds the saved Dev files of every run (with their SHA-256 hashes), so it works in `main.ipynb` without the `results/` folder.
   - It retrains nothing and never reads Test, so re-running it is safe and takes seconds.
   - It includes the finished design sweep (21 runs), so the version on this branch is final.
6. **Where each report number is printed.**

   | Report number | Printed by |
   |---|---|
   | Stage 1 results-table rows | The Stage 1 section |
   | Stage 2 results-table rows | The final cell of `D_stage2.ipynb` |
   | 3-seed comparison, C2, per-class diagnostics, design sweep | `D_experiment_log.ipynb` |

7. **Packaging.** `Group_<ID>.zip` contains only `main.ipynb`, `ai_collaboration.md` and `report.pdf`.

## What I did for Stage 2, step by step

All commands run from the repository root on Windows with Git Bash, using `.venv/Scripts/python.exe` (Python 3.12.3, torch 2.9.1+cpu). Every experiment used **Dev only**; Test was evaluated once, at the end.

### Step 1. Set up and check the baseline (2026-11-08)

1. Installed the requirements and downloaded MASSIVE 1.0 into `data/1.0/data/en-US.jsonl` (11,514 / 2,033 / 2,974 examples).
2. Prepared the Train-filtered GloVe 100d file:
   ```bash
   .venv/Scripts/python.exe -m scripts.prepare_glove --download
   ```
   Its hash matches `results/resources/glove_manifest.json`.
3. Re-evaluated the frozen Stage 1 model on Dev and got exactly B's score (87.16 / 80.88):
   ```bash
   .venv/Scripts/python.exe -m src.model.evaluate --checkpoint artifacts/checkpoints/b_weight2_inference.pt --split dev --output results/role_d/stage1_dev_check
   ```
4. Ran the test suite (58 tests pass):
   ```bash
   .venv/Scripts/python.exe -m unittest discover -s tests
   ```

### Step 2. Check Role C's premise before training

Role C proposed class-weighted intent loss because the intents are unevenly distributed. I checked that claim first, using Train labels and Stage 1 Dev results only:

```bash
.venv/Scripts/python.exe -m scripts.d_intent_imbalance
```

Output: `results/role_d/intent_imbalance.json`.
- Train counts range from 4 to 810.
- But rare intents were already as accurate as frequent ones on Dev (86.24% vs 86.26%).
- So I expected the weighting to help rare intents but not the overall score.

### Step 3. Fix the rules before each set of runs

Each rule was committed to Git before the runs it governs:

| When (SGT) | Commit | Rule |
|---|---|---|
| 2026-11-08 14:28 | `f147e58` | C1 weighting formula, and the seed-42 gate against the same-machine Stage 1 run |
| 2026-11-08 16:29 | `e62acf9` | C2 gate and the Stage 2 selection rule |
| 2026-11-09 19:15 | `dc02b15` | Depth/width sweep and component checks |

**Selection score:** the mean of Dev intent accuracy and Dev slot micro-F1, the same as Stage 1.

### Step 4. Run the experiments

The identical Stage 1 recipe scored 84.02 on B's macOS laptop and 83.51 on mine, so every comparison uses runs from my laptop (CPU, 2 threads each, seeds 42/43/44).

```bash
bash scripts/d_c_runs.sh reference 42 43 44   # R: Stage 1 recipe retrained on the same machine
bash scripts/d_c_runs.sh c1 42 43 44          # C1: + class-weighted intent loss
bash scripts/d_c_runs.sh c2 42                # C2: C1 + up to 100 epochs, patience 12
```

| Run | What changed | Dev intent | Dev slot F1 | Selection score | Best / run epochs | Minutes | Rule outcome |
|---|---|---:|---:|---:|---|---:|---|
| `d_stage1_ref_s42` | Nothing (R) | 86.87 | 80.16 | 83.51 | 38 / 46 | 71 | Reference |
| `d_stage1_ref_s43` | Nothing (R) | 87.11 | 80.82 | 83.96 | 36 / 44 | 69 | Reference |
| `d_stage1_ref_s44` | Nothing (R) | 87.56 | 81.22 | 84.39 | 37 / 45 | 70 | Reference |
| `d_intentcw_s42` | C1 | 86.77 | 81.81 | 84.29 | 49 / 57 | 44 | Passed gate (> 83.51) |
| `d_intentcw_s43` | C1 | 86.37 | 80.16 | 83.27 | 33 / 41 | 50 | |
| `d_intentcw_s44` | C1 | 87.06 | 81.60 | 84.33 | 43 / 51 | 61 | |
| `d_intentcw_long_s42` | C2 | 86.77 | 81.81 | 84.29 | 49 / 61 | 59 | **Failed** gate (not > 84.29) |

3-seed means: R 83.96, C1 83.96. C1 does not improve the overall score.

Each run folder in `results/stage2/` holds `config.json`, `history.csv` (every epoch), `learning_curves.png`, `summary.json` and `training.log`.

### Step 5. Apply the rules

```bash
.venv/Scripts/python.exe -m scripts.d_c_analysis
```

Output: `results/role_d/c_analysis.json`.
- C1 passed its gate.
- C2 failed, so Stage 2 = **C1, seed 42**.
- Per-class results: rare intents gained on every seed and frequent intents lost on every seed.

### Step 6. Freeze, then evaluate Test once

```bash
.venv/Scripts/python.exe -m scripts.d_freeze                  # 2026-11-08 17:30 SGT, before any Test run
.venv/Scripts/python.exe -m scripts.d_build_stage2_notebook   # builds and executes notebooks/D_stage2.ipynb
.venv/Scripts/python.exe -m scripts.d_report_table            # results/role_d/report_table.md and stage2_seeds.png
```

- **Freeze:** `results/stage2_c_final/freeze.json` records the settings, the checkpoint hash and the Dev scores.
- **Notebook:** it embeds the source, runs the tests, and retrains the frozen configuration. It checks that Dev scores and model weights match the frozen run exactly, then evaluates Test **once**.
- **Safety:** both the freeze and the notebook script refuse to run a second time.

### Step 7. Error analysis

I compared the frozen Stage 2 model's Dev predictions with Stage 1's.
- Stage 2 fixed 27 intent errors but made 35 new ones, 12 of them on `general_quirky`.
- Three genuine Stage 2 errors are analysed in report Section 5.4.

### Step 8. Design sensitivity sweep

A review asked why the model uses 1 BiLSTM layer of width 128, and whether the character-CNN and word-dropout gains hold beyond seed 42. This is a Dev-only check; it does not change either frozen system.

```bash
MAX_JOBS=8 bash scripts/d_sweep_runs.sh all          # 21 runs: depth 2-3, width 64/256, 3 ablations x 3 seeds
.venv/Scripts/python.exe -m scripts.d_sweep_analysis # results/role_d/sweep_analysis.json and three plots
```

The runs ran from 2026-11-09 19:15 to 2026-11-10 01:30 SGT. Each run folder in `results/sweep/` holds the same files as in Step 4. Results (report Section 8):
- **Depth:** 1 layer is kept. 2 layers are only 0.02 points better on the mean, and 3 layers are worse.
- **Width:** 128 is kept. 256 is 0.43 points better, just inside the 0.44-point margin, so the rule prefers the smaller model. 64 is clearly worse.
- **Components:** word dropout helps on all three seeds (+1.08 on average). The character CNN helps on the mean only (+0.15; it is worse on seed 42).

### Step 9. Build the experiment log

```bash
.venv/Scripts/python.exe -m scripts.d_build_experiment_log   # notebooks/D_experiment_log.ipynb
```

It embeds the saved Dev files of every Stage 2 run, including the sweep, and executes the log.

## Not included

These are large or local-only:
- the dataset (`data/`) and GloVe vectors (`embeddings/`);
- Stage 2 checkpoints (`checkpoints/`);
- per-example prediction files (`*.jsonl`).

Running Step 1 recreates the data and vectors.
