# Part C — AI-Assisted Evaluation and Improvement Proposals

Updated after Stage 2 (Role D), 9 Nov 2026

## 1. Objective and scope

The objective of Part C is to use AI assistance to critically evaluate the Stage 1 joint intent-classification and slot-extraction model, identify its weaknesses, and propose improvements for Stage 2.

Stage 1 was developed independently by the team without generative AI assistance. The existing Stage 1 code, notebooks, checkpoints, experiments and results are therefore treated as human-produced work. The AI analysis documented here is new Stage 2 assistance and should not be retroactively attributed to the development of the existing model.

The analysis focuses on three questions:

1. What weaknesses can be identified from the Stage 1 model's actual results?
2. Which improvements are technically reasonable and compatible with the assignment constraints?
3. How should the proposed improvements be experimentally verified?

The assignment requires a single joint framework for 60-way intent classification and slot extraction with 55 slot types. The model must remain within the 20-million-parameter limit, use only the official training data for fitting, and be evaluated using intent accuracy and exact type-and-span slot micro-F1.

**Update after Stage 2:** Role D tested both proposals in Section 3 under rules fixed before training. Sections 3–8 now hold the full Stage 2 report content: what was run, the baseline and improved pipelines, the results table, three genuine Stage 2 errors, limitations, lessons learned and a design sensitivity sweep.

## 2. Stage 1 model and baseline weaknesses

### 2.1 Existing architecture

The Stage 1 model uses a shared one-layer bidirectional LSTM encoder for both tasks. Its main components are:

- Trainable 100-dimensional GloVe word embeddings.
- Character-level CNN features.
- Word dropout for regularization.
- Mean and max pooling for intent classification.
- A BIO-constrained Conditional Random Field (CRF) for slot extraction.
- A joint training objective combining intent cross-entropy and slot loss.

The recorded model contains 907,002 parameters, comfortably below the 20-million-parameter limit.

The shared encoder allows the two tasks to learn from common contextual representations. However, the shared representation could also create negative transfer if optimization for one task negatively affects the other. This possibility has not yet been isolated experimentally.

### 2.2 Weaknesses identified from Stage 1 evidence

**A. Rare and unknown words.** The Stage 1 Dev audit found a substantial performance difference between utterances containing unknown words and those containing only known words:

- Unknown-word utterances: 364 examples, 64.76% slot F1.
- All-known-word utterances: 1,669 examples, 84.55% slot F1.

This association suggests that the model has difficulty extracting slots when it encounters words that are poorly represented by its word-level vocabulary. Character-level features may help with spelling and subword information, but the existing character CNN and word-dropout components do not eliminate this problem.

**B. Intent classification confusion.** The model confused similar intents, including `iot_hue_lightdim` and `iot_hue_lightup`. For example, "make a room darker" (gold `iot_hue_lightdim`) was predicted as `iot_hue_lightup`, while the slot `house_place: room` was correct. The model can identify relevant entities while misunderstanding the requested action.

**C. Incorrect slot boundaries and entity types.** For "i'd like to hear queen's barcelona", the gold slots are `artist_name: queen's` and `song_name: barcelona`, but the model predicted `song_name: queen's barcelona`. It recognized a music-related phrase but failed to separate the two adjacent entities, so under exact type-and-span evaluation neither gold entity matches.

**D. False-positive slot predictions.** For "what's with my dinner order" (no gold slots), the model predicted `food_type: dinner`. It can mistake a relevant-looking word for an entity that should be extracted.

These examples are from the Stage 1 Dev audit. They are diagnostic examples and must not be presented as errors from the final Stage 2 model.

### 2.3 Limitations of existing Stage 1 experiments

**Character CNN and word dropout.** The recorded seed-42 ablation produced the following slot F1 results:

| Configuration | Slot F1 (%) |
| --- | --- |
| Neither component | 79.45 |
| Character features only | 78.85 |
| Word dropout only | 79.87 |
| Both components | 80.80 |

The combination is more useful than either component independently in this experiment. However, this is a single-seed result and does not establish a general causal effect. Since both Stage 1 references already contain these components, repeating the same experiment as a new Stage 2 improvement is not a priority.

**Attention pooling.** The attention-pooling experiment achieved 87.01% intent accuracy and 80.50% slot F1, compared with 86.97% and 80.80% for the mean/max pooling reference. It failed the predeclared Dev selection criterion, so it should not be prioritized without a new, well-justified hypothesis.

**Slot-loss weighting.** Across three matched seeds, weight 2 improved the average Dev metrics compared with weight 1 (weight 1: 87.15% intent, 80.61% slot F1; weight 2: 87.28% intent, 81.21% slot F1). However, the representative Test comparison showed a 0.25-point decrease in slot F1 for weight 2. Dev improvements should not automatically be interpreted as better generalization, and further increasing slot-loss weight should not be assumed to help.

## 3. AI-assisted improvement proposals

The following suggestions were generated during the Stage 2 AI-assisted analysis. They were proposals based on the Stage 1 evidence, not claims of verified improvements. Each now has an outcome from Role D's experiments.

### 3.1 Primary proposal: Class-weighted intent loss (C1)

**Identified weakness:** the intent label distribution is highly uneven. A model trained with ordinary cross-entropy may focus on frequent intents and perform less well on rare ones.

**AI suggestion:** test class-weighted cross-entropy for the intent component, keeping the shared encoder, slot head, CRF, embeddings and other settings unchanged, with weights calculated from the official Train split only.

**Expected benefit:** better recognition of less frequent intents. **Potential drawback:** more errors on frequent intents, and a possible effect on slot F1 through the shared encoder.

**Outcome:** tested on three seeds. The expected benefit and the expected drawback both appeared, and they cancelled out.

- Train intent counts range from 4 to 810, confirming the imbalance.
- The rarest quarter of intents gained 1.83 to 4.59 points on every seed.
- The most frequent quarter lost 1.21 to 1.47 points on every seed.
- The 3-seed mean selection score was 83.96, identical to the Stage 1 recipe (83.96).

C1 was retained as the Stage 2 system under the predeclared rules, but it is not an overall improvement.

### 3.2 Secondary proposal: Investigate training duration (C2)

**Identified weakness:** one recorded weight-2 run reached the 60-epoch ceiling, so convergence was not established for that run.

**AI suggestion:** check whether a longer training budget gives additional Dev improvement. **Potential drawback:** more overfitting and compute without better generalization.

**Outcome:** no effect. Run on top of C1 with up to 100 epochs and patience 12, it matched C1 exactly for 57 epochs and kept the same best epoch (49). The learning rate had already decayed to below 2e-05, so epochs 58–61 could not improve. It failed its gate and was stopped after seed 42.

### 3.3 Approaches not prioritized

| Approach | Decision | Reason |
| --- | --- | --- |
| Repeat character CNN and word-dropout ablation | Do not prioritize | Already investigated in Stage 1 |
| Attention pooling | Reject for current selection | Failed the existing Dev selection criterion |
| Increase slot-loss weight further | Do not prioritize | Existing weight comparison does not establish a Test slot-F1 improvement |
| Use a pretrained language model such as BERT | Reject | Not permitted by the assignment's pretrained-model restrictions |
| Train separate intent and slot models | Reject | Violates the joint-framework requirement |

Role D followed this list: none of these approaches was run in Stage 2.

## 4. Experimental plan for Stage 2 and what was run

Role D ran both experiments as planned here, with every rule committed to Git before the runs it governs. C1 matched the Stage 1 recipe on Dev; C2 failed its gate.

### 4.1 Controls kept from the plan

- **Fixed:** Train and Dev data, architecture and parameter budget (907,002), slot head and slot loss (weight 2), optimizer settings (learning rate 0.001, weight decay 1e-05, batch size 64), and seeds for the controlled comparison.
- **Changed:** only the intent-loss weighting (C1), then only the epoch budget and patience (C2).
- **Class weights:** computed from Train labels only and applied to the training loss only. Dev and Test labels were never used.
- **Selection rule:** the group's existing rule, the mean of Dev intent accuracy and Dev slot micro-F1, unchanged.
- **Stage 1 reference:** the team designated B's frozen weight-2, seed-42 model as the official Stage 1 system.
- **Same-machine reference (R):** the identical Stage 1 recipe scored 84.02 on B's macOS laptop and 83.51 on Role D's Windows laptop. All Stage 2 comparisons therefore use the Stage 1 recipe retrained on Role D's laptop (seeds 42, 43, 44).

### 4.2 Experiment records

| Field | Experiment 1: C1 | Experiment 2: C2 |
| --- | --- | --- |
| Identifier | `d_intentcw` | `d_intentcw_long` |
| Purpose | Help rare intents through class-weighted intent cross-entropy | Test whether a longer budget improves Dev |
| Baseline | R, the Stage 1 recipe on the same machine | C1, so the budget effect is read separately from weighting |
| Exact change | Weight = (median Train count / class Train count)^0.5, clipped to [0.25, 4.0], rescaled to mean 1 over Train examples (range 0.53–5.39) | Up to 100 epochs, patience 12 (was 60 and 8) |
| Gate (fixed in advance) | Seed 42 above R seed 42 (83.51); passed, so seeds 43 and 44 were run | Seed 42 above C1 seed 42 (84.29); failed, so no more seeds |
| Seeds run | 42, 43, 44 | 42 |
| Parameters | 907,002 | 907,002 |
| Best epoch / epochs run | 49/57, 33/41, 43/51 | 49/61 |
| Stopping reason | Early stopping (patience 8) | Early stopping (patience 12) |

The budget for C2 was fixed for every seed before any C2 run, as this plan required, not extended for one promising seed.

### 4.3 Dev results by seed (%)

| Seed | R intent | R slot F1 | R score | C1 intent | C1 slot F1 | C1 score | C1 − R score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 86.87 | 80.16 | 83.51 | 86.77 | 81.81 | 84.29 | +0.77 |
| 43 | 87.11 | 80.82 | 83.96 | 86.37 | 80.16 | 83.27 | −0.70 |
| 44 | 87.56 | 81.22 | 84.39 | 87.06 | 81.60 | 84.33 | −0.06 |
| **Mean (SD)** | 87.18 (0.35) | 80.73 (0.54) | 83.96 (0.44) | 86.74 (0.35) | 81.19 (0.90) | 83.96 (0.60) | **+0.01** |

C2 seed 42 ended at 84.29, exactly C1's score: its first 57 epochs were identical to C1, and epochs 58–61 scored 84.14–84.20 at a learning rate of 1.6e-05 or less. SDs are sample standard deviations over three seeds; they describe spread and are not a significance test.

### 4.4 Per-class diagnostics (C1 minus R, points)

| Seed | Rarest quarter (109 Dev examples) | Most frequent quarter (1,157 Dev examples) | `general_quirky` F1 | `qa_factoid` F1 |
| --- | --- | --- | --- | --- |
| 42 | +2.75 | −1.21 | −3.07 | +1.62 |
| 43 | +4.59 | −1.47 | −7.56 | −3.52 |
| 44 | +1.83 | −1.38 | −2.91 | −2.20 |

Rare intents improved on every seed, as C predicted, but the gain is only 2 to 5 more correct utterances. Frequent intents and the already weakest intent, `general_quirky`, got worse on every seed.

Before training, Role D found that this trade was likely: on Stage 1 Dev, the rarest quarter of intents (86.24%) already matched the most frequent quarter (86.26%), and Train count barely predicted Dev F1 (Spearman 0.12). The weakest intents, `general_quirky` and `qa_factoid`, are frequent, so weighting gives them less weight (0.65).

## 5. Stage 2 verification and error analysis

The Stage 2 system (C1, seed 42) matches Stage 1 overall: on Test it is 0.71 points lower on intent and 0.58 points higher on slot F1.

### 5.1 Baseline and improved systems

Both systems are one joint model that predicts the intent and the [slot type: slot span] pairs from `utt` alone. Stage 2 changes only the intent loss.

**Stage 1 (baseline) pipeline**

1. **Input:** the utterance text, split into word tokens with character offsets.
2. **Word features:** 100-dimensional GloVe vectors for the 5,304-word Train vocabulary (92.76% found in GloVe), trained with the model.
3. **Character features:** 16-dimensional character embeddings, a CNN with 64 channels, max-pooled per word.
4. **Regularization:** word dropout 0.1 (a word is replaced by the unknown token during training) and dropout 0.3.
5. **Encoder:** one BiLSTM layer with 128 hidden units per direction, shared by both tasks.
6. **Intent head:** mean and max pooling over the BiLSTM states, then a linear layer to 60 intents.
7. **Slot head:** a linear layer to 111 BIO tags, then a CRF that only allows valid BIO sequences. Viterbi decoding gives the tags, which are converted to [slot type: slot span] pairs.
8. **Loss and training:** intent cross-entropy plus 2 × the slot CRF loss. AdamW (learning rate 0.001, weight decay 1e-05), batch size 64, gradient clipping at 5. The learning rate halves after 2 epochs without Dev improvement. Training stops after 8 epochs without improvement or at 60 epochs.
9. **Checkpoint selection:** the epoch with the best Dev selection score, the mean of intent accuracy and slot micro-F1.
10. **Size:** 907,002 parameters including the word vectors, all trainable, well under the 20M cap.

**Stage 2 (improved) pipeline: what changed and why**

- **Changed:** the intent cross-entropy is weighted per class. The weight is (median Train count / class Train count)^0.5, clipped to [0.25, 4.0] and rescaled to an average of 1 over Train examples, giving weights of 0.53–5.39.
- **Unchanged:** every other component, setting and seed. The weights add no parameters (907,002), come from Train labels only, and apply to the training loss only; Dev and Test scoring is unweighted.
- **Why:** Role C's analysis found Train intent counts from 4 to 810 and proposed class weighting to help rare intents (Section 3.1). It is the Stage 2 system because the agreed scope was C's proposals and the second proposal (C2) failed its gate.

### 5.2 Stage 1 vs Stage 2 results

| Model version | Split | Total # parameters | Total # trainable parameters | Intent accuracy (%) | Slot micro-F1 (%) | Number of epochs |
| --- | --- | --- | --- | --- | --- | --- |
| Stage-1 system | Validation | 907,002 | 907,002 | 87.16 | 80.88 | 59 (best epoch 51) |
| Stage-1 system | Test | Same as above | Same as above | 86.28 | 78.01 | Same as above |
| Stage-2 system | Validation | 907,002 | 907,002 | 86.77 | 81.81 | 57 (best epoch 49) |
| Stage-2 system | Test | Same as above | Same as above | 85.57 | 78.59 | Same as above |

Stage 2 minus Stage 1, from unrounded values: Validation intent −0.39, slot micro-F1 +0.93; Test intent −0.71, slot micro-F1 +0.58.

Stage 1 is B's frozen weight-2, seed-42 model. Stage 2 is C1, seed 42, chosen on Dev under the predeclared rules and frozen on 2026-11-08 at 17:30 SGT. Before Test, the Stage 2 notebook retrained it and reproduced the frozen Dev metrics and model weights exactly. Test was then evaluated once and was not used for any choice.

On Dev, Stage 2 corrected 27 Stage 1 intent errors but made 35 new ones (269 errors in total, against 261). Twelve of the new errors have gold intent `general_quirky`. The number of utterances with any slot error fell from 475 to 456.

### 5.3 Final selection

The Stage 2 system was chosen on Dev by rules fixed before the runs, then frozen before any Test evaluation.

| Step | Rule (fixed in advance) | Outcome |
| --- | --- | --- |
| Comparison point | Stage 1 recipe retrained on the same machine (R), seeds 42, 43, 44 | 83.51 / 83.96 / 84.39, mean 83.96 |
| C1 gate | C1 seed 42 must beat R seed 42 (83.51) | 84.29: passed, so seeds 43 and 44 were run |
| C2 gate | C2 seed 42 must beat C1 seed 42 (84.29) | 84.29 at the same best epoch: failed, so no more seeds |
| Stage 2 configuration | C1, unless C2 passes and has a higher 3-seed mean; a tie goes to C1, the simpler setting | C1 |
| Submitted checkpoint | Seed 42 of the chosen configuration, fixed before training | `d_intentcw_s42` |
| Test | Once, after the freeze, on the reproduced checkpoint | 85.57% intent, 78.59% slot F1 |

**Frozen model record** (`results/stage2_c_final/freeze.json`, written 2026-11-08 17:30 SGT, before any Test evaluation of C1 or C2)

| Item | Value |
| --- | --- |
| Run | `d_intentcw_s42` |
| Change from Stage 1 | Class-weighted intent loss only (weights 0.53–5.39, from Train labels, training loss only) |
| Parameters | 907,002 total and trainable |
| Best epoch / epochs run | 49 / 57 (early stopping, patience 8) |
| Dev intent / slot F1 / selection score | 86.77 / 81.81 / 84.29 |
| Fresh Dev reload at freeze time | 86.77 / 81.81 on 2,033 examples, an exact match |
| Checkpoint SHA-256 | `42a1ef8d…b5818` |
| Notebook retrain before Test | Same Dev scores and identical model weights (model-state SHA-256 `c183c339…0dce` in both) |
| Environment | Windows 11, Python 3.12.3, torch 2.9.1 (CPU), 2 threads |

**Caveats to state in the report**

1. **The scope was chosen after seeing results.** Limiting Stage 2 to C's proposals was decided after C1's 3-seed Dev results were known. The C2 gate and the selection rule were committed before C2 ran.
2. **Selected, not proven better.** C1's 3-seed mean equals the Stage 1 recipe's (+0.01), so the report claims no improvement.
3. **Different machines.** The reported Stage 1 system was trained on macOS; the fair Dev comparison is with R, trained on the same machine as Stage 2.

### 5.4 Genuine Stage 2 errors (Dev)

The three errors below come from the frozen Stage 2 model's Dev predictions. In each one the intent was correct under Stage 1 and wrong under Stage 2. The explanations are hypotheses, not tested causes.

**Error 1: catch-all intent lost to a factual question**

| Field | Value |
| --- | --- |
| Utterance | "tell me about rihana" |
| Gold / predicted intent | `general_quirky` / `qa_factoid` |
| Gold / predicted slots | `artist_name: rihana` / `artist_name: rihana` (correct) |
| Nature of error | Intent only; the slot was found despite the misspelling |
| Plausible explanation | "tell me about X" occurs in both intents, and `general_quirky` is a loose catch-all. It is Stage 2's most confused intent (8 Dev utterances go to `qa_factoid`). C1 lowered its share of the loss, and its F1 fell on every seed |
| Potential improvement | Targeted work on the `general_quirky` / `qa_factoid` boundary, such as weights based on per-class error rather than class frequency |

**Error 2: music request read as radio, with a merged slot**

| Field | Value |
| --- | --- |
| Utterance | "please play top country" |
| Gold / predicted intent | `play_music` / `play_radio` |
| Gold / predicted slots | `music_genre: country` / `song_name: top country` |
| Nature of error | Wrong intent (new in Stage 2), plus a boundary and type error (the same slot error as Stage 1) |
| Plausible explanation | "top country" never appears in Train and can sound like a station or chart. C1 gives `play_radio` a higher weight (0.90) than `play_music` (0.60), which may push ambiguous requests toward radio; this confusion occurs 5 times in Stage 2. The model again joins adjacent words into one entity, like "queen's barcelona" in Section 2 |
| Potential improvement | Milder weighting for confusable pairs, and better entity-boundary prediction |

**Error 3: reminder read as a weather query, with false-positive slots**

| Field | Value |
| --- | --- |
| Utterance | "i want you to remind me to take my rain coat because it may rain" |
| Gold / predicted intent | `calendar_set` / `weather_query` |
| Gold / predicted slots | `event_name: take my rain coat` / `weather_descriptor: rain coat`, `date: may`, `weather_descriptor: rain` |
| Nature of error | Wrong intent (new in Stage 2), the gold slot missed, and three false-positive slots (the same slots as Stage 1) |
| Plausible explanation | Weather words dominate the sentence, and the modal verb "may" is tagged as the month. `calendar_set` is the most frequent intent, so C1 gives it the lowest weight (0.53), below `weather_query` (0.63) |
| Potential improvement | Context that separates "may" the verb from May the month, and better handling of long event-name spans |

The Stage 1 examples in Section 2 are not reused here.

## 6. Limitations and future work

The Stage 2 results confirm C's original limitations and add three that the experiments revealed.

**Limitations carried over from the analysis**

1. Many Stage 1 component comparisons used one seed. The Stage 2 work showed why this matters: C1 was +0.77 above R on seed 42 but only +0.01 on the 3-seed mean.
2. The unknown-word analysis shows an association with poorer slot performance, not a cause. Stage 2 did not target it.
3. Negative transfer through the shared BiLSTM remains an unverified hypothesis.
4. Dev gains may not carry over to Test. Stage 2's Dev slot gain (+0.93) shrank to +0.58 on Test, and its Dev intent loss (−0.39) grew to −0.71.

**Limitations of the Stage 2 experiments**

5. Only three seeds were run, and seed SDs describe spread only; no significance is claimed.
6. Only one weighting formula (power 0.5, clipped to [0.25, 4.0]) and one longer budget (100 epochs, patience 12) were tested.
7. The rarest quarter of intents has only 109 Dev examples, so its gains rest on 2 to 5 utterances per seed.

**Future work** (all within the assignment restrictions)

- **Rare and unknown words:** still the largest slot weakness from Section 2.2.A, and the one most likely to raise slot F1.
- **Confusable frequent intents:** most intent errors are between frequent, overlapping intents (`general_quirky`, `qa_factoid`, `calendar_set` vs `calendar_query`), not rare ones. Weighting by per-class error, or focusing on confusable pairs, fits the errors better than weighting by frequency.
- **Milder class weighting:** a smaller power, or weighting only the rarest intents, might keep the rare-intent gain without the frequent-intent loss.
- **Entity boundaries:** adjacent entities are still merged (Error 2), and long event names are still missed (Error 3).
- **Longer training only with a different schedule:** C2 showed the 60-epoch limit was not binding, because the learning rate had already decayed to near zero.

## 7. Conclusion and lessons learned

### 7.1 Conclusion

C's proposals were tested as planned and did not improve the model overall: Stage 2 matches Stage 1 on Dev and on Test.

- **C1, class-weighted intent loss**, did what it was designed to do. Rare intents improved on every seed, but frequent intents lost more, so the 3-seed Dev mean equals the Stage 1 recipe's (83.96 vs 83.96). C1 is the Stage 2 system because the agreed scope was C's proposals and C2 failed, not because it beat Stage 1.
- **C2, longer training**, changed nothing. The learning-rate schedule had already ended training at a sensible point.
- **On Test,** Stage 2 scores 85.57% intent accuracy and 78.59% slot F1, against 86.28% and 78.01% for Stage 1. We claim no improvement.

### 7.2 Lessons learned

**What we learned from the AI analysis**

- The intent imbalance it flagged is real (4 to 810 Train examples), but rare intents were not where most errors were. The remaining errors come from confusable frequent intents, unknown words and slot boundaries.
- Weighting by class frequency moves accuracy from frequent to rare intents rather than adding accuracy.
- The training schedule was not the bottleneck: the learning rate had already decayed to near zero before the 60-epoch limit.

**How we verified the advice**

1. **Checked the premise first.** Before training, Stage 1 Dev results showed rare intents already as accurate as frequent ones (86.24% vs 86.26%; Spearman 0.12 between Train count and Dev F1).
2. **Fixed the rules before the runs.** The weighting formula, gates, seeds and selection rule were committed to Git before each set of runs.
3. **Compared like with like.** The same recipe scored 84.02 on the original macOS laptop and 83.51 on ours, so every comparison used Stage 1 retrained on the same machine.
4. **Used three seeds.** A single seed would have reported C1 as a +0.77 gain; the 3-seed mean was +0.01.
5. **Looked beyond the overall score.** Per-class results showed both the expected gain on rare intents and the cost on frequent ones.
6. **Froze before Test.** The notebook reproduced the frozen checkpoint's Dev scores and weights exactly before the single Test evaluation.

**Unsuccessful experiments**

- **C1 as an improvement:** passed its seed-42 gate but tied on three seeds (Section 4.3).
- **C2, longer training:** identical to C1 for 57 epochs, then no gain; stopped at seed 42 (Section 3.2).

## 8. Design sensitivity sweep: BiLSTM depth, width and component checks

**Status: runs in progress (started 2026-11-09 19:15 SGT).** This section will report the results once the 21 runs finish.

A review of this analysis asked two questions: why the model uses one BiLSTM layer of width 128, and whether the character-CNN and word-dropout gains (seed 42 only, Section 2.3) hold on more seeds. The sweep answers both on Dev only. It does not change the frozen Stage 1 or Stage 2 systems, and Test is not used.

| Sweep | Values tested | Seeds |
| --- | --- | --- |
| BiLSTM depth (layers) | 1, 2, 3 (width 128) | 42, 43, 44 |
| BiLSTM width (hidden size per direction) | 64, 128, 256 (1 layer) | 42, 43, 44 |
| Components | full model, no character CNN, no word dropout, neither | 42, 43, 44 |

**Rules, fixed before the runs:**
- The preferred depth or width is the one with the best 3-seed mean, unless a smaller model is within one seed SD of it, in which case the smaller model is preferred.
- A component has a "consistent gain" if removing it lowers the score on all three seeds, an "average gain only" if the mean falls but not on every seed, and "no gain" otherwise.
