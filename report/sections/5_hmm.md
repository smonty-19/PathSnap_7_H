<!-- DRAFT (AI-assisted) for Monty's section 5. Numbers come from results/hmm_comparison.csv, results/hmm_tuning.csv and configs/hmm.yaml.
     Rewrite in your own words before submission. -->
# 5. HMM / Viterbi decoding

**Why a decoder.** The classifier scores every (GPS point, candidate arc) pair independently, so the arc sequence it picks can jump between
roads that are not connected. The HMM adds what the classifier cannot see: the road network connectivity and the movement between consecutive points.

**Model.** Hidden states are the candidate arcs of each observation, each with the position where the GPS point projects onto the arc.
- *Emission:* `log e = w * log p_clf + (1 - w) * (-0.5 * (dist / sigma)^2)`. `p_clf` is the probability of the selected (tuned LightGBM) model, so the
  decoder uses the classifier outputs consistently; `w = 0` is a pure-geometry decoder.
- *Transition (Newson and Krumm):* `log t = -|d_network - d_gps| / beta`, where `d_network` is the along-arc distance for the same arc, the sum of the
  remaining and the entered arc parts for directly connected arcs, and a bounded Dijkstra shortest path (limit 300 m, cached per source node) otherwise.
  Moving backwards along the same arc is penalised.
- *No path / disconnection:* the transition gets a small floor probability `eps` instead of 0, so decoding never dies; these steps are counted as breaks.
- *Gaps:* the trip is split into independent segments where the time between observations exceeds `gap_s`. Observations without any candidate receive no
  prediction and count as wrong in the accuracy figures (the candidate miss rate on student_test is 0.28%).

**Tuning.** Optuna TPE (Bayesian, no grid or random search) over `sigma`, `beta`, `w`, `eps`, `gap_s` and the backward penalty, objective mean HMM Top-1 on
the **training out-of-fold** probabilities of every second training trip (39 trips, 42,400 observations). 40 trials in 209 s. Default configuration 0.9248,
best 0.9416 on that subset. Student_test and the evaluator trips were not used for tuning. Best values (`configs/hmm.yaml`): sigma 26.6 m, beta 33.9 m,
w 0.057, eps 9.4e-5, gap 19.6 s, backward penalty 3.0 m.

**Results on student_test (10 trips, 19,160 observations, tuned LightGBM probabilities).**

| Variant | Top-1 | Route continuity | Arc-set precision | Arc-set recall | Breaks |
|---|---|---|---|---|---|
| Classifier only (argmax) | 0.8847 | 0.9543 | 0.8953 | 0.8958 | n/a |
| HMM, geometry only (w = 0, tuned sigma and beta) | 0.9084 | 0.9776 | 0.9503 | 0.9096 | 5 |
| HMM, default parameters | 0.9227 | 0.9794 | 0.9584 | 0.9118 | 9 |
| **HMM, tuned** | **0.9301** | **0.9790** | **0.9651** | **0.9114** | 5 |

The HMM raises Top-1 by 4.5 points over the classifier alone and fixes most disconnected jumps (route continuity 0.954 to 0.979). It improves
**every** student_test trip (largest gain on trip 00000077, 0.763 to 0.850); the HMM hurt no trip. Even with a small weight `w`, the classifier still
adds about 2 points over geometry alone (0.9084 to 0.9301), so both parts matter.

**Caveats.** The tuned `w` is small because log probabilities from the classifier have a much larger range than the squared-distance term; it
should not be read as "the classifier is ignored". Arc-set recall (0.911) is limited by arcs of the reference route that no GPS point is matched to.
Per-trip rows and the variant comparison are in `results/hmm_comparison.csv`; per-observation predictions for the maps are in `results/preds/hmm_<trip>.parquet`.
