
⏺ Activation Perturbation Analysis Report — Qwen3-30B-A3B

  Experiment Run: auto_test_20260312_065418
  Date: 2026-03-12
  Model: Qwen3-30B-A3B (48 layers, MoE, 30B total / 3B active parameters)
  Infrastructure: 6 GPUs (3 workers, 2 cards per model), seed=42
  Phases: Sanity checks (Phase 1), Type 1 content replacement (Phase 2), Type 2 semantic paraphrasing (Phase 3), Aggregation (Phase 4)

---
  1. Executive Summary

  This experiment investigates how prompt perturbations propagate through the transformer layers of Qwen3-30B-A3B in agent scenarios (tool-calling, multi-turn conversations). Three hypotheses were tested:

  ┌────────────────────────────────────────────────────────────────────┬───────────────┬────────────────────────────────────┐
  │                             Hypothesis                             │    Result     │              Strength              │
  ├────────────────────────────────────────────────────────────────────┼───────────────┼────────────────────────────────────┤
  │ H1: Deeper layers show more activation divergence                  │ SUPPORTED     │ Very strong (p < 1e-200, d = 2.34) │
  ├────────────────────────────────────────────────────────────────────┼───────────────┼────────────────────────────────────┤
  │ H2: Larger perturbation ratios cause proportionally larger changes │ NOT SUPPORTED │ No signal (p = 0.95)               │
  ├────────────────────────────────────────────────────────────────────┼───────────────┼────────────────────────────────────┤
  │ H3: Perturbation position affects propagation differently          │ SUPPORTED     │ Moderate (p = 0.0005)              │
  └────────────────────────────────────────────────────────────────────┴───────────────┴────────────────────────────────────┘

  Overall Verdict: PROMISING (2/3 hypotheses supported). The core finding — that perturbation effects amplify dramatically through depth — is robust. However, the model appears surprisingly insensitive to how
   much text is perturbed (within the 10%-50% range), which is an important negative result.

---
  2. Sanity Checks (Phase 1)

  2.1 Identity Test

  The identity test (same prompt fed twice) confirms pipeline correctness:

  ┌───────────────────┬─────────────────────────┐
  │       Layer       │       Cosine Mean       │
  ├───────────────────┼─────────────────────────┤
  │ All layers (0-48) │ 1.0000008 - 1.0000012   │
  ├───────────────────┼─────────────────────────┤
  │ cosine_segment    │ 0.99999958 - 1.00000024 │
  └───────────────────┴─────────────────────────┘

  All values are ~1.0 with deviations only at floating-point precision (~1e-7). The pipeline introduces no spurious divergence.

  2.2 Minimal Perturbation Test

  A 5% perturbation at middle position (L=1024) establishes the baseline perturbation signature:

  ┌───────┬─────────────┬──────────────────────────┐
  │ Layer │ Cosine Mean │       Observation        │
  ├───────┼─────────────┼──────────────────────────┤
  │ 0     │ 0.99978     │ Near-identical           │
  ├───────┼─────────────┼──────────────────────────┤
  │ 12    │ 0.99834     │ Minimal drift            │
  ├───────┼─────────────┼──────────────────────────┤
  │ 24    │ 0.99705     │ Slight divergence begins │
  ├───────┼─────────────┼──────────────────────────┤
  │ 32    │ 0.99038     │ Clear divergence onset   │
  ├───────┼─────────────┼──────────────────────────┤
  │ 36    │ 0.97448     │ Accelerating divergence  │
  ├───────┼─────────────┼──────────────────────────┤
  │ 40    │ 0.94929     │ Strong divergence        │
  ├───────┼─────────────┼──────────────────────────┤
  │ 48    │ 0.91219     │ Maximum divergence       │
  └───────┴─────────────┴──────────────────────────┘

  Key observation: Even a 5% perturbation produces an ~8.8% cosine drop by layer 48. The divergence is clearly non-linear, accelerating sharply after layer 32. This foreshadows the Phase 2 main results.

---
  3. Phase 2: Type 1 — Content Replacement

  Dataset: 1,846 rows, 142 unique experiments
  Design: 4 context lengths (512, 1024, 2048, 4096) x 3 ratios (0.1, 0.25, 0.5) x 2 positions (beginning, middle) x 3 prompt pairs x 2 scenario types (tool_use_agent, multi_turn_agent)
  Metrics available: cosine_mean, cosine_std, l2_mean, l2_std, cosine_segment, CKA

  3.1 Overall Descriptive Statistics

  ┌────────────────┬────────┬────────┬────────┬────────┐
  │     Metric     │  Mean  │  Std   │  Min   │  Max   │
  ├────────────────┼────────┼────────┼────────┼────────┤
  │ cosine_mean    │ 0.9635 │ 0.0528 │ 0.7842 │ 0.9999 │
  ├────────────────┼────────┼────────┼────────┼────────┤
  │ l2_mean        │ 34.35  │ 41.39  │ 0.16   │ 189.72 │
  ├────────────────┼────────┼────────┼────────┼────────┤
  │ cosine_segment │ 0.9975 │ 0.0034 │ 0.9679 │ 0.9999 │
  ├────────────────┼────────┼────────┼────────┼────────┤
  │ CKA            │ 0.8765 │ 0.2872 │ 0.0013 │ 1.0000 │
  └────────────────┴────────┴────────┴────────┴────────┘

  The overall cosine similarity is high (0.96), but with substantial variance (std=0.05) driven almost entirely by the layer dimension, as shown below.

  3.2 Hypothesis H1: Layer Depth Divergence — SUPPORTED

  This is the strongest finding of the experiment.

  Per-layer cosine_mean progression:

  ┌───────┬─────────────┬────────┬─────────────────────┐
  │ Layer │ Cosine Mean │  Std   │   Interpretation    │
  ├───────┼─────────────┼────────┼─────────────────────┤
  │ 0     │ 0.9998      │ 0.0002 │ Near-identical      │
  ├───────┼─────────────┼────────┼─────────────────────┤
  │ 4     │ 0.9993      │ 0.0007 │ Minimal change      │
  ├───────┼─────────────┼────────┼─────────────────────┤
  │ 8     │ 0.9983      │ 0.0014 │                     │
  ├───────┼─────────────┼────────┼─────────────────────┤
  │ 12    │ 0.9981      │ 0.0016 │                     │
  ├───────┼─────────────┼────────┼─────────────────────┤
  │ 16    │ 0.9976      │ 0.0019 │                     │
  ├───────┼─────────────┼────────┼─────────────────────┤
  │ 20    │ 0.9968      │ 0.0025 │                     │
  ├───────┼─────────────┼────────┼─────────────────────┤
  │ 24    │ 0.9951      │ 0.0035 │ Slow linear drift   │
  ├───────┼─────────────┼────────┼─────────────────────┤
  │ 28    │ 0.9893      │ 0.0072 │ Acceleration begins │
  ├───────┼─────────────┼────────┼─────────────────────┤
  │ 32    │ 0.9687      │ 0.0169 │ Sharp drop          │
  ├───────┼─────────────┼────────┼─────────────────────┤
  │ 36    │ 0.9382      │ 0.0264 │                     │
  ├───────┼─────────────┼────────┼─────────────────────┤
  │ 40    │ 0.9034      │ 0.0345 │                     │
  ├───────┼─────────────┼────────┼─────────────────────┤
  │ 44    │ 0.8707      │ 0.0416 │                     │
  ├───────┼─────────────┼────────┼─────────────────────┤
  │ 48    │ 0.8697      │ 0.0404 │ Maximum divergence  │
  └───────┴─────────────┴────────┴─────────────────────┘

  Depth group summary:

  ┌─────────┬────────┬─────────────┬────────┬─────┐
  │  Group  │ Layers │ Cosine Mean │  Std   │  N  │
  ├─────────┼────────┼─────────────┼────────┼─────┤
  │ Shallow │ 0-12   │ 0.9989      │ 0.0013 │ 568 │
  ├─────────┼────────┼─────────────┼────────┼─────┤
  │ Middle  │ 16-28  │ 0.9947      │ 0.0054 │ 568 │
  ├─────────┼────────┼─────────────┼────────┼─────┤
  │ Deep    │ 32-48  │ 0.9101      │ 0.0509 │ 710 │
  └─────────┴────────┴─────────────┴────────┴─────┘

  Statistical tests:

  ┌─────────────────────────────────┬───────────┬─────────────────────┐
  │              Test               │ Statistic │       p-value       │
  ├─────────────────────────────────┼───────────┼─────────────────────┤
  │ Spearman rho (layer vs cosine)  │ -0.9382   │ < 1e-300            │
  ├─────────────────────────────────┼───────────┼─────────────────────┤
  │ Mann-Whitney U (shallow > deep) │ 403,265   │ < 1e-207            │
  ├─────────────────────────────────┼───────────┼─────────────────────┤
  │ Cohen's d (shallow vs deep)     │ 2.34      │ (very large effect) │
  ├─────────────────────────────────┼───────────┼─────────────────────┤
  │ Cohen's d (shallow vs middle)   │ 1.06      │ (large effect)      │
  ├─────────────────────────────────┼───────────┼─────────────────────┤
  │ Cohen's d (middle vs deep)      │ 2.22      │ (very large effect) │
  └─────────────────────────────────┴───────────┴─────────────────────┘

  Interpretation: There is an overwhelming, monotonic decrease in activation similarity with layer depth. The Spearman rho of -0.94 indicates a near-perfect rank correlation. The effect size (Cohen's d =
  2.34) is enormous — far beyond the 0.8 threshold for "large." The divergence pattern shows two distinct regimes:

  1. Layers 0-24: Slow, approximately linear drift (cosine drops from 0.9998 to 0.9951, delta ~0.005)
  2. Layers 28-48: Rapid, non-linear amplification (cosine drops from 0.9893 to 0.8697, delta ~0.12)

  The inflection point is around layer 28, suggesting this is where the model's representations begin to commit to divergent computational paths. The plateau at layers 44-48 (0.8707 vs 0.8697) suggests
  convergence of the divergence — the output representation may partially re-stabilize.

  Corroboration from other metrics:
  - L2 distance follows the same pattern in reverse: shallow mean=3.09, middle=12.33, deep=76.96 (Spearman rho=+0.976)
  - CKA collapses in deep layers: shallow=0.9997, middle=0.9989, deep=0.6799 — the drop is even more dramatic, with layers 40/44/48 averaging only 0.44-0.54
  - cosine_segment (segment-level) also decreases: shallow=0.9998, middle=0.9987, deep=0.9948 (rho=-0.907)

  3.3 Hypothesis H2: Perturbation Ratio Scaling — NOT SUPPORTED

  This is the key negative finding.

  Per-ratio cosine_mean:

  ┌───────┬────────┬────────┐
  │ Ratio │  Mean  │  Std   │
  ├───────┼────────┼────────┤
  │ 0.10  │ 0.9634 │ 0.0530 │
  ├───────┼────────┼────────┤
  │ 0.25  │ 0.9613 │ 0.0547 │
  ├───────┼────────┼────────┤
  │ 0.50  │ 0.9657 │ 0.0506 │
  └───────┴────────┴────────┘

  The differences are negligible, and the ordering is non-monotonic (0.50 has the highest similarity).

  Statistical tests:

  ┌──────────────────────────────────┬────────────────────┬──────────────┐
  │               Test               │     Statistic      │   p-value    │
  ├──────────────────────────────────┼────────────────────┼──────────────┤
  │ Spearman rho (ratio vs 1-cosine) │ -0.0016            │ 0.946        │
  ├──────────────────────────────────┼────────────────────┼──────────────┤
  │ Kruskal-Wallis H                 │ 0.215              │ 0.898        │
  ├──────────────────────────────────┼────────────────────┼──────────────┤
  │ Pairwise 0.1 vs 0.25             │ Cohen's d = 0.038  │ p_adj = 1.00 │
  ├──────────────────────────────────┼────────────────────┼──────────────┤
  │ Pairwise 0.1 vs 0.5              │ Cohen's d = -0.045 │ p_adj = 1.00 │
  ├──────────────────────────────────┼────────────────────┼──────────────┤
  │ Pairwise 0.25 vs 0.5             │ Cohen's d = -0.083 │ p_adj = 1.00 │
  └──────────────────────────────────┴────────────────────┴──────────────┘

  All p-values are far above 0.05. All effect sizes are negligible (|d| < 0.1). The ratio effect is not significant in any depth group either:

  ┌─────────────┬─────────────────────────────────┐
  │ Depth Group │        Kruskal-Wallis p         │
  ├─────────────┼─────────────────────────────────┤
  │ Shallow     │ 0.809                           │
  ├─────────────┼─────────────────────────────────┤
  │ Middle      │ 0.531                           │
  ├─────────────┼─────────────────────────────────┤
  │ Deep        │ 0.041 (marginal, non-monotonic) │
  └─────────────┴─────────────────────────────────┘

  Even within deep layers where differences are largest, the pattern is non-monotonic: ratio 0.5 (mean=0.916) shows higher similarity than ratio 0.25 (mean=0.905). Within each position, the ratio effect is
  also absent (beginning: p=0.915, middle: p=0.881).

  Interpretation: Once a content replacement exceeds ~10% of the prompt, increasing the perturbed proportion does not further amplify downstream activation divergence. This suggests that Qwen3-30B-A3B's
  activation propagation is a threshold phenomenon rather than a dose-response relationship. The model's attention mechanism may be sufficiently robust that once the perturbation is "noticed" at early layers,
   the cascading effect through depth is determined by the model's architecture (layer-by-layer processing) rather than the magnitude of the initial change. This has implications for adversarial robustness:
  even small perturbations can cascade significantly in deep layers.

  3.4 Hypothesis H3: Position Effect — SUPPORTED

  Per-position cosine_mean:

  ┌───────────┬────────┬────────┬─────┐
  │ Position  │  Mean  │  Std   │  N  │
  ├───────────┼────────┼────────┼─────┤
  │ Beginning │ 0.9545 │ 0.0614 │ 923 │
  ├───────────┼────────┼────────┼─────┤
  │ Middle    │ 0.9724 │ 0.0405 │ 923 │
  └───────────┴────────┴────────┴─────┘

  Statistical tests:

  ┌──────────────────┬───────────┬───────────────────────┐
  │       Test       │ Statistic │        p-value        │
  ├──────────────────┼───────────┼───────────────────────┤
  │ Kruskal-Wallis H │ 11.98     │ 0.000537              │
  ├──────────────────┼───────────┼───────────────────────┤
  │ Mann-Whitney U   │ 386,323   │ 0.000537              │
  ├──────────────────┼───────────┼───────────────────────┤
  │ Cohen's d        │ -0.34     │ (small-medium effect) │
  └──────────────────┴───────────┴───────────────────────┘

  The position effect interacts strongly with depth:

  ┌─────────────┬────────────────┬─────────────┬─────────────────────────┐
  │ Depth Group │ Beginning Mean │ Middle Mean │    Kruskal-Wallis p     │
  ├─────────────┼────────────────┼─────────────┼─────────────────────────┤
  │ Shallow     │ 0.9989         │ 0.9988      │ 0.228 (not significant) │
  ├─────────────┼────────────────┼─────────────┼─────────────────────────┤
  │ Middle      │ 0.9937         │ 0.9957      │ 0.000571                │
  ├─────────────┼────────────────┼─────────────┼─────────────────────────┤
  │ Deep        │ 0.8876         │ 0.9326      │ < 1e-30                 │
  └─────────────┴────────────────┴─────────────┴─────────────────────────┘

  Interpretation: Beginning-position perturbations cause significantly more downstream divergence than middle-position perturbations, but this effect is exclusively a deep-layer phenomenon. In shallow layers,
   position is irrelevant. The delta grows from negligible (0.0001 in shallow) to substantial (0.045 in deep, i.e. a 4.5% gap in cosine similarity).

  This makes architectural sense: perturbations at the beginning of the prompt affect more subsequent tokens through causal attention, creating a larger "blast radius" for activation divergence. Middle
  perturbations leave the initial token representations intact, providing a more stable computational foundation for deep layers.

  3.5 Context Length Effect

  Though not a primary hypothesis, context length shows a statistically significant but small effect:

  ┌────────────────┬─────────────┐
  │ Context Length │ Cosine Mean │
  ├────────────────┼─────────────┤
  │ 512            │ 0.9696      │
  ├────────────────┼─────────────┤
  │ 1024           │ 0.9626      │
  ├────────────────┼─────────────┤
  │ 2048           │ 0.9595      │
  ├────────────────┼─────────────┤
  │ 4096           │ 0.9621      │
  └────────────────┴─────────────┘

  Pairwise Cohen's d values are all < 0.2 (negligible to small). There is a slight trend toward more divergence at longer contexts, but it plateaus. Within shallow/middle layers, longer contexts actually show
   higher similarity (Spearman rho = +0.63 and +0.61), while in deep layers the trend reverses (rho = -0.21). This suggests that longer contexts provide more stable shallow representations but allow more
  divergent deep computations.

  3.6 Metric Correlations (Phase 2)

  ┌───────────────────────────────┬───────────┬──────────────┐
  │          Metric Pair          │ Pearson r │ Spearman rho │
  ├───────────────────────────────┼───────────┼──────────────┤
  │ cosine_mean vs l2_mean        │ -0.962    │ -0.984       │
  ├───────────────────────────────┼───────────┼──────────────┤
  │ cosine_mean vs cosine_segment │ 0.744     │ 0.946        │
  ├───────────────────────────────┼───────────┼──────────────┤
  │ cosine_mean vs CKA            │ 0.761     │ 0.980        │
  ├───────────────────────────────┼───────────┼──────────────┤
  │ cosine_segment vs CKA         │ 0.531     │ 0.917        │
  ├───────────────────────────────┼───────────┼──────────────┤
  │ l2_mean vs CKA                │ -0.783    │ -0.965       │
  └───────────────────────────────┴───────────┴──────────────┘

  All metrics are highly correlated (|rho| > 0.91), confirming they capture the same underlying phenomenon. The lower Pearson correlations for CKA indicate non-linear relationships, consistent with CKA's
  known behavior at extreme similarity values.

  3.7 Extreme Cases (Phase 2)

  Most divergent (lowest cosine):
  - All 10 worst cases are at layers 44 or 48
  - 8/10 are multi_turn_agent scenarios at beginning position
  - The most extreme: multi_turn_agent_type1_L512_R0.25_Pbeginning_pair2, layer 48, cosine=0.784

  Most preserved (highest cosine):
  - All 10 best cases are at layer 0 (or layer 4)
  - All are tool_use_agent scenarios at longer context lengths (2048, 4096)
  - The most preserved: tool_use_agent_type1_L4096_R0.5_Pmiddle_pair2, layer 0, cosine=0.99999

  Pattern: Multi-turn agent prompts are more sensitive to perturbation than tool-use agent prompts, especially at the beginning position. This suggests that multi-turn conversation structures create more
  opportunity for perturbation cascading.

---
  4. Phase 3: Type 2 — Semantic Paraphrasing

  Dataset: 2,730 rows, 210 unique experiments
  Design: 4 context lengths x 3 ratios x 3 positions (beginning, middle, end) x 3 pairs x 2 scenario types
  Metrics available: cosine_segment, CKA (no token-level cosine_mean/l2 — paraphrasing changes token count, so token alignment is not applicable; only segment-level mean-pooled comparison is valid)

  4.1 Overall Descriptive Statistics

  ┌────────────────┬────────┬────────┬────────┬────────┐
  │     Metric     │  Mean  │  Std   │  Min   │  Max   │
  ├────────────────┼────────┼────────┼────────┼────────┤
  │ cosine_segment │ 0.9841 │ 0.0232 │ 0.8449 │ 0.9999 │
  ├────────────────┼────────┼────────┼────────┼────────┤
  │ CKA            │ 0.7146 │ 0.2469 │ 0.0031 │ 0.9999 │
  └────────────────┴────────┴────────┴────────┴────────┘

  Segment-level similarity is generally high (0.98), confirming that semantic paraphrases preserve the meaning captured in activation space. However, CKA is notably lower (0.71), indicating that while
  mean-pooled representations are similar, the internal structure of segment activations differs more.

  4.2 Layer Depth Pattern

  ┌───────┬────────────────┬───────┐
  │ Layer │ cosine_segment │  CKA  │
  ├───────┼────────────────┼───────┤
  │ 0     │ 0.9949         │ 0.651 │
  ├───────┼────────────────┼───────┤
  │ 4     │ 0.9950         │ 0.764 │
  ├───────┼────────────────┼───────┤
  │ 8     │ 0.9932         │ 0.747 │
  ├───────┼────────────────┼───────┤
  │ 12    │ 0.9924         │ 0.783 │
  ├───────┼────────────────┼───────┤
  │ 16    │ 0.9885         │ 0.790 │
  ├───────┼────────────────┼───────┤
  │ 20    │ 0.9852         │ 0.786 │
  ├───────┼────────────────┼───────┤
  │ 24    │ 0.9826         │ 0.775 │
  ├───────┼────────────────┼───────┤
  │ 28    │ 0.9784         │ 0.759 │
  ├───────┼────────────────┼───────┤
  │ 32    │ 0.9796         │ 0.721 │
  ├───────┼────────────────┼───────┤
  │ 36    │ 0.9803         │ 0.679 │
  ├───────┼────────────────┼───────┤
  │ 40    │ 0.9794         │ 0.607 │
  ├───────┼────────────────┼───────┤
  │ 44    │ 0.9726         │ 0.603 │
  ├───────┼────────────────┼───────┤
  │ 48    │ 0.9712         │ 0.625 │
  └───────┴────────────────┴───────┘

  Depth group summary (cosine_segment):

  ┌─────────┬────────┬────────┐
  │  Group  │  Mean  │  Std   │
  ├─────────┼────────┼────────┤
  │ Shallow │ 0.9939 │ 0.0099 │
  ├─────────┼────────┼────────┤
  │ Middle  │ 0.9837 │ 0.0283 │
  ├─────────┼────────┼────────┤
  │ Deep    │ 0.9766 │ 0.0235 │
  └─────────┴────────┴────────┘

  Spearman rho (layer vs cosine_segment) = -0.599 (p < 1e-265). The depth effect exists but is much weaker than Phase 2's -0.938. The divergence profile also differs: instead of a sharp inflection at layer
  28, Phase 3 shows a more gradual decline that partially plateaus in layers 32-40 before a final dip at 44-48.

  CKA tells a different story: It peaks in middle layers (16-20, ~0.79) and drops in both shallow (0.65 at layer 0) and deep layers (0.60-0.63 at layers 40-48). The low shallow CKA is notable — it suggests
  that even at layer 0, semantically equivalent paraphrases have substantially different activation structures, even though their mean representations are similar (cosine_segment=0.995). This is consistent
  with BPE tokenization differences creating very different token-level patterns that converge at the segment level.

  4.3 Ratio Effect in Phase 3 — Reversed Direction

  Unlike Phase 2, the ratio effect is highly significant in Phase 3, but in the opposite direction to the original hypothesis:

  ┌───────┬─────────────────────┐
  │ Ratio │ cosine_segment Mean │
  ├───────┼─────────────────────┤
  │ 0.10  │ 0.9764              │
  ├───────┼─────────────────────┤
  │ 0.25  │ 0.9838              │
  ├───────┼─────────────────────┤
  │ 0.50  │ 0.9915              │
  └───────┴─────────────────────┘

  Higher ratios show higher similarity. All pairwise comparisons are significant:

  ┌─────────────┬───────────┬─────────┐
  │ Comparison  │ Cohen's d │  p_adj  │
  ├─────────────┼───────────┼─────────┤
  │ 0.1 vs 0.25 │ -0.28     │ < 1e-7  │
  ├─────────────┼───────────┼─────────┤
  │ 0.1 vs 0.5  │ -0.66     │ < 1e-43 │
  ├─────────────┼───────────┼─────────┤
  │ 0.25 vs 0.5 │ -0.46     │ < 1e-18 │
  └─────────────┴───────────┴─────────┘

  Interpretation: This seemingly paradoxical result has a logical explanation. In Type 2 (paraphrasing), a larger perturbation ratio means paraphrasing a larger segment of the prompt. When more of the prompt
  is paraphrased, the paraphrased segment includes more surrounding context that helps anchor the meaning. A small paraphrased region (10%) might change critical local phrasing without enough context to
  maintain semantic alignment, while a large paraphrased region (50%) includes enough redundant information that the model converges to a similar representation.

  This is also consistent with the ratio effect within depth groups being significant at all levels (Shallow: p < 1e-32, Middle: p < 1e-25, Deep: p < 1e-14) — it is a genuine, pervasive effect.

  4.4 Position Effect in Phase 3

  ┌───────────┬─────────────────────┐
  │ Position  │ cosine_segment Mean │
  ├───────────┼─────────────────────┤
  │ Beginning │ 0.9916              │
  ├───────────┼─────────────────────┤
  │ End       │ 0.9808              │
  ├───────────┼─────────────────────┤
  │ Middle    │ 0.9796              │
  └───────────┴─────────────────────┘

  Beginning is significantly more similar than end/middle (Cohen's d ~0.58, p < 1e-10), while end and middle are not significantly different (d=0.04, p=0.77).

  This pattern intensifies in deeper layers:

  ┌─────────────┬───────────┬────────┬────────┬──────────────┐
  │ Depth Group │ Beginning │  End   │ Middle │   p-value    │
  ├─────────────┼───────────┼────────┼────────┼──────────────┤
  │ Shallow     │ 0.9963    │ 0.9929 │ 0.9923 │ 0.145 (n.s.) │
  ├─────────────┼───────────┼────────┼────────┼──────────────┤
  │ Middle      │ 0.9932    │ 0.9800 │ 0.9774 │ < 1e-5       │
  ├─────────────┼───────────┼────────┼────────┼──────────────┤
  │ Deep        │ 0.9865    │ 0.9717 │ 0.9713 │ < 1e-12      │
  └─────────────┴───────────┴────────┴────────┴──────────────┘

  Interpretation: Paraphrasing at the beginning of the prompt produces the least divergence — the opposite of Phase 2's finding. This reversal makes sense: in Type 1, beginning replacement introduces
  different content, causing causal attention to propagate different information forward. In Type 2, beginning paraphrase keeps the same meaning while the rest of the prompt remains literally identical,
  allowing the model to converge early and maintain stability. For end/middle paraphrasing, the subsequent literal text is shorter or split, giving less opportunity for re-convergence.

  4.5 Context Length Effect in Phase 3

  ┌────────────────┬─────────────────────┐
  │ Context Length │ cosine_segment Mean │
  ├────────────────┼─────────────────────┤
  │ 512            │ 0.9701              │
  ├────────────────┼─────────────────────┤
  │ 1024           │ 0.9838              │
  ├────────────────┼─────────────────────┤
  │ 2048           │ 0.9901              │
  ├────────────────┼─────────────────────┤
  │ 4096           │ 0.9928              │
  └────────────────┴─────────────────────┘

  A clear monotonic increase — longer contexts produce more similar paraphrase representations (Cohen's d = -0.90 for 512 vs 4096, a large effect). This is the strongest context length effect observed. Longer
   prompts provide more redundant context around the paraphrased segment, helping the model converge to equivalent representations.

  4.6 Extreme Cases (Phase 3)

  Most divergent: All 10 lowest cosine_segment values (0.844-0.857) are from multi_turn_agent at L=512, ratio=0.1, layers 24-28, at middle or end positions. The convergence of these extreme cases on a single
  condition (short context, small ratio, non-beginning position) reinforces that Type 2 divergence is maximized when there is minimal context to anchor the paraphrased meaning.

  Most preserved: All top cases are tool_use_agent at L=4096, ratio=0.5, layer 0, cosine_segment > 0.9999. Maximum preservation occurs with maximum context and maximum paraphrased region — consistent with the
   redundancy hypothesis.

---
  5. Cross-Phase Comparison

  ┌──────────────────────────┬───────────────────────────────┬───────────────────────────────────────────────────┐
  │          Aspect          │ Phase 2 (Content Replacement) │           Phase 3 (Semantic Paraphrase)           │
  ├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Primary metric           │ cosine_mean (token-level)     │ cosine_segment (segment-level)                    │
  ├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Overall similarity       │ 0.9635                        │ 0.9841                                            │
  ├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Layer depth effect       │ Very strong (rho=-0.94)       │ Moderate (rho=-0.60)                              │
  ├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Inflection point         │ Layer 28 (sharp)              │ Gradual, no clear inflection                      │
  ├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Ratio effect             │ None (p=0.95)                 │ Strong but reversed (higher ratio = more similar) │
  ├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Position: most divergent │ Beginning                     │ End/Middle                                        │
  ├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Context length effect    │ Weak (d < 0.2)                │ Strong (d = 0.90)                                 │
  ├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────────────────┤
  │ Most sensitive scenario  │ multi_turn_agent, beginning   │ multi_turn_agent, short context                   │
  └──────────────────────────┴───────────────────────────────┴───────────────────────────────────────────────────┘

  The two perturbation types reveal fundamentally different propagation mechanisms:

  1. Content replacement (Type 1) triggers an architecture-driven cascade: the perturbation signal amplifies through depth regardless of its magnitude, suggesting that the model's layer-by-layer processing
    inherently amplifies any distributional shift. The threshold behavior (no ratio effect) implies the model's attention mechanism detects and reacts to any content change, then the cascade is governed by
    network dynamics.
  2. Semantic paraphrasing (Type 2) triggers a context-dependent convergence: the model can partially recover equivalent representations when the meaning is preserved, but this recovery depends on having
    sufficient surrounding context. The reversed ratio and position effects point to a mechanism where redundant literal context helps "pull" paraphrased representations back toward the original.

---
  6. Key Findings Summary

  1. Layer depth is the dominant factor in activation divergence. The Spearman correlation of -0.94 and Cohen's d of 2.34 make this the most robust finding. The divergence accelerates non-linearly after layer
      28 in Qwen3-30B-A3B.
  2. Perturbation magnitude (10-50%) does not matter for content replacement. This is a threshold effect — once any content is replaced, the cascading divergence is determined by network architecture, not
    perturbation size. This has implications for adversarial robustness research.
  3. Perturbation position matters, but differently for different perturbation types. Beginning-position content replacement causes the most divergence (causal attention propagation), while beginning-position
      paraphrasing causes the least (early convergence with preserved semantics).
  4. CKA collapses dramatically in deep layers for content replacement (from 0.999 to 0.44-0.54), more severely than cosine similarity, suggesting that not just the direction but the geometric structure of
    activations is disrupted.
  5. Multi-turn agent scenarios are consistently more sensitive to perturbation than tool-use agent scenarios, across both perturbation types.
  6. Context length strongly modulates paraphrase sensitivity but has minimal effect on content replacement — suggesting that additional context helps the model recover meaning but cannot prevent cascading
    from different content.

---
  7. Limitations

  1. Only 2 positions tested in Phase 2 (beginning, middle) — the end position was missing, limiting the position analysis.
  2. No token-level metrics for Phase 3 — paraphrasing changes token count, so only segment-level comparisons are valid, making cross-phase metric comparison indirect.
  3. Single model tested — findings may be specific to Qwen3-30B-A3B's MoE architecture and may not generalize to dense models or different sizes.
  4. Limited prompt pair diversity — 3 pairs per scenario type; results could be sensitive to specific prompt content.
  5. The H2 null result needs careful interpretation — it may reflect a genuine threshold effect, or it could mean that the ratio range (10-50%) is too narrow / the replacement content too similar across
    ratios.

---
  8. Recommendations

  1. Proceed with the research. The 2/3 hypothesis support, with the core hypothesis (H1) extremely robust, validates the approach.
  2. Investigate the ratio threshold. Test finer-grained ratios (1%, 2%, 5%) to identify the onset threshold where divergence begins, and test whether ratios beyond 50% continue the plateau.
  3. Add the end position to Type 1 experiments to complete the position analysis.
  4. Explore the layer-28 inflection point — what architectural feature of Qwen3-30B-A3B at that depth causes the divergence acceleration? Compare with other model architectures.
  5. Study the multi-turn vs tool-use sensitivity difference — is it driven by prompt structure, conversation template, or the nature of the replaced content?