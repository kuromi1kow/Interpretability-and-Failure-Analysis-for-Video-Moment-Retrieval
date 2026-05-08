# Presentation Visuals

Use these figures for the experiment slides:

- Temporal bias:
  [temporal_bias_length_buckets.png](/Users/zukhriddin/Documents/Interpretability-and-Failure-Analysis-for-Video-Moment-Retrieval/docs/figures/temporal_bias_length_buckets.png)
  This is the clean summary figure. It shows the short-moment failure clearly.
- Temporal bias example slide:
  [temporal_bias_failure_examples.png](/Users/zukhriddin/Documents/Interpretability-and-Failure-Analysis-for-Video-Moment-Retrieval/docs/figures/temporal_bias_failure_examples.png)
  Use this if you want one “real examples” slide after the summary chart.

- Verb vs noun:
  [verb_noun_metric_drop.png](/Users/zukhriddin/Documents/Interpretability-and-Failure-Analysis-for-Video-Moment-Retrieval/docs/figures/verb_noun_metric_drop.png)
  This is the main experiment result.
- Verb vs noun example slide:
  [verb_noun_query_examples.png](/Users/zukhriddin/Documents/Interpretability-and-Failure-Analysis-for-Video-Moment-Retrieval/docs/figures/verb_noun_query_examples.png)
  Use this to explain what “verb tokens” and “noun tokens” mean.

- Head ablation:
  [head_ablation_heatmap.png](/Users/zukhriddin/Documents/Interpretability-and-Failure-Analysis-for-Video-Moment-Retrieval/docs/figures/head_ablation_heatmap.png)
  This is the strongest slide figure because it shows all 32 heads at once.
- Head ablation backup figure:
  [head_ablation_top_heads.png](/Users/zukhriddin/Documents/Interpretability-and-Failure-Analysis-for-Video-Moment-Retrieval/docs/figures/head_ablation_top_heads.png)
  Use this if you want a simpler “top harmful heads” slide.

Suggested talk track:

- Temporal bias:
  “The models are much better than bias-only baselines, but they systematically over-predict short moments.”
- Verb vs noun:
  “Masking nouns hurts more than masking verbs, which suggests the model relies more on object/entity cues than action semantics.”
- Head ablation:
  “No single head is critical. The boundary prediction behavior is distributed across the transformer.”
