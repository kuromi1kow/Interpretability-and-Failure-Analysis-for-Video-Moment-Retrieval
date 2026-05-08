# Speaker Script — Video Moment Retrieval Presentation

**Speaker:** Zukhriddin Fakhriddinov (solo)
**Target length:** ~5:00 minutes  ·  21 slides
**Pace:** ~150 words/minute. Total ~720 words.

> Tip: pause briefly when you advance the slide — let the audience take in the visual before you start talking.

---

## Slide 1 — Title  *(0:00 – 0:08, ~8s)*

Hi everyone — I'm Zukhriddin. Today I'll walk you through our team's work on video moment retrieval, focusing on what these models do well, where they fail, and why.

## Slide 2 — Live Example  *(0:08 – 0:26, ~18s)*

Quick demo first, so the task is concrete. We feed the model a two-and-a-half-minute vlog and the query *“a woman wearing a glass is speaking in front of the camera.”* CG-DETR returns the window 63 to 92 seconds with 0.98 confidence — and you can see the frames line up: at 1:31 she's looking straight at the camera. Text in, time window out. That's the task.

## Slide 3 — What is Video Moment Retrieval?  *(0:26 – 0:42, ~16s)*

Formally: untrimmed video plus a natural language query in, predicted start and end times out. Unlike classification, the output is a continuous time window. The model has to do visual understanding, language understanding, and cross-modal alignment all at once.

## Slide 4 — What the Model Actually Outputs  *(0:42 – 0:54, ~12s)*

This is the canonical visualization from the QD-DETR paper — same idea with their highlight saliency overlay. The model reads the query and returns a time window. That mental model carries through the rest of the talk.

## Slide 5 — Dataset &amp; Setup  *(0:54 – 1:09, ~15s)*

We worked on QVHighlights — about ten thousand queries over 150-second videos. Pre-extracted CLIP, SlowFast, and PANN features. Trained on BU's SCC cluster using the Lighthouse framework, evaluating three DETR-style baselines.

## Slide 6 — Three Baselines  *(1:09 – 1:23, ~14s)*

Moment-DETR, QD-DETR, and CG-DETR. Going left to right, each adds more query awareness — Moment-DETR is query-agnostic, QD-DETR conditions video features on the query, CG-DETR adds calibrated word-to-moment grounding.

## Slide 7 — Baseline Performance  *(1:23 – 1:41, ~18s)*

Out of the box, query awareness clearly helps. QD-DETR jumps eight points in R1@0.5 over Moment-DETR; CG-DETR is slightly better still. The baselines work — but the much more interesting question is *how* they work. That's what we spent our time on.

## Slide 8 — Three Analysis Directions  *(1:41 – 1:54, ~13s)*

Three angles: are these models exploiting temporal bias? Do they rely on verbs or nouns? And can we localize boundary prediction to specific attention heads?

## Slide 9 — Temporal Bias  *(1:54 – 2:18, ~24s)*

We built three dumb baselines — random, center-of-video, training-distribution mean. All three score around 8 R1@0.5. QD-DETR scores 62, so the models are doing real work. But sliced by length, short moments under 10 seconds are the main failure mode — the model predicts an average of 32 seconds for moments that are actually under 5.

## Slide 10 — Bias Result Summary  *(2:18 – 2:35, ~17s)*

This chart confirms the pattern across all three baselines: short moments are systematically over-predicted. The models can find roughly the right region, but they don't tighten the window when the event is brief.

## Slide 11 — Failure Cases  *(2:35 – 2:46, ~11s)*

Concrete examples. Blue is ground truth, orange is the prediction. You can literally see the predicted windows drifting too early or too wide on short ground truths.

## Slide 12 — Verb vs. Noun Pipeline  *(2:46 – 3:03, ~17s)*

For our second analysis: does the model lean on verbs or nouns? We POS-tagged every query with spaCy, aligned tokens to CLIP's tokenizer, and then zeroed out either all verb or all noun tokens before re-running evaluation.

## Slide 13 — Nouns Matter 2.6x More  *(3:03 – 3:23, ~20s)*

Striking result. Masking nouns drops R1@0.5 by 2.32 points. Masking verbs only drops it by 0.9. That's a 2.6 times bigger impact from nouns. So QD-DETR is grounding moments primarily through entity and object cues — not action semantics.

## Slide 14 — Ablation Chart  *(3:23 – 3:34, ~11s)*

Same pattern across every metric — R1@0.5, R1@0.7, mAP. The noun-masked bar is always the lowest.

## Slide 15 — Tagged Query Examples  *(3:34 – 3:43, ~9s)*

Concrete examples of what was masked — verbs orange, nouns teal. So when we say “remove nouns,” we literally mean these words.

## Slide 16 — Head Ablation  *(3:43 – 4:00, ~17s)*

Third analysis was mechanistic. QD-DETR has four attention modules, eight heads each — 32 total. We zeroed each head individually and measured the drop. The most damaging single head only cost 0.77 R1@0.5 points.

## Slide 17 — Heatmap of All 32 Heads  *(4:00 – 4:12, ~12s)*

This heatmap shows that no single head dominates. Boundary prediction is *distributed* across the transformer — not localized in any one component.

## Slide 18 — Top Harmful Heads  *(4:12 – 4:24, ~12s)*

Even the worst single-head ablation barely dents accuracy. Temporal grounding emerges from the whole network, not from a specialized circuit.

## Slide 19 — Summary  *(4:24 – 4:50, ~26s)*

To wrap up — query-aware models clearly win. The dominant failure mode is short moments. Object nouns matter about 2.6 times more than action verbs. And there are no specialized boundary heads — grounding is a distributed property of the whole transformer. The two highest-leverage future directions are short-moment handling and stronger action semantics.

## Slide 20 — Next Steps  *(4:50 – 5:05, ~15s)*

That points us at multi-head ablations, activation patching to trace causal flow, short-moment augmentation, cross-dataset replication on Charades-STA, and trying action-specialized video-language backbones.

## Slide 21 — Thank You  *(5:05 – 5:10, ~5s)*

Thank you — happy to take questions.

---

### Delivery notes
- Slide 2 (the live demo) is your most important slide for grabbing the audience early. Don't rush it — let them see the frames before you start talking.
- Strongest beats to slow down on: **62 vs 8** on slide 9, **2.6x** on slide 13, and **only -0.77** on slide 18. These are your three “wow” moments.
- If you're running long, the easiest cuts are slides 11 and 15 — say “concrete examples here, you can see how it goes” and move on.
- Slide 8 is a section break — feel free to take a one-second pause before launching into it.
