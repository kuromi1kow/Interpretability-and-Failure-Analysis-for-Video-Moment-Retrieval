#!/usr/bin/env python3
"""
VMR Project: Temporal Interpretability — Attention Visualization & Erasure Tests
Stage 5 of the project.

This script:
1. Extracts cross-attention weights from DETR-style models
2. Plots attention heatmaps over clip positions with GT overlay
3. Compares predicted vs GT saliency curves
4. Runs erasure (faithfulness) tests: mask top-k attended clips, measure drop

Usage:
    # Extract and visualize attention (GPU)
    python src/analysis/attention_analysis.py visualize \
        --checkpoint checkpoints/moment_detr_clip_seed0/best.pt \
        --annotations data/qvhighlights/annotations/highlight_val_release.jsonl \
        --features_dir data/qvhighlights/features/ \
        --output_dir results/interpretability/ \
        --num_examples 20

    # Run erasure tests (GPU)
    python src/analysis/attention_analysis.py erasure \
        --checkpoint checkpoints/moment_detr_clip_seed0/best.pt \
        --annotations data/qvhighlights/annotations/highlight_val_release.jsonl \
        --features_dir data/qvhighlights/features/ \
        --output_dir results/interpretability/ \
        --k_values 1 3 5 10
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


# =============================================================================
# Attention Extraction Hooks
# =============================================================================

class AttentionHook:
    """
    Register forward hooks on Transformer attention layers to capture
    cross-attention weights during inference.

    This is model-agnostic: it hooks into any nn.MultiheadAttention layer.
    For Moment-DETR / QD-DETR / CG-DETR, the decoder cross-attention
    is where the model attends from query tokens to video clips.
    """

    def __init__(self, model):
        self.attention_maps = {}
        self.hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model):
        """Register hooks on all MultiheadAttention layers."""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)

    def _hook_fn(self, name):
        def hook(module, input, output):
            # MultiheadAttention returns (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]  # (batch, num_heads, tgt_len, src_len) or (batch, tgt_len, src_len)
                if attn_weights is not None:
                    self.attention_maps[name] = attn_weights.detach().cpu()
        return hook

    def clear(self):
        self.attention_maps = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_attention_heatmap(attention_weights, query_tokens, num_clips,
                           gt_windows, vid_duration, clip_duration,
                           title="", save_path=None):
    """
    Plot an attention heatmap: x-axis = clip index (time), y-axis = query tokens.
    Overlay ground-truth interval boundaries.

    Args:
        attention_weights: (num_query_tokens, num_clips) attention matrix
        query_tokens: list of query token strings
        num_clips: number of video clips
        gt_windows: list of [start, end] ground-truth intervals
        vid_duration: total video duration in seconds
        clip_duration: duration of each clip in seconds
        title: plot title
        save_path: path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, max(3, len(query_tokens) * 0.4 + 1)))

    # Plot heatmap
    im = ax.imshow(attention_weights, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Attention weight", shrink=0.8)

    # X-axis: clip indices → time
    time_ticks = np.linspace(0, num_clips - 1, min(10, num_clips))
    time_labels = [f"{t * clip_duration:.0f}s" for t in time_ticks]
    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels, fontsize=8)
    ax.set_xlabel("Video time", fontsize=10)

    # Y-axis: query tokens
    if query_tokens:
        ax.set_yticks(range(len(query_tokens)))
        ax.set_yticklabels(query_tokens, fontsize=8)
    ax.set_ylabel("Query tokens", fontsize=10)

    # Overlay GT windows
    for window in gt_windows:
        start_clip = window[0] / clip_duration
        end_clip = window[1] / clip_duration
        ax.axvline(x=start_clip, color="lime", linewidth=2, linestyle="--", alpha=0.8)
        ax.axvline(x=end_clip, color="lime", linewidth=2, linestyle="--", alpha=0.8)
        # Shade the GT region
        ax.axvspan(start_clip, end_clip, alpha=0.15, color="lime")

    ax.set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_saliency_comparison(pred_saliency, gt_saliency, gt_windows,
                              vid_duration, clip_duration,
                              title="", save_path=None):
    """
    Plot predicted vs ground-truth saliency curves.

    Args:
        pred_saliency: (num_clips,) predicted saliency scores
        gt_saliency: (num_clips,) ground-truth saliency scores
        gt_windows: list of [start, end] intervals
        vid_duration: total video duration
        clip_duration: duration per clip
        save_path: output path
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))

    num_clips = len(pred_saliency)
    x = np.arange(num_clips) * clip_duration

    # Normalize both to [0, 1] for comparison
    pred_norm = (pred_saliency - pred_saliency.min()) / (pred_saliency.max() - pred_saliency.min() + 1e-8)
    gt_norm = (gt_saliency - gt_saliency.min()) / (gt_saliency.max() - gt_saliency.min() + 1e-8)

    ax.plot(x, gt_norm, color="#2E5A88", linewidth=2, label="GT saliency", alpha=0.8)
    ax.plot(x, pred_norm, color="#E8634A", linewidth=2, label="Predicted saliency", alpha=0.8)
    ax.fill_between(x, gt_norm, alpha=0.15, color="#2E5A88")

    # Shade GT windows
    for window in gt_windows:
        ax.axvspan(window[0], window[1], alpha=0.2, color="lime",
                   label="GT moment" if window == gt_windows[0] else "")

    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Normalized saliency", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, vid_duration)
    ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_saliency_correlation(pred_saliency, gt_saliency):
    """Compute Pearson and Spearman correlation between predicted and GT saliency."""
    from scipy.stats import pearsonr, spearmanr

    pearson_r, pearson_p = pearsonr(pred_saliency, gt_saliency)
    spearman_r, spearman_p = spearmanr(pred_saliency, gt_saliency)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }


# =============================================================================
# Erasure / Faithfulness Tests
# =============================================================================

def erasure_test_single(model, features, text_features, attention_weights,
                        k, method="zero"):
    """
    Mask the top-k most-attended clips and re-run inference.

    Args:
        model: the VMR model
        features: (1, num_clips, feat_dim) video features
        text_features: (1, seq_len, feat_dim) text features
        attention_weights: (num_clips,) per-clip attention importance
        k: number of top clips to mask
        method: "zero" (zero out features) or "mean" (replace with mean)

    Returns:
        predictions with masked features
    """
    # Get top-k clip indices
    top_k_indices = np.argsort(-attention_weights)[:k]

    # Clone features
    masked_features = features.clone()

    if method == "zero":
        masked_features[0, top_k_indices, :] = 0.0
    elif method == "mean":
        mean_feat = features[0].mean(dim=0, keepdim=True)
        masked_features[0, top_k_indices, :] = mean_feat

    # Re-run inference
    with torch.no_grad():
        outputs = model(masked_features, text_features)

    return outputs


def plot_erasure_results(erasure_results, output_dir):
    """
    Plot erasure test results: performance vs number of clips removed.

    erasure_results: dict {
        "model": {k: {"R1@0.5": val, "method": "zero"}, ...},
        "random": {k: {"R1@0.5": val}, ...}
    }
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for condition, results in erasure_results.items():
        ks = sorted(results.keys())
        values = [results[k].get("R1@0.5", 0) for k in ks]
        label = f"Top-{condition}" if condition != "random" else "Random erasure"
        style = "-o" if condition != "random" else "--s"
        color = "#2E5A88" if condition == "attention" else "#E8634A"
        ax.plot(ks, values, style, label=label, color=color, linewidth=2, markersize=6)

    ax.set_xlabel("Number of clips removed", fontsize=11)
    ax.set_ylabel("R1@0.5 (%)", fontsize=11)
    ax.set_title("Erasure Faithfulness Test", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "erasure_test.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# Placeholder: Full Pipeline
# =============================================================================

def visualize_command(args):
    """
    Full visualization pipeline.
    NOTE: This is a template — you will need to adapt the model loading
    and inference code to match your specific Lighthouse/Moment-DETR setup.
    """
    print("=" * 60)
    print("Attention Visualization Pipeline")
    print("=" * 60)
    print()
    print("This script provides the analysis framework.")
    print("To complete the pipeline, you need to:")
    print()
    print("  1. Load your trained model checkpoint")
    print("  2. Register AttentionHook(model)")
    print("  3. Run inference on val examples")
    print("  4. Extract attention maps from hook.attention_maps")
    print("  5. Call plot_attention_heatmap() and plot_saliency_comparison()")
    print()
    print("Example integration with Lighthouse:")
    print()
    print("  from lighthouse.models import MomentDETR")
    print("  model = MomentDETR.from_pretrained(args.checkpoint)")
    print("  hook = AttentionHook(model)")
    print("  model.eval()")
    print("  with torch.no_grad():")
    print("      output = model(video_features, text_features)")
    print("  # hook.attention_maps now contains all attention weights")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Demo: generate a synthetic example to verify plotting works
    print("Generating demo plots with synthetic data...")

    # Synthetic attention heatmap
    np.random.seed(42)
    num_clips = 75
    num_tokens = 8
    attn = np.random.rand(num_tokens, num_clips)
    # Make a peak in the middle to simulate real attention
    attn[:, 25:40] += 1.5
    attn = attn / attn.sum(axis=1, keepdims=True)

    query_tokens = ["the", "person", "adds", "salt", "to", "the", "pan", "[EOS]"]
    gt_windows = [[34.0, 42.0]]
    vid_duration = 150.0
    clip_duration = vid_duration / num_clips

    plot_attention_heatmap(
        attn, query_tokens, num_clips,
        gt_windows, vid_duration, clip_duration,
        title="Demo: Cross-Attention over Video Clips",
        save_path=output_dir / "demo_attention_heatmap.pdf"
    )
    print(f"  Saved demo heatmap: {output_dir / 'demo_attention_heatmap.pdf'}")

    # Synthetic saliency comparison
    pred_saliency = np.random.rand(num_clips) * 0.3
    pred_saliency[20:45] += np.random.rand(25) * 0.7
    gt_saliency = np.zeros(num_clips)
    gt_saliency[25:40] = np.random.rand(15) * 0.5 + 0.5

    plot_saliency_comparison(
        pred_saliency, gt_saliency, gt_windows,
        vid_duration, clip_duration,
        title="Demo: Predicted vs GT Saliency",
        save_path=output_dir / "demo_saliency_comparison.pdf"
    )
    print(f"  Saved demo saliency: {output_dir / 'demo_saliency_comparison.pdf'}")

    # Demo erasure plot
    erasure_results = {
        "attention": {1: {"R1@0.5": 42.0}, 3: {"R1@0.5": 35.0}, 5: {"R1@0.5": 28.0}, 10: {"R1@0.5": 18.0}},
        "random": {1: {"R1@0.5": 43.5}, 3: {"R1@0.5": 41.0}, 5: {"R1@0.5": 38.5}, 10: {"R1@0.5": 34.0}},
    }
    plot_erasure_results(erasure_results, output_dir)

    print()
    print("Demo plots generated. Replace synthetic data with real model outputs.")


def erasure_command(args):
    """Erasure test pipeline (template)."""
    print("=" * 60)
    print("Erasure Faithfulness Test Pipeline")
    print("=" * 60)
    print()
    print("To run erasure tests:")
    print()
    print("  1. Load model and attach AttentionHook")
    print("  2. Run inference on all val examples, collecting attention maps")
    print("  3. For each k in [1, 3, 5, 10]:")
    print("     a. Remove top-k attended clips (zero out features)")
    print("     b. Re-run inference")
    print("     c. Compute metrics on erased predictions")
    print("  4. Also run random erasure as baseline")
    print("  5. Compare: if attention erasure drops more than random,")
    print("     attention is faithful")
    print()
    print(f"k values to test: {args.k_values}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Attention Analysis & Erasure Tests")
    subparsers = parser.add_subparsers(dest="command")

    # Visualize
    vis_parser = subparsers.add_parser("visualize")
    vis_parser.add_argument("--checkpoint", type=str, default=None)
    vis_parser.add_argument("--annotations", type=str, default=None)
    vis_parser.add_argument("--features_dir", type=str, default=None)
    vis_parser.add_argument("--output_dir", type=str, default="results/interpretability")
    vis_parser.add_argument("--num_examples", type=int, default=20)

    # Erasure
    era_parser = subparsers.add_parser("erasure")
    era_parser.add_argument("--checkpoint", type=str, default=None)
    era_parser.add_argument("--annotations", type=str, default=None)
    era_parser.add_argument("--features_dir", type=str, default=None)
    era_parser.add_argument("--output_dir", type=str, default="results/interpretability")
    era_parser.add_argument("--k_values", type=int, nargs="+", default=[1, 3, 5, 10])

    args = parser.parse_args()

    if args.command == "visualize":
        visualize_command(args)
    elif args.command == "erasure":
        erasure_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
