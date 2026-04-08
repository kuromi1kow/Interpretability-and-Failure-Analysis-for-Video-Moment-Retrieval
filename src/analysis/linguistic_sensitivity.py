#!/usr/bin/env python3
"""
VMR Project: Verb vs. Noun Sensitivity Analysis
Stage 7 of the project — linguistic ablation experiments.

This script:
1. Parses validation queries with spaCy to identify verbs and nouns
2. Creates ablated query variants (verb-masked, noun-masked, verb-swapped)
3. Re-encodes modified queries through CLIP text encoder
4. Runs inference with modified text features
5. Compares metrics across ablation conditions

Usage:
    # Step 1: Generate ablated queries (CPU)
    python src/analysis/linguistic_sensitivity.py generate \
        --annotations data/qvhighlights/annotations/highlight_val_release.jsonl \
        --output_dir results/linguistic/

    # Step 2: Re-encode queries through CLIP (GPU)
    python src/analysis/linguistic_sensitivity.py encode \
        --queries_dir results/linguistic/ \
        --output_dir results/linguistic/features/

    # Step 3: Analyze results after inference (CPU)
    python src/analysis/linguistic_sensitivity.py analyze \
        --original_pred results/predictions/model_val.jsonl \
        --verb_masked_pred results/predictions/model_val_verb_masked.jsonl \
        --noun_masked_pred results/predictions/model_val_noun_masked.jsonl \
        --verb_swapped_pred results/predictions/model_val_verb_swapped.jsonl \
        --gt data/qvhighlights/annotations/highlight_val_release.jsonl \
        --output_dir results/linguistic/
"""

import json
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

try:
    import spacy
    NLP = None  # Loaded lazily
except ImportError:
    print("WARNING: spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm")
    NLP = None


def get_nlp():
    """Lazily load spaCy model."""
    global NLP
    if NLP is None:
        NLP = spacy.load("en_core_web_sm")
    return NLP


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


# =============================================================================
# Step 1: Generate ablated queries
# =============================================================================

def parse_query(query):
    """Parse a query and return token info with POS tags."""
    nlp = get_nlp()
    doc = nlp(query)
    tokens = []
    for token in doc:
        tokens.append({
            "text": token.text,
            "pos": token.pos_,
            "lemma": token.lemma_,
            "idx": token.i,
            "is_verb": token.pos_ == "VERB",
            "is_noun": token.pos_ in ("NOUN", "PROPN"),
        })
    return tokens


def mask_verbs(query):
    """Replace all verbs with [MASK]."""
    tokens = parse_query(query)
    result = []
    masked_count = 0
    for t in tokens:
        if t["is_verb"]:
            result.append("[MASK]")
            masked_count += 1
        else:
            result.append(t["text"])
    return " ".join(result), masked_count


def mask_nouns(query):
    """Replace all nouns with [MASK]."""
    tokens = parse_query(query)
    result = []
    masked_count = 0
    for t in tokens:
        if t["is_noun"]:
            result.append("[MASK]")
            masked_count += 1
        else:
            result.append(t["text"])
    return " ".join(result), masked_count


def swap_verbs(query, verb_vocab):
    """Replace each verb with a random different verb from the vocabulary."""
    tokens = parse_query(query)
    result = []
    swapped_count = 0
    for t in tokens:
        if t["is_verb"] and len(verb_vocab) > 1:
            # Pick a random verb that's different from the original
            candidates = [v for v in verb_vocab if v.lower() != t["text"].lower()]
            if candidates:
                replacement = random.choice(candidates)
                result.append(replacement)
                swapped_count += 1
            else:
                result.append(t["text"])
        else:
            result.append(t["text"])
    return " ".join(result), swapped_count


def generate_ablated_queries(annotations_path, output_dir):
    """Generate all ablated query variants and save them."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_jsonl(annotations_path)
    print(f"Loaded {len(data)} validation samples")

    # First pass: collect verb vocabulary from all queries
    print("Collecting verb vocabulary...")
    verb_vocab = set()
    query_stats = {"total": 0, "has_verb": 0, "has_noun": 0}

    for item in data:
        query = item.get("query", "")
        tokens = parse_query(query)
        verbs = [t["text"].lower() for t in tokens if t["is_verb"]]
        nouns = [t["text"].lower() for t in tokens if t["is_noun"]]
        verb_vocab.update(verbs)
        query_stats["total"] += 1
        if verbs:
            query_stats["has_verb"] += 1
        if nouns:
            query_stats["has_noun"] += 1

    verb_vocab = list(verb_vocab)
    print(f"  Verb vocabulary size: {len(verb_vocab)}")
    print(f"  Queries with verbs: {query_stats['has_verb']}/{query_stats['total']}")
    print(f"  Queries with nouns: {query_stats['has_noun']}/{query_stats['total']}")

    # Second pass: generate ablated queries
    print("\nGenerating ablated queries...")
    results = {
        "original": [],
        "verb_masked": [],
        "noun_masked": [],
        "verb_swapped": [],
        "metadata": [],
    }

    random.seed(42)  # Reproducibility for verb swapping

    for item in data:
        qid = item.get("qid", item.get("query_id"))
        query = item.get("query", "")

        # Original
        results["original"].append({"qid": qid, "query": query})

        # Verb masked
        vm_query, vm_count = mask_verbs(query)
        results["verb_masked"].append({"qid": qid, "query": vm_query})

        # Noun masked
        nm_query, nm_count = mask_nouns(query)
        results["noun_masked"].append({"qid": qid, "query": nm_query})

        # Verb swapped
        vs_query, vs_count = swap_verbs(query, verb_vocab)
        results["verb_swapped"].append({"qid": qid, "query": vs_query})

        # Metadata
        results["metadata"].append({
            "qid": qid,
            "original": query,
            "verb_masked": vm_query,
            "noun_masked": nm_query,
            "verb_swapped": vs_query,
            "n_verbs_masked": vm_count,
            "n_nouns_masked": nm_count,
            "n_verbs_swapped": vs_count,
        })

    # Save all variants
    for variant_name, variant_data in results.items():
        out_path = output_dir / f"queries_{variant_name}.jsonl"
        with open(out_path, "w") as f:
            for item in variant_data:
                f.write(json.dumps(item) + "\n")
        print(f"  Saved: {out_path} ({len(variant_data)} queries)")

    # Print examples
    print("\n--- Example ablations ---")
    for i in range(min(5, len(results["metadata"]))):
        m = results["metadata"][i]
        print(f"\n  Original:     {m['original']}")
        print(f"  Verb-masked:  {m['verb_masked']} ({m['n_verbs_masked']} verbs)")
        print(f"  Noun-masked:  {m['noun_masked']} ({m['n_nouns_masked']} nouns)")
        print(f"  Verb-swapped: {m['verb_swapped']} ({m['n_verbs_swapped']} swaps)")

    return results


# =============================================================================
# Step 2: Re-encode queries through CLIP
# =============================================================================

def encode_queries_clip(queries_dir, output_dir):
    """
    Re-encode ablated queries through CLIP text encoder.
    This requires the CLIP model weights (small download).
    """
    try:
        import torch
        from transformers import CLIPTokenizer, CLIPTextModel
    except ImportError:
        print("ERROR: Need torch and transformers. Install them first.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries_dir = Path(queries_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLIP text encoder
    print("Loading CLIP text encoder...")
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_model = CLIPTextModel.from_pretrained(model_name).to(device).eval()

    # Encode each variant
    variants = ["original", "verb_masked", "noun_masked", "verb_swapped"]

    for variant in variants:
        query_file = queries_dir / f"queries_{variant}.jsonl"
        if not query_file.exists():
            print(f"  Skipping {variant}: file not found")
            continue

        queries = load_jsonl(query_file)
        print(f"\n  Encoding {variant} ({len(queries)} queries)...")

        features = {}
        batch_size = 64

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            texts = [q["query"] for q in batch]
            qids = [q["qid"] for q in batch]

            inputs = tokenizer(texts, padding=True, truncation=True,
                             max_length=77, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = text_model(**inputs)
                # Use pooled output (CLS token)
                text_feats = outputs.pooler_output  # (batch, 512)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            for qid, feat in zip(qids, text_feats.cpu().numpy()):
                features[qid] = feat.tolist()

        # Save as numpy
        out_path = output_dir / f"text_features_{variant}.npy"
        # Save as dict: {qid: feature_vector}
        np.save(out_path, features)
        print(f"  Saved: {out_path}")

    print("\nEncoding complete!")


# =============================================================================
# Step 3: Analyze results
# =============================================================================

def analyze_results(original_pred, verb_masked_pred, noun_masked_pred,
                    verb_swapped_pred, gt_path, output_dir):
    """Compare metrics across ablation conditions."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.eval.evaluate import compute_recall_at_iou

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    gt_data = load_jsonl(gt_path)
    ground_truths = {}
    for item in gt_data:
        qid = item.get("qid", item.get("query_id"))
        ground_truths[qid] = item.get("relevant_windows", [])

    # Load predictions for each condition
    conditions = {
        "Full query": original_pred,
        "Verb-masked": verb_masked_pred,
        "Noun-masked": noun_masked_pred,
        "Verb-swapped": verb_swapped_pred,
    }

    all_results = {}

    print("=" * 60)
    print("Linguistic Sensitivity Results")
    print("=" * 60)

    for cond_name, pred_path in conditions.items():
        if pred_path is None or not Path(pred_path).exists():
            print(f"\n  [{cond_name}]: SKIPPED (file not found)")
            continue

        pred_data = load_jsonl(pred_path)
        predictions = {}
        for item in pred_data:
            qid = item.get("qid", item.get("query_id"))
            if "pred_relevant_windows" in item:
                predictions[qid] = item["pred_relevant_windows"][0]
            elif "predicted_times" in item:
                predictions[qid] = item["predicted_times"][0]

        recall = compute_recall_at_iou(predictions, ground_truths)
        all_results[cond_name] = recall

        print(f"\n  {cond_name}:")
        for k, v in recall.items():
            print(f"    {k}: {v:.2f}")

    if len(all_results) < 2:
        print("\nNeed at least 2 conditions to compare. Run inference with ablated queries first.")
        return

    # Compute deltas relative to full query
    if "Full query" in all_results:
        print("\n--- Performance Drops ---")
        full = all_results["Full query"]
        for cond_name, results in all_results.items():
            if cond_name == "Full query":
                continue
            print(f"\n  {cond_name} vs Full query:")
            for metric in ["R1@0.3", "R1@0.5", "R1@0.7"]:
                delta = results[metric] - full[metric]
                print(f"    {metric}: {delta:+.2f} ({'↓' if delta < 0 else '↑'})")

    # Generate figures
    plot_linguistic_comparison(all_results, output_dir)

    # Interpretation
    print("\n--- Interpretation ---")
    if "Verb-masked" in all_results and "Noun-masked" in all_results and "Full query" in all_results:
        full_r = all_results["Full query"]["R1@0.5"]
        verb_r = all_results["Verb-masked"]["R1@0.5"]
        noun_r = all_results["Noun-masked"]["R1@0.5"]

        verb_drop = full_r - verb_r
        noun_drop = full_r - noun_r

        if noun_drop > verb_drop:
            print(f"  Noun masking causes a LARGER drop ({noun_drop:.1f} pts) than verb masking ({verb_drop:.1f} pts)")
            print(f"  → The model relies MORE on object/entity cues than action semantics")
            print(f"  → This is a potential weakness for temporal grounding")
        elif verb_drop > noun_drop:
            print(f"  Verb masking causes a LARGER drop ({verb_drop:.1f} pts) than noun masking ({noun_drop:.1f} pts)")
            print(f"  → The model relies MORE on action semantics (verbs)")
            print(f"  → This suggests the model is learning temporal action cues")
        else:
            print(f"  Verb and noun masking have similar impact ({verb_drop:.1f} vs {noun_drop:.1f} pts)")

    # Save summary
    summary = {"results": {k: {m: float(v) for m, v in r.items()} for k, r in all_results.items()}}
    summary_path = output_dir / "linguistic_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


def plot_linguistic_comparison(results_dict, output_dir):
    """Grouped bar chart of linguistic ablation results."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    conditions = list(results_dict.keys())
    metrics = ["R1@0.3", "R1@0.5", "R1@0.7"]
    x = np.arange(len(metrics))
    width = 0.8 / len(conditions)

    colors = ["#2E5A88", "#E8634A", "#5DAE5F", "#D4A843"]

    for i, cond in enumerate(conditions):
        values = [results_dict[cond].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=cond,
                      color=colors[i % len(colors)], edgecolor="white")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Verb vs. Noun Sensitivity Analysis", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * (len(conditions) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, max(max(r.values()) for r in results_dict.values()) * 1.15)

    plt.tight_layout()
    out_path = output_dir / "linguistic_sensitivity.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Linguistic Sensitivity Analysis")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate ablated queries")
    gen_parser.add_argument("--annotations", type=str, required=True)
    gen_parser.add_argument("--output_dir", type=str, default="results/linguistic")

    # Encode
    enc_parser = subparsers.add_parser("encode", help="Re-encode queries through CLIP")
    enc_parser.add_argument("--queries_dir", type=str, required=True)
    enc_parser.add_argument("--output_dir", type=str, default="results/linguistic/features")

    # Analyze
    ana_parser = subparsers.add_parser("analyze", help="Analyze results across conditions")
    ana_parser.add_argument("--original_pred", type=str, required=True)
    ana_parser.add_argument("--verb_masked_pred", type=str, default=None)
    ana_parser.add_argument("--noun_masked_pred", type=str, default=None)
    ana_parser.add_argument("--verb_swapped_pred", type=str, default=None)
    ana_parser.add_argument("--gt", type=str, required=True)
    ana_parser.add_argument("--output_dir", type=str, default="results/linguistic")

    args = parser.parse_args()

    if args.command == "generate":
        generate_ablated_queries(args.annotations, args.output_dir)
    elif args.command == "encode":
        encode_queries_clip(args.queries_dir, args.output_dir)
    elif args.command == "analyze":
        analyze_results(
            args.original_pred, args.verb_masked_pred,
            args.noun_masked_pred, args.verb_swapped_pred,
            args.gt, args.output_dir
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
