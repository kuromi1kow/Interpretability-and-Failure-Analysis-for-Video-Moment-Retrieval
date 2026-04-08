#!/usr/bin/env python3
"""Head ablation study for QD-DETR on QVHighlights."""

import os
import sys
import json
import logging
import numpy as np

LIGHTHOUSE_ROOT = "/projectnb/cs585/projects/VMR/vmr_project/lighthouse"
PROJECT_ROOT = "/projectnb/cs585/projects/VMR/vmr_project"
sys.path.insert(0, LIGHTHOUSE_ROOT)
sys.path.insert(0, os.path.join(LIGHTHOUSE_ROOT, "training"))
os.chdir(LIGHTHOUSE_ROOT)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from easydict import EasyDict
from training.config import BaseOptions
from training.evaluate import setup_model, eval_epoch
from training.dataset import StartEndDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT = os.path.join(LIGHTHOUSE_ROOT, "results/qd_detr/qvhighlight/clip_slowfast/best.ckpt")
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis/head_ablation/outputs")


class HeadAblationHook:
    def __init__(self, head_idx, n_heads, embed_dim):
        self.head_idx = head_idx
        self.head_dim = embed_dim // n_heads
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            attn_out = output[0]
        else:
            attn_out = output
        start = self.head_idx * self.head_dim
        end = start + self.head_dim
        if attn_out.dim() == 3:
            attn_out_new = attn_out.clone()
            attn_out_new[..., start:end] = 0.0
            return (attn_out_new,) + output[1:] if isinstance(output, tuple) else attn_out_new
        return output

    def register(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cudnn.benchmark = True

    config = BaseOptions(
        model="qd_detr", dataset="qvhighlight", feature="clip_slowfast",
        resume=CHECKPOINT, domain=None,
    )
    config.parse(); opt = config.option
    opt.model_path = CHECKPOINT
    opt.eval_split_name = "val"
    opt.device = "cuda" if torch.cuda.is_available() else "cpu"
    opt.results_dir = OUT_DIR

    dataset_config = EasyDict(
        dset_name=opt.dset_name, domain=None,
        data_path=opt.eval_path, ctx_mode=opt.ctx_mode,
        v_feat_dirs=opt.v_feat_dirs, a_feat_dirs=opt.a_feat_dirs,
        q_feat_dir=opt.t_feat_dir, q_feat_type="last_hidden_state",
        v_feat_types=opt.v_feat_types, a_feat_types=opt.a_feat_types,
        max_q_l=opt.max_q_l, max_v_l=opt.max_v_l, max_a_l=opt.max_a_l,
        clip_len=opt.clip_length, max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type, load_labels=True,
    )
    eval_dataset = StartEndDataset(**dataset_config)

    model, criterion, _, _ = setup_model(opt)
    ckpt = torch.load(CHECKPOINT, weights_only=False, map_location=opt.device)
    model.load_state_dict(ckpt["model"])
    logger.info(f"Model loaded from {CHECKPOINT}")

    # Find attention modules
    attn_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            attn_modules.append({
                "name": name, "module": module,
                "n_heads": module.num_heads,
                "embed_dim": module.embed_dim,
            })
    logger.info(f"Found {len(attn_modules)} attention modules")
    for am in attn_modules:
        logger.info(f"  {am['name']}: {am['n_heads']} heads, dim={am['embed_dim']}")

    # Baseline
    logger.info("--- Baseline (no ablation) ---")
    with torch.no_grad():
        baseline_metrics, _, _ = eval_epoch(None, model, eval_dataset, opt, "baseline_preds.jsonl", criterion)
    bl = baseline_metrics["brief"]
    logger.info(f"Baseline R1@0.5={bl['MR-full-R1@0.5']:.2f}  R1@0.7={bl['MR-full-R1@0.7']:.2f}")

    # Per-head ablation
    results = []
    for am in attn_modules:
        for head_idx in range(am["n_heads"]):
            tag = f"{am['name']}_h{head_idx}"
            logger.info(f"--- Ablating {tag} ---")

            hook = HeadAblationHook(head_idx, am["n_heads"], am["embed_dim"])
            hook.register(am["module"])

            with torch.no_grad():
                metrics, _, _ = eval_epoch(None, model, eval_dataset, opt, f"{tag}_preds.jsonl", criterion)
            m = metrics["brief"]
            hook.remove()

            results.append({
                "module": am["name"], "head": head_idx,
                "R1@0.5": m["MR-full-R1@0.5"], "R1@0.7": m["MR-full-R1@0.7"],
                "mAP": m["MR-full-mAP"],
                "delta_R1@0.5": round(m["MR-full-R1@0.5"] - bl["MR-full-R1@0.5"], 2),
                "delta_R1@0.7": round(m["MR-full-R1@0.7"] - bl["MR-full-R1@0.7"], 2),
            })
            logger.info(f"  R1@0.5={m['MR-full-R1@0.5']:.2f} (d={results[-1]['delta_R1@0.5']:+.2f})")

    # Save
    output = {
        "baseline": bl,
        "modules": [{"name": am["name"], "n_heads": am["n_heads"], "embed_dim": am["embed_dim"]}
                     for am in attn_modules],
        "ablation_results": results,
    }
    with open(os.path.join(OUT_DIR, "head_ablation_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*70)
    print("HEAD ABLATION SUMMARY (QD-DETR)")
    print("="*70)
    print(f"Baseline: R1@0.5={bl['MR-full-R1@0.5']:.2f}  R1@0.7={bl['MR-full-R1@0.7']:.2f}")
    print("\nTop 10 most impactful heads (largest R1@0.5 drop):")
    for i, r in enumerate(sorted(results, key=lambda r: r["delta_R1@0.5"])[:10]):
        print(f"  {i+1}. {r['module']}.head{r['head']}  "
              f"delta_R1@0.5={r['delta_R1@0.5']:+.2f}  delta_R1@0.7={r['delta_R1@0.7']:+.2f}")


if __name__ == "__main__":
    main()
