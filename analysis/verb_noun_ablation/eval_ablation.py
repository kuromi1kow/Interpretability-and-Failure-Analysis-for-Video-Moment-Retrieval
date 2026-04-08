#!/usr/bin/env python3
"""Run QD-DETR evaluation with original, verb-masked, and noun-masked text features."""

import os
import sys
import json
import logging

LIGHTHOUSE_ROOT = "/projectnb/cs585/projects/VMR/vmr_project/lighthouse"
PROJECT_ROOT = "/projectnb/cs585/projects/VMR/vmr_project"
sys.path.insert(0, LIGHTHOUSE_ROOT)
sys.path.insert(0, os.path.join(LIGHTHOUSE_ROOT, "training"))
os.chdir(LIGHTHOUSE_ROOT)

import torch
import torch.backends.cudnn as cudnn
from easydict import EasyDict
from training.config import BaseOptions
from training.evaluate import setup_model, eval_epoch
from training.dataset import StartEndDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT = os.path.join(LIGHTHOUSE_ROOT, "results/qd_detr/qvhighlight/clip_slowfast/best.ckpt")
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis/verb_noun_ablation/outputs")

TEXT_FEAT_CONDITIONS = {
    "original": os.path.join(PROJECT_ROOT, "data/qvhighlights/txt_features/clip_text"),
    "verb_masked": os.path.join(PROJECT_ROOT, "data/qvhighlights/txt_features/clip_text_verb_masked"),
    "noun_masked": os.path.join(PROJECT_ROOT, "data/qvhighlights/txt_features/clip_text_noun_masked"),
}


def run_eval_with_text_dir(text_dir, condition_name):
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {condition_name} | text_dir: {text_dir}")
    logger.info(f"{'='*60}")

    if not os.path.isdir(text_dir):
        logger.warning(f"Dir not found: {text_dir}. Skipping.")
        return None

    # Build config with positional args
    config = BaseOptions(
        model="qd_detr",
        dataset="qvhighlight",
        feature="clip_slowfast",
        resume=CHECKPOINT,
        domain=None,
    )
    config.parse(); opt = config.option

    # Override
    opt.t_feat_dir = text_dir
    opt.model_path = CHECKPOINT
    opt.eval_split_name = "val"
    opt.device = "cuda" if torch.cuda.is_available() else "cpu"

    results_dir = os.path.join(OUT_DIR, f"eval_{condition_name}")
    os.makedirs(results_dir, exist_ok=True)
    opt.results_dir = results_dir

    cudnn.benchmark = True

    dataset_config = EasyDict(
        dset_name=opt.dset_name, domain=None,
        data_path=opt.eval_path, ctx_mode=opt.ctx_mode,
        v_feat_dirs=opt.v_feat_dirs, a_feat_dirs=opt.a_feat_dirs,
        q_feat_dir=text_dir,
        q_feat_type="last_hidden_state",
        v_feat_types=opt.v_feat_types, a_feat_types=opt.a_feat_types,
        max_q_l=opt.max_q_l, max_v_l=opt.max_v_l, max_a_l=opt.max_a_l,
        clip_len=opt.clip_length, max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type, load_labels=True,
    )

    eval_dataset = StartEndDataset(**dataset_config)
    model, criterion, _, _ = setup_model(opt)
    ckpt = torch.load(CHECKPOINT, weights_only=False, map_location=opt.device)
    model.load_state_dict(ckpt["model"])
    logger.info(f"Loaded checkpoint: {CHECKPOINT}")

    save_name = f"ablation_{condition_name}_val_preds.jsonl"
    with torch.no_grad():
        metrics, _, _ = eval_epoch(None, model, eval_dataset, opt, save_name, criterion)

    logger.info(f"{condition_name}: {json.dumps(metrics['brief'], indent=2)}")
    with open(os.path.join(results_dir, f"{condition_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics["brief"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}
    for cond, text_dir in TEXT_FEAT_CONDITIONS.items():
        result = run_eval_with_text_dir(text_dir, cond)
        if result:
            all_results[cond] = result

    with open(os.path.join(OUT_DIR, "ablation_comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print("VERB/NOUN ABLATION COMPARISON (QD-DETR on QVHighlights val)")
    print("="*70)
    for cond, m in all_results.items():
        print(f"  {cond:15s}  R1@0.5={m['MR-full-R1@0.5']:6.2f}  "
              f"R1@0.7={m['MR-full-R1@0.7']:6.2f}  mAP={m['MR-full-mAP']:6.2f}")
    if "original" in all_results and "verb_masked" in all_results:
        d = all_results["original"]["MR-full-R1@0.5"] - all_results["verb_masked"]["MR-full-R1@0.5"]
        print(f"\n  Verb masking R1@0.5 drop: {d:+.2f}")
    if "original" in all_results and "noun_masked" in all_results:
        d = all_results["original"]["MR-full-R1@0.5"] - all_results["noun_masked"]["MR-full-R1@0.5"]
        print(f"  Noun masking R1@0.5 drop: {d:+.2f}")


if __name__ == "__main__":
    main()
