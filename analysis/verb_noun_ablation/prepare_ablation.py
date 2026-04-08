#!/usr/bin/env python3
"""Prepare verb-masked and noun-masked query features for ablation.

Uses spaCy for POS tagging. Maps words to CLIP subword token positions
by tokenizing each word individually, then zeros out corresponding rows
in the pre-extracted CLIP text last_hidden_state.

Usage (on SCC, CPU only):
    source /projectnb/cs585/projects/VMR/vmr_env/bin/activate
    cd /projectnb/cs585/projects/VMR/vmr_project
    python analysis/verb_noun_ablation/prepare_ablation.py
"""

import os
import json
import csv
import numpy as np
import spacy
from transformers import CLIPTokenizer
from pathlib import Path

PROJECT_ROOT = "/projectnb/cs585/projects/VMR/vmr_project"
ANNO_DIR = os.path.join(PROJECT_ROOT, "data/qvhighlights/annotations")
ORIG_TEXT_DIR = os.path.join(PROJECT_ROOT, "data/qvhighlights/txt_features/clip_text")
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis/verb_noun_ablation/outputs")

VERB_MASKED_DIR = os.path.join(PROJECT_ROOT, "data/qvhighlights/txt_features/clip_text_verb_masked")
NOUN_MASKED_DIR = os.path.join(PROJECT_ROOT, "data/qvhighlights/txt_features/clip_text_noun_masked")

CLIP_MODEL = "openai/clip-vit-base-patch32"


def get_word_to_token_indices(tokenizer, text, max_length=32):
    """Map each word to its CLIP subword token indices.

    CLIP tokenizer: [SOS] tok1 tok2 ... tokN [EOS] [PAD]...
    Token index 0 = SOS, then subwords, then EOS.
    """
    words = text.lower().split()

    # Tokenize full sentence
    full_ids = tokenizer.encode(text, max_length=max_length, truncation=True)
    # full_ids includes [SOS]=49406 at start, [EOS]=49407 at end

    # Tokenize each word individually (without special tokens)
    word_token_ids = []
    for w in words:
        w_ids = tokenizer.encode(w)[1:-1]  # strip SOS/EOS
        word_token_ids.append(w_ids)

    # Align: walk through full_ids[1:-1] (skip SOS and EOS) matching word tokens
    content_ids = full_ids[1:-1]  # strip SOS, EOS
    word_to_indices = {}
    pos = 0
    for wi, w_ids in enumerate(word_token_ids):
        token_indices = []
        matched = True
        for j, wid in enumerate(w_ids):
            if pos < len(content_ids) and content_ids[pos] == wid:
                token_indices.append(pos + 1)  # +1 because SOS is at index 0
                pos += 1
            else:
                matched = False
                break
        if matched and token_indices:
            word_to_indices[wi] = token_indices
        elif not matched:
            # Skip ahead if mismatch (shouldn't happen often)
            pos += len(w_ids)

    return words, word_to_indices


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VERB_MASKED_DIR, exist_ok=True)
    os.makedirs(NOUN_MASKED_DIR, exist_ok=True)

    print("Loading spaCy and CLIP tokenizer...")
    nlp = spacy.load("en_core_web_sm")
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL)

    # Load annotations
    val_data = []
    with open(os.path.join(ANNO_DIR, "highlight_val_release.jsonl")) as f:
        for line in f:
            val_data.append(json.loads(line.strip()))
    train_data = []
    with open(os.path.join(ANNO_DIR, "highlight_train_release.jsonl")) as f:
        for line in f:
            train_data.append(json.loads(line.strip()))

    all_data = val_data + train_data
    val_qids = {item["qid"] for item in val_data}
    print(f"Processing {len(all_data)} queries ({len(val_data)} val + {len(train_data)} train)...")

    # Query composition analysis
    query_analysis = []
    for item in all_data:
        doc = nlp(item["query"])
        verbs = [t.text for t in doc if t.pos_ == "VERB"]
        nouns = [t.text for t in doc if t.pos_ in ("NOUN", "PROPN")]
        query_analysis.append({
            "qid": item["qid"],
            "query": item["query"],
            "n_words": len(item["query"].split()),
            "n_verbs": len(verbs),
            "n_nouns": len(nouns),
            "verbs": "|".join(verbs),
            "nouns": "|".join(nouns),
            "split": "val" if item["qid"] in val_qids else "train",
        })

    with open(os.path.join(OUT_DIR, "query_composition.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=query_analysis[0].keys())
        w.writeheader()
        w.writerows(query_analysis)

    val_analysis = [q for q in query_analysis if q["split"] == "val"]
    print(f"  Val: mean verbs={np.mean([q['n_verbs'] for q in val_analysis]):.2f}, "
          f"mean nouns={np.mean([q['n_nouns'] for q in val_analysis]):.2f}, "
          f"0-verb queries={sum(1 for q in val_analysis if q['n_verbs']==0)}")

    # Create masked features
    print("Creating masked .npz files...")
    n_created = 0
    n_skipped = 0
    n_verb_tokens_zeroed = 0
    n_noun_tokens_zeroed = 0

    for item in all_data:
        qid = item["qid"]
        query = item["query"]
        npz_name = f"qid{qid}.npz"
        orig_path = os.path.join(ORIG_TEXT_DIR, npz_name)

        if not os.path.exists(orig_path):
            n_skipped += 1
            continue

        orig = np.load(orig_path)
        lhs = orig["last_hidden_state"].copy()
        po = orig["pooler_output"].copy()
        num_tokens = lhs.shape[0]

        # spaCy POS tags (on original casing)
        doc = nlp(query)
        spacy_words = [t.text.lower() for t in doc]
        spacy_pos = [t.pos_ for t in doc]

        # CLIP token alignment
        words, word_to_indices = get_word_to_token_indices(tokenizer, query)

        # Map spaCy POS to CLIP word indices
        # spaCy and str.split() may differ, so do fuzzy matching
        word_pos_map = {}
        si = 0
        for wi, word in enumerate(words):
            if si < len(spacy_words):
                # Try to match
                if word == spacy_words[si] or word in spacy_words[si] or spacy_words[si] in word:
                    word_pos_map[wi] = spacy_pos[si]
                    si += 1
                else:
                    # Try next spacy token
                    word_pos_map[wi] = "X"
                    # Check if spacy has a multi-token word
                    if si + 1 < len(spacy_words) and spacy_words[si + 1] in word:
                        si += 2
                    else:
                        si += 1
            else:
                word_pos_map[wi] = "X"

        # Verb-masked
        verb_lhs = lhs.copy()
        for wi, pos in word_pos_map.items():
            if pos == "VERB" and wi in word_to_indices:
                for ti in word_to_indices[wi]:
                    if ti < num_tokens:
                        verb_lhs[ti] = 0.0
                        n_verb_tokens_zeroed += 1

        # Noun-masked
        noun_lhs = lhs.copy()
        for wi, pos in word_pos_map.items():
            if pos in ("NOUN", "PROPN") and wi in word_to_indices:
                for ti in word_to_indices[wi]:
                    if ti < num_tokens:
                        noun_lhs[ti] = 0.0
                        n_noun_tokens_zeroed += 1

        np.savez(os.path.join(VERB_MASKED_DIR, npz_name),
                 last_hidden_state=verb_lhs.astype(np.float16),
                 pooler_output=po)
        np.savez(os.path.join(NOUN_MASKED_DIR, npz_name),
                 last_hidden_state=noun_lhs.astype(np.float16),
                 pooler_output=po)

        n_created += 1
        if n_created % 2000 == 0:
            print(f"    {n_created} queries processed...")

    print(f"  Done: {n_created} created, {n_skipped} skipped")
    print(f"  Verb tokens zeroed: {n_verb_tokens_zeroed}")
    print(f"  Noun tokens zeroed: {n_noun_tokens_zeroed}")

    summary = {
        "features_created": n_created,
        "features_skipped": n_skipped,
        "verb_tokens_zeroed": n_verb_tokens_zeroed,
        "noun_tokens_zeroed": n_noun_tokens_zeroed,
        "verb_masked_dir": VERB_MASKED_DIR,
        "noun_masked_dir": NOUN_MASKED_DIR,
        "mean_verbs_val": round(np.mean([q["n_verbs"] for q in val_analysis]), 2),
        "mean_nouns_val": round(np.mean([q["n_nouns"] for q in val_analysis]), 2),
    }
    with open(os.path.join(OUT_DIR, "ablation_prep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {os.path.join(OUT_DIR, 'ablation_prep_summary.json')}")


if __name__ == "__main__":
    main()
