#!/usr/bin/env python3
"""
VMR Project: GPU Verification Script
Run this on an SCC GPU node to confirm everything works.
Usage: python scripts/verify_gpu.py
"""

import sys
import os

def check(label, condition, detail=""):
    status = "✓" if condition else "✗"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition

def main():
    print("=" * 50)
    print("VMR Environment Verification")
    print("=" * 50)
    all_ok = True

    # Python version
    v = sys.version_info
    all_ok &= check("Python version", v.major == 3 and v.minor >= 10,
                     f"{v.major}.{v.minor}.{v.micro}")

    # PyTorch
    try:
        import torch
        all_ok &= check("PyTorch imported", True, torch.__version__)
        all_ok &= check("CUDA available", torch.cuda.is_available(),
                         f"device count: {torch.cuda.device_count()}" if torch.cuda.is_available() else "NO GPU FOUND")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                check(f"  GPU {i}", True, f"{name}, {mem:.1f} GB")

            # Quick tensor test
            x = torch.randn(100, 100, device="cuda")
            y = torch.mm(x, x.T)
            check("CUDA tensor ops", True, f"matmul result shape: {y.shape}")
    except ImportError:
        all_ok &= check("PyTorch imported", False, "NOT INSTALLED")

    # Transformers
    try:
        import transformers
        all_ok &= check("transformers", True, transformers.__version__)
    except ImportError:
        all_ok &= check("transformers", False, "NOT INSTALLED")

    # einops
    try:
        import einops
        all_ok &= check("einops", True, einops.__version__)
    except ImportError:
        all_ok &= check("einops", False, "NOT INSTALLED")

    # spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("The person adds salt to the pan")
        verbs = [t.text for t in doc if t.pos_ == "VERB"]
        nouns = [t.text for t in doc if t.pos_ == "NOUN"]
        all_ok &= check("spaCy + en_core_web_sm", True,
                         f"verbs={verbs}, nouns={nouns}")
    except Exception as e:
        all_ok &= check("spaCy", False, str(e))

    # scipy, pandas, matplotlib
    for pkg_name in ["scipy", "pandas", "matplotlib", "seaborn", "h5py", "tensorboard", "sklearn"]:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "ok")
            all_ok &= check(pkg_name, True, ver)
        except ImportError:
            all_ok &= check(pkg_name, False, "NOT INSTALLED")

    # Check Lighthouse
    try:
        import lighthouse
        all_ok &= check("Lighthouse", True)
    except ImportError:
        all_ok &= check("Lighthouse", False, "run: cd lighthouse && pip install -e .")

    print()
    if all_ok:
        print("All checks PASSED ✓")
        print("You are ready to start training!")
    else:
        print("Some checks FAILED ✗")
        print("Fix the issues above before proceeding.")

    print("=" * 50)
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())