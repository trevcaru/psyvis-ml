"""Test C pre-flight: resolve the ~10 timm models and report the clean top-1 SPREAD first.

The range-restriction check (per the brief): Imagenette is 10 classes and these models are near
ceiling. If the clean-accuracy spread is under ~3 points, the x-axis is too compressed for the
correlation test to mean anything — stop and redesign before burning GPU on the full sweep.

REAL DATA ONLY. Every accuracy is a real forward pass of a real pretrained model on the real 3925
Imagenette val images. A model that fails to resolve/download is DROPPED and reported — never
substituted with a placeholder.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

import psyvis_ml as pe

DATA_DIR = r"C:/Users/glopu/OneDrive/Desktop/Experiments/Exp_assets/imagenette2-320/val"
EXPECTED_N = 3925

# (timm name, architecture family) — span family and capacity for range on both axes.
MODELS = [
    ("resnet18", "CNN"),
    ("resnet50", "CNN"),
    ("resnet101", "CNN"),
    ("efficientnet_b0", "CNN"),
    ("convnext_tiny", "CNN"),
    ("vit_small_patch16_224", "ViT"),
    ("vit_base_patch16_224", "ViT"),
    ("deit_small_patch16_224", "ViT"),
    ("swin_tiny_patch4_window7_224", "hybrid"),
    ("mobilenetv3_large_100", "CNN"),
]


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    import timm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"device: {device} ({dev_name}) · torch {torch.__version__} · timm {timm.__version__}",
          flush=True)

    # Load the real val set ONCE (full, no split), same path/count as Test A.
    t0 = time.time()
    ds = pe.datasets.imagenet_subset(DATA_DIR, max_per_class=None, image_size=224)
    n = len(ds.images)
    print(f"loaded {n} real images in {time.time() - t0:.1f}s", flush=True)
    if n != EXPECTED_N:
        raise SystemExit(f"FATAL: image count {n} != Test A's {EXPECTED_N}")
    chance = 1.0 / ds.num_classes

    results = []
    dropped = []
    for name, fam in MODELS:
        print(f"\n[{name}] ({fam}) resolving + loading pretrained weights ...", flush=True)
        try:
            clf = pe.models.timm_classifier(name, pretrained=True)
        except Exception as e:  # noqa: BLE001 — real failure: drop and report, never substitute
            print(f"  DROP {name}: {type(e).__name__}: {e}", flush=True)
            dropped.append((name, f"{type(e).__name__}: {e}"))
            continue
        try:
            params_m = sum(p.numel() for p in
                           timm.create_model(name, pretrained=False).parameters()) / 1e6
        except Exception:  # noqa: BLE001
            params_m = float("nan")

        # Clean top-1 on the FULL real val set (the undegraded 'accuracy' axis + pretrained gate).
        t0 = time.time()
        correct = 0
        for j in range(n):
            logits = np.asarray(clf(ds.images[j]), dtype=float).ravel()
            if int(np.argmax(logits)) == int(ds.labels[j]):
                correct += 1
            if (j + 1) % 1000 == 0:
                print(f"  ... {j + 1}/{n}", flush=True)
        acc = correct / n
        dt = time.time() - t0
        gate = "PASS" if acc >= 0.5 else "FAIL"
        print(f"  clean top-1 = {acc:.4f}  params={params_m:.1f}M  ({dt:.1f}s)  "
              f"pretrained-gate[{gate}] (chance={chance:.4f})", flush=True)
        if acc < 0.5:
            raise SystemExit(f"FATAL: {name} clean top-1 {acc:.4f} not far above chance "
                             f"{chance:.4f} — weights look like random init, not pretrained.")
        results.append({"name": name, "family": fam, "params_m": params_m, "clean_acc": acc,
                        "seconds": dt})

    # ---- spread verdict ---- #
    accs = [r["clean_acc"] for r in results]
    print("\n" + "=" * 70, flush=True)
    print("CLEAN TOP-1 SPREAD (the x-axis range-restriction gate)", flush=True)
    print("=" * 70, flush=True)
    for r in sorted(results, key=lambda x: -x["clean_acc"]):
        print(f"  {r['name']:32} {r['family']:6} {r['clean_acc'] * 100:6.2f}%  "
              f"{r['params_m']:6.1f}M", flush=True)
    if dropped:
        print("\n  DROPPED (unavailable, not substituted):", flush=True)
        for nm, why in dropped:
            print(f"    {nm}: {why}", flush=True)
    if len(accs) >= 2:
        spread_pts = (max(accs) - min(accs)) * 100
        print(f"\n  spread = {spread_pts:.2f} points  "
              f"(min {min(accs) * 100:.2f}%  max {max(accs) * 100:.2f}%  "
              f"n_models={len(accs)})", flush=True)
        if spread_pts < 3.0:
            print("\n  >>> STOP: spread < 3 points. Range restriction is FATAL for a "
                  "correlation on clean accuracy. Redesign the x-axis (harder dataset, or "
                  "degraded-condition accuracy at a fixed level) before the full sweep.",
                  flush=True)
        else:
            print("\n  >>> spread >= 3 points: clean accuracy has enough range to proceed "
                  "(still report the caveat).", flush=True)


if __name__ == "__main__":
    main()
