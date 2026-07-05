"""Optional real-model convenience wrappers (behind the ``[demo]`` extra).

The library's contract is that a ``model`` is just a callable ``image -> logits`` — no
framework lock-in. ``timm_classifier`` is a *convenience* helper that wraps a frozen,
eval-mode ``timm`` ImageNet classifier (with its correct preprocessing) into exactly that
callable, so the demos can point the finished instrument at real models. torch/timm are
imported lazily inside the function, so ``import psyvis_ml`` still works on the core deps
alone; only *calling* this needs the ``[demo]`` extra.
"""

from __future__ import annotations

import numpy as np

__all__ = ["timm_classifier"]


def timm_classifier(model_name, *, pretrained=True, device=None, input_clip=(0.0, 1.0)):
    """Wrap a ``timm`` ImageNet classifier as a callable ``image -> logits`` (numpy).

    The returned callable accepts a single image as an ``(H, W, 3)`` or ``(H, W)`` array in
    ``[0, 1]`` (whatever the stimulus pipeline produced), clips it to a valid pixel range,
    resizes to the model's input size, applies the model's own mean/std normalization, and
    returns the 1000-way logits as a 1-D numpy array. The model is frozen and in eval mode.

    Parameters
    ----------
    model_name
        Any ``timm`` model name, e.g. ``"resnet18"``, ``"resnet50"``, ``"vit_small_patch16_224"``.
    pretrained
        Load pretrained ImageNet weights (required for meaningful curves).
    device
        Torch device string; defaults to CUDA when available, else CPU.
    input_clip
        ``(lo, hi)`` pixel clip applied before normalization (stimulus manipulations such as
        contrast scaling can push values out of range; real models need valid pixels). Pass
        ``None`` to disable — but note the realized manipulation level is then the clipped one.
    """
    try:
        import timm
        import torch
        import torch.nn.functional as F
        from timm.data import resolve_data_config
    except ImportError as e:  # pragma: no cover - exercised only without the demo extra
        raise ImportError(
            "timm_classifier needs torch + timm; install the demo extra: "
            'pip install -e ".[demo]"'
        ) from e

    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval()
    model.requires_grad_(False)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cfg = resolve_data_config({}, model=model)
    mean = np.asarray(cfg["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(cfg["std"], dtype=np.float32).reshape(1, 3, 1, 1)
    _, in_h, in_w = cfg["input_size"]
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)

    def _to_batched_chw(img):
        a = np.asarray(img, dtype=np.float32)
        if a.ndim == 2:  # grayscale (e.g. a distractor canvas) -> 3 channels
            a = np.repeat(a[:, :, None], 3, axis=2)
        if a.ndim == 3 and a.shape[2] == 1:
            a = np.repeat(a, 3, axis=2)
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(f"expected an (H, W) or (H, W, 3) image; got shape {a.shape}.")
        if input_clip is not None:
            a = np.clip(a, input_clip[0], input_clip[1])
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        return t.to(device)

    @torch.no_grad()
    def classify(img):
        t = _to_batched_chw(img)
        if t.shape[-2:] != (in_h, in_w):
            t = F.interpolate(t, size=(in_h, in_w), mode="bilinear", align_corners=False)
        t = (t - mean_t) / std_t
        logits = model(t)
        return logits.squeeze(0).detach().cpu().numpy()

    classify.__name__ = f"timm:{model_name}"
    return classify
