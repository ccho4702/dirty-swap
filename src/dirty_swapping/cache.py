"""Equal-length KV replacement for Transformers DynamicCache-style layers.

This module does not regenerate text or update any downstream cache position.
The caller owns candidate generation and must ensure both caches use the same
prefix, absolute positions, model, and attention implementation.
"""

from __future__ import annotations

from typing import Any

import torch


def swap_kv_segment(main_cache: Any, alternative_cache: Any, start: int, length: int) -> None:
    """Overwrite [start, start + length) in every key and value tensor in place.

    Expected tensor layout is [batch, kv_heads, sequence, head_dim]. Only
    full-attention DynamicCache layers are supported. Fail closed for other
    cache types rather than silently applying an incorrect partial edit.
    """
    if start < 0 or length <= 0:
        raise ValueError("start must be nonnegative and length must be positive")
    main_layers = getattr(main_cache, "layers", None)
    alternative_layers = getattr(alternative_cache, "layers", None)
    if not main_layers or not alternative_layers or len(main_layers) != len(alternative_layers):
        raise ValueError("both caches must have the same nonzero layer count")

    end = start + length
    for index, (main, alternative) in enumerate(zip(main_layers, alternative_layers)):
        if any(
            getattr(layer, "is_sliding", False) or getattr(layer, "is_croppable", True) is False
            for layer in (main, alternative)
        ):
            raise ValueError(f"layer {index}: sliding or static cache is unsupported")
        for name in ("keys", "values"):
            destination = getattr(main, name, None)
            source = getattr(alternative, name, None)
            if destination is None or source is None or destination.ndim != 4 or source.ndim != 4:
                raise ValueError(f"layer {index}: unsupported {name} tensor")
            if (
                destination.shape[0:2] != source.shape[0:2]
                or destination.shape[3] != source.shape[3]
            ):
                raise ValueError(f"layer {index}: incompatible {name} shape")
            if destination.dtype != source.dtype or destination.device != source.device:
                raise ValueError(f"layer {index}: incompatible {name} dtype or device")
            if end > destination.shape[2] or end > source.shape[2]:
                raise ValueError(f"layer {index}: replacement exceeds {name} sequence length")

    # Validate all layers before any write, so invalid input never leaves a partial swap.
    with torch.inference_mode():
        for main, alternative in zip(main_layers, alternative_layers):
            for name in ("keys", "values"):
                destination = getattr(main, name)
                source = getattr(alternative, name)
                destination[:, :, start:end, :].copy_(source[:, :, start:end, :])
