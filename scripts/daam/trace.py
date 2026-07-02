from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from backend.attention import attention_function

from .utils import PromptAnalyzer

__all__ = ['DaamTracer', 'HeatMap', 'trace']


# backend.sampling.sampling_function uses COND = 0 / UNCOND = 1
COND = 0

# Cross-attention maps from every UNet block are upsampled to this latent
# resolution before being aggregated across blocks and timesteps.
HEATMAP_LATENT_SIZE = 64


class HeatMap:
    def __init__(self, prompt_analyzer: PromptAnalyzer, prompt: str, heat_maps: torch.Tensor):
        # heat_maps shape: (tokens, H, W)
        self.prompt_analyzer = prompt_analyzer.create(prompt)
        self.heat_maps = heat_maps
        self.prompt = prompt

    def compute_word_heat_map(self, word: str, word_idx: int = None) -> Optional[torch.Tensor]:
        merge_idxs, _ = self.prompt_analyzer.calc_word_indecies(word)
        if len(merge_idxs) == 0:
            return None
        return self.heat_maps[merge_idxs].mean(0)


class DaamTracer:
    """Captures DAAM cross-attention heat maps on a Forge Neo UNet.

    Instead of monkey-patching ``ldm`` modules (as the original A1111 port did),
    this registers an ``attn2`` replacement patch through Forge's official
    ModelPatcher API. The patch computes the real attention output with the
    backend's ``attention_function`` (so the generated image is unchanged) and
    additionally records the softmax attention weights of the *conditional*
    pass for visualization.
    """

    def __init__(self, unet_patcher, height: int, width: int, context_size: int):
        self.unet_patcher = unet_patcher
        self.height = height
        self.width = width
        self.context_size = context_size

        # block_key -> img_idx -> running sum tensor (tokens, S, S)
        self.maps: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)
        # block_key -> img_idx -> number of accumulated captures
        self.counts: Dict[str, Dict[int, int]] = defaultdict(dict)

    # ------------------------------------------------------------------ hooks

    def hook(self):
        diffusion_model = self.unet_patcher.model.diffusion_model
        patch = self._make_patch()

        n_input = len(diffusion_model.input_blocks)
        n_output = len(diffusion_model.output_blocks)

        # Registering at (block_name, number) (transformer_index=None) applies the
        # patch to every transformer block inside that SpatialTransformer.
        for i in range(n_input):
            self.unet_patcher.set_model_attn2_replace(patch, "input", i)
        self.unet_patcher.set_model_attn2_replace(patch, "middle", 0)
        for i in range(n_output):
            self.unet_patcher.set_model_attn2_replace(patch, "output", i)

    def reset(self):
        self.maps = defaultdict(dict)
        self.counts = defaultdict(dict)

    # --------------------------------------------------------------- patching

    def _make_patch(self):
        tracer = self

        def patch(q, k, v, extra_options):
            # Real output: keep generation identical to a non-DAAM run.
            out = attention_function(q, k, v, extra_options["n_heads"], mask=None)
            try:
                tracer._capture(q, k, extra_options)
            except Exception:
                # Visualization must never break generation.
                pass
            return out

        return patch

    @torch.no_grad()
    def _capture(self, q: torch.Tensor, k: torch.Tensor, extra_options: dict):
        seq_k = k.shape[1]
        # Only the conditional prompt's context has this token length.
        if seq_k != self.context_size:
            return

        heads = extra_options["n_heads"]
        b = q.shape[0]
        seq_q = q.shape[1]
        dim_head = q.shape[-1] // heads
        scale = dim_head ** -0.5

        # (b, heads, seq, dim_head)
        qr = q.reshape(b, seq_q, heads, dim_head).permute(0, 2, 1, 3).float()
        kr = k.reshape(b, seq_k, heads, dim_head).permute(0, 2, 1, 3).float()

        sim = torch.einsum('b h i d, b h j d -> b h i j', qr, kr) * scale
        sim = sim.softmax(dim=-1)
        attn = sim.mean(1)  # average over heads -> (b, seq_q, seq_k)

        h, w = self._spatial_dims(seq_q, extra_options.get("original_shape"))
        if h is None:
            return

        block_key = self._block_caption(extra_options.get("block", ("", 0)))

        cond_or_uncond = extra_options.get("cond_or_uncond", [COND])
        groups = max(1, len(cond_or_uncond))
        per = max(1, b // groups)

        for g, kind in enumerate(cond_or_uncond):
            if kind != COND:
                continue
            for j in range(per):
                row = g * per + j
                if row >= b:
                    break
                m = attn[row].permute(1, 0).reshape(seq_k, h, w)  # (tokens, h, w)
                m = F.interpolate(
                    m.unsqueeze(0), size=(HEATMAP_LATENT_SIZE, HEATMAP_LATENT_SIZE), mode='bicubic'
                ).squeeze(0).cpu()
                self._accumulate(block_key, j, m)

    def _spatial_dims(self, seq_q: int, original_shape):
        if original_shape is not None and len(original_shape) >= 4:
            lh, lw = original_shape[2], original_shape[3]
        else:
            side = int(round(math.sqrt(seq_q)))
            lh = lw = side

        h = int(round(math.sqrt(seq_q * lh / lw)))
        h = max(1, h)
        w = seq_q // h
        if h * w != seq_q:
            side = int(round(math.sqrt(seq_q)))
            if side * side == seq_q:
                return side, side
            return None, None
        return h, w

    def _accumulate(self, block_key: str, img_idx: int, m: torch.Tensor):
        cur = self.maps[block_key].get(img_idx)
        if cur is None:
            self.maps[block_key][img_idx] = m.clone()
            self.counts[block_key][img_idx] = 1
        else:
            self.maps[block_key][img_idx] = cur + m
            self.counts[block_key][img_idx] += 1

    @staticmethod
    def _block_caption(block) -> str:
        name, num = block[0], block[1]
        if name == "input":
            return f"IN{num:02d}"
        if name == "middle":
            return "MID"
        if name == "output":
            return f"OUT{num:02d}"
        return f"{name}{num}"

    # ------------------------------------------------------------- aggregation

    @property
    def block_keys(self) -> List[str]:
        return sorted(self.maps.keys())

    def compute_global_heat_map(self, prompt_analyzer, prompt, batch_index, block_keys=None) -> Optional[HeatMap]:
        """Aggregate captured maps for a single image across the given blocks.

        ``block_keys=None`` combines every block (the default DAAM behaviour).
        Pass a single-element list to get a per-layer heat map.
        """
        if block_keys is None:
            block_keys = list(self.maps.keys())

        total = None
        used = 0
        for bk in block_keys:
            m = self.maps.get(bk, {}).get(batch_index)
            if m is None:
                continue
            count = max(1, self.counts.get(bk, {}).get(batch_index, 1))
            avg = m / count
            total = avg if total is None else total + avg
            used += 1

        if total is None or used == 0:
            return None

        heat = total / used  # (tokens, S, S)
        return HeatMap(prompt_analyzer, prompt, heat)


# Backwards-compatible alias with the original ``trace(...)`` factory.
trace = DaamTracer
