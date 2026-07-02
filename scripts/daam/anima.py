from __future__ import annotations

import math
from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .trace import COND, HEATMAP_LATENT_SIZE, HeatMap

__all__ = ['AnimaDaamTracer', 'AnimaPromptAnalyzer']


class AnimaPromptAnalyzer:
    """Maps attention words to their positions in the anima cross-attention context.

    anima (Cosmos-Predict2) conditions its DiT cross-attention on the output of
    ``Qwen3_06B.preprocess_text_embeds`` -> ``LLMAdapter``. That adapter embeds the
    **T5 token ids** as queries and cross-attends to the Qwen hidden states, so the
    cross-attention *key* positions correspond one-to-one to the T5 tokens of the
    prompt (padded to 512). Word->position lookups therefore use the T5 tokenizer,
    mirroring ``utils.PromptAnalyzer`` (which uses the CLIP tokenizer for SD/SDXL).
    """

    def __init__(self, engine, text: str):
        # engine is a backend.text_processing.anima_engine.AnimaTextProcessingEngine
        self.engine = engine

        chunks = engine.tokenize_line(text)
        # AnimaTextProcessingEngine always produces a single chunk.
        self.tokens: List[int] = list(chain.from_iterable(chunk.t5_tokens for chunk in chunks))

        # Number of real (non-padding) cross-attention key positions.
        self.context_size = len(self.tokens)
        # Mirrors utils.PromptAnalyzer.token_count for the shared log line.
        self.token_count = len(self.tokens)

    def create(self, text: str) -> 'AnimaPromptAnalyzer':
        return AnimaPromptAnalyzer(self.engine, text)

    def encode(self, text: str) -> List[int]:
        return self.engine.t5_tokenizer([text], truncation=False, add_special_tokens=False)["input_ids"][0]

    def _needle_candidates(self, word: str) -> List[List[int]]:
        # T5 uses a case-sensitive SentencePiece tokenizer where a word mid-prompt
        # carries a leading-space marker and tokenizes differently from the bare
        # word. Try several surface forms so casual lower-case input still matches.
        seen = set()
        candidates: List[List[int]] = []
        for surface in (word, " " + word, word.lower(), " " + word.lower()):
            ids = self.encode(surface)
            key = tuple(ids)
            if ids and key not in seen:
                seen.add(key)
                candidates.append(ids)
        return candidates

    def calc_word_indecies(self, word: str, limit: int = -1, start_pos: int = 0):
        merge_idxs: List[int] = []

        tokens = self.tokens
        candidates = self._needle_candidates(word)
        if not candidates:
            return merge_idxs, 0

        limit_count = 0
        current_pos = 0
        for i, token in enumerate(tokens):
            current_pos = i
            if i < start_pos:
                continue

            for needles in candidates:
                if needles[0] != token:
                    continue
                nxt = i + 1
                success = True
                for needle in needles[1:]:
                    if nxt >= len(tokens) or needle != tokens[nxt]:
                        success = False
                        break
                    nxt += 1
                if success:
                    merge_idxs.extend(list(range(i, nxt)))
                    if limit > 0:
                        limit_count += 1
                        if limit_count >= limit:
                            return merge_idxs, current_pos
                    break

        return merge_idxs, current_pos


class AnimaDaamTracer:
    """Captures DAAM cross-attention heat maps on an anima (Cosmos-Predict2) DiT.

    The classic ``set_model_attn2_replace`` ModelPatcher hook does not fire for
    anima: its ``SelfCrossAttention`` modules call ``attention_function`` directly
    instead of consulting the patch dict (unlike SD/SDXL ``SpatialTransformer``).
    Instead we register a ``forward_pre_hook`` on every ``block.cross_attn`` module,
    recompute the conditional softmax attention from its real inputs, and aggregate
    the maps. The model's own forward is untouched, so the image is unchanged.
    """

    def __init__(self, diffusion_model, height: int, width: int, context_size: int):
        self.diffusion_model = diffusion_model
        self.height = height
        self.width = width
        # Real (non-padding) token count of the conditional prompt; bounds the
        # token dimension we keep so we don't carry 512 padded rows around.
        self.context_size = max(1, context_size)

        # block_key -> img_idx -> running sum tensor (tokens, S, S)
        self.maps: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)
        # block_key -> img_idx -> number of accumulated captures
        self.counts: Dict[str, Dict[int, int]] = defaultdict(dict)

        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------ hooks

    def hook(self):
        self.remove()
        blocks = getattr(self.diffusion_model, "blocks", None)
        if blocks is None:
            return
        for i, block in enumerate(blocks):
            cross_attn = getattr(block, "cross_attn", None)
            if cross_attn is None:
                continue
            handle = cross_attn.register_forward_pre_hook(
                self._make_pre_hook(f"B{i:02d}"), with_kwargs=True
            )
            self._handles.append(handle)

    def remove(self):
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []

    def reset(self):
        self.maps = defaultdict(dict)
        self.counts = defaultdict(dict)

    # --------------------------------------------------------------- capturing

    def _make_pre_hook(self, block_key: str):
        tracer = self

        def pre_hook(module, args, kwargs):
            # SelfCrossAttention.forward(x, context, rope_emb=..., transformer_options=...)
            try:
                x = args[0]
                context = args[1] if len(args) > 1 else kwargs.get("context")
                if context is None:
                    return None
                transformer_options = kwargs.get("transformer_options")
                if transformer_options is None and len(args) > 3:
                    transformer_options = args[3]
                tracer._capture(module, x, context, transformer_options or {}, block_key)
            except Exception:
                # Visualization must never break generation.
                pass
            return None  # do not modify the real forward inputs

        return pre_hook

    @torch.no_grad()
    def _capture(self, module, x: torch.Tensor, context: torch.Tensor, transformer_options: dict, block_key: str):
        heads = module.n_heads
        dim_head = module.head_dim
        scale = dim_head ** -0.5

        b = x.shape[0]
        seq_q = x.shape[1]
        seq_k = context.shape[1]

        # Recompute q/k exactly as SelfCrossAttention.compute_qkv does for a
        # cross-attention block: project, split heads, RMS-norm. No RoPE is applied
        # to cross-attention (is_SelfAttn is False), so the maps need no rotary term.
        q = module.q_proj(x).reshape(b, seq_q, heads, dim_head)
        k = module.k_proj(context).reshape(b, seq_k, heads, dim_head)
        q = module.q_norm(q).permute(0, 2, 1, 3).float()  # (b, heads, seq_q, dim_head)
        k = module.k_norm(k).permute(0, 2, 1, 3).float()  # (b, heads, seq_k, dim_head)

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * scale
        sim = sim.softmax(dim=-1)
        attn = sim.mean(1)  # average over heads -> (b, seq_q, seq_k)

        # Keep only the real text tokens (drop the 512-padding tail).
        keep = min(self.context_size, seq_k)

        h, w = self._spatial_dims(seq_q)
        if h is None:
            return

        cond_or_uncond = transformer_options.get("cond_or_uncond", [COND])
        groups = max(1, len(cond_or_uncond))
        per = max(1, b // groups)

        for g, kind in enumerate(cond_or_uncond):
            if kind != COND:
                continue
            for j in range(per):
                row = g * per + j
                if row >= b:
                    break
                m = attn[row, :, :keep].permute(1, 0).reshape(keep, h, w)  # (tokens, h, w)
                m = F.interpolate(
                    m.unsqueeze(0), size=(HEATMAP_LATENT_SIZE, HEATMAP_LATENT_SIZE), mode='bicubic'
                ).squeeze(0).cpu()
                self._accumulate(block_key, j, m)

    def _spatial_dims(self, seq_q: int):
        # anima image tokens are (t h w) with t == 1 for text2image; the patch grid
        # keeps the latent aspect ratio (width / height).
        if self.height and self.width:
            aspect = self.width / self.height
            h = int(round(math.sqrt(seq_q / aspect)))
            h = max(1, h)
            w = seq_q // h
            if h * w == seq_q:
                return h, w

        side = int(round(math.sqrt(seq_q)))
        if side * side == seq_q:
            return side, side
        return None, None

    def _accumulate(self, block_key: str, img_idx: int, m: torch.Tensor):
        cur = self.maps[block_key].get(img_idx)
        if cur is None:
            self.maps[block_key][img_idx] = m.clone()
            self.counts[block_key][img_idx] = 1
        else:
            self.maps[block_key][img_idx] = cur + m
            self.counts[block_key][img_idx] += 1

    # ------------------------------------------------------------- aggregation

    @property
    def block_keys(self) -> List[str]:
        return sorted(self.maps.keys())

    def compute_global_heat_map(self, prompt_analyzer, prompt, batch_index, block_keys=None) -> Optional[HeatMap]:
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
