from __future__ import annotations

from itertools import chain
from typing import List, Optional

import math

import torch
import torch.nn.functional as F

from comfy_kitchen import apply_rope

from .anima import AnimaDaamTracer, AnimaPromptAnalyzer
from .trace import COND, HEATMAP_LATENT_SIZE

__all__ = [
    'FluxDaamTracer', 'FluxPromptAnalyzer',
    'LuminaDaamTracer', 'LuminaPromptAnalyzer',
]


class FluxPromptAnalyzer(AnimaPromptAnalyzer):
    """Maps attention words to T5-XXL token positions for Flux / Chroma.

    The Flux txt stream is conditioned on the T5-XXL encoding of the prompt
    (``T5TextProcessingEngine``), so key positions inside the joint attention
    correspond one-to-one to the T5 tokens (prompt + EOS, padded to 256).
    Reuses AnimaPromptAnalyzer's SentencePiece word-matching logic.
    """

    def __init__(self, engine, text: str):
        # engine is a backend.text_processing.t5_engine.T5TextProcessingEngine
        self.engine = engine

        chunks, _ = engine.tokenize_line(text)
        tokens: List[int] = list(chain.from_iterable(chunk.tokens for chunk in chunks))
        # tokenize_line pads each chunk to min_length with id_pad; drop that
        # tail so context_size counts only real tokens (incl. the EOS).
        id_pad = getattr(engine, "id_pad", 0)
        while tokens and tokens[-1] == id_pad:
            tokens.pop()

        self.tokens = tokens
        self.context_size = max(1, len(tokens))
        self.token_count = len(tokens)

    def create(self, text: str) -> 'FluxPromptAnalyzer':
        return FluxPromptAnalyzer(self.engine, text)

    def encode(self, text: str) -> List[int]:
        return self.engine.tokenize([text])[0]


class LuminaPromptAnalyzer(AnimaPromptAnalyzer):
    """Maps attention words to Gemma2 token positions for Lumina 2 / Neta Lumina.

    Lumina's cap (caption) tokens are the Gemma2 encoding of the *templated*
    prompt (``GemmaTextProcessingEngine`` prepends the system template and a
    BOS token), so the same template expansion is applied here before
    tokenizing to keep word positions aligned with the conditioning.
    """

    def __init__(self, engine, text: str):
        # engine is a backend.text_processing.gemma_engine.GemmaTextProcessingEngine
        self.engine = engine

        line = engine.process_template(text, False)
        chunks = engine.tokenize_line(line)
        tokens: List[int] = list(chain.from_iterable(chunk.tokens for chunk in chunks))

        self.tokens = tokens
        self.context_size = max(1, len(tokens))
        self.token_count = len(tokens)

    def create(self, text: str) -> 'LuminaPromptAnalyzer':
        return LuminaPromptAnalyzer(self.engine, text)

    def encode(self, text: str) -> List[int]:
        return self.engine.tokenize([text])[0]


class JointDiTTracerBase(AnimaDaamTracer):
    """Shared plumbing for joint-attention DiTs (Flux, Chroma, Lumina 2).

    Unlike anima's true cross-attention, these models concatenate text and
    image tokens into a single self-attention. The image->text heat map is the
    softmax attention of the image-query rows restricted to the text-key
    columns. The softmax is computed per head to keep the transient
    (seq_img x seq_total) matrix small.
    """

    def _attn_img_to_txt(self, q_img: torch.Tensor, k: torch.Tensor, keep: int,
                         key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # q_img: (b, heads, seq_img, d), k: (b, heads, seq_total, d)
        # returns (b, seq_img, keep) attention averaged over heads
        heads = q_img.shape[1]
        scale = q_img.shape[-1] ** -0.5
        total = None
        for hd in range(heads):
            sim = torch.einsum('b i d, b j d -> b i j', q_img[:, hd].float(), k[:, hd].float()) * scale
            if key_mask is not None:
                sim = sim.masked_fill(~key_mask[:, None, :], float('-inf'))
            sim = sim.softmax(dim=-1)[:, :, :keep]
            total = sim if total is None else total + sim
        return total / heads

    def _store(self, attn: torch.Tensor, keep: int, h: int, w: int,
               transformer_options: dict, block_key: str):
        # attn: (b, h*w, keep); split batch rows into cond/uncond groups and
        # accumulate only the conditional ones (mirrors AnimaDaamTracer).
        b = attn.shape[0]
        cond_or_uncond = (transformer_options or {}).get("cond_or_uncond", [COND])
        groups = max(1, len(cond_or_uncond))
        per = max(1, b // groups)

        for g, kind in enumerate(cond_or_uncond):
            if kind != COND:
                continue
            for j in range(per):
                row = g * per + j
                if row >= b:
                    break
                m = attn[row].permute(1, 0).reshape(keep, h, w)  # (tokens, h, w)
                m = F.interpolate(
                    m.unsqueeze(0), size=(HEATMAP_LATENT_SIZE, HEATMAP_LATENT_SIZE), mode='bicubic'
                ).squeeze(0).cpu()
                self._accumulate(block_key, j, m)

    def _patch_grid(self) -> tuple:
        # Latent is 1/8 of the pixel size and the DiT patchifies 2x2, so one
        # image token covers 16x16 pixels (rounded up by pad_to_patch_size).
        return math.ceil(self.height / 16), math.ceil(self.width / 16)


class FluxDaamTracer(JointDiTTracerBase):
    """Captures DAAM heat maps on Flux / Chroma ``DoubleStreamBlock``s.

    Registers a ``forward_pre_hook`` on every double block, recomputes q/k
    exactly as the block's forward does (modulation, qkv, QKNorm, RoPE) and
    extracts the image->text part of the joint attention. Single-stream
    blocks are ignored; the double blocks carry the word-level signal.
    """

    def hook(self):
        self.remove()
        blocks = getattr(self.diffusion_model, "double_blocks", None)
        if blocks is None:
            return
        for i, block in enumerate(blocks):
            handle = block.register_forward_pre_hook(
                self._make_pre_hook(f"D{i:02d}"), with_kwargs=True
            )
            self._handles.append(handle)

    def _make_pre_hook(self, block_key: str):
        tracer = self

        def pre_hook(module, args, kwargs):
            # DoubleStreamBlock.forward(img, txt, vec, pe, attn_mask=None,
            #   modulation_dims_img=None, modulation_dims_txt=None, transformer_options={})
            try:
                def get(name, pos):
                    if name in kwargs:
                        return kwargs[name]
                    return args[pos] if len(args) > pos else None

                img = get("img", 0)
                txt = get("txt", 1)
                vec = get("vec", 2)
                pe = get("pe", 3)
                if img is None or txt is None or vec is None:
                    return None
                tracer._capture(
                    module, img, txt, vec, pe,
                    get("modulation_dims_img", 5), get("modulation_dims_txt", 6),
                    get("transformer_options", 7) or {}, block_key,
                )
            except Exception:
                # Visualization must never break generation.
                pass
            return None  # do not modify the real forward inputs

        return pre_hook

    @torch.no_grad()
    def _capture(self, module, img, txt, vec, pe, modulation_dims_img, modulation_dims_txt,
                 transformer_options: dict, block_key: str):
        from backend.nn.flux import apply_mod

        if module.modulation:
            img_mod1, _ = module.img_mod(vec)
            txt_mod1, _ = module.txt_mod(vec)
        else:
            (img_mod1, _), (txt_mod1, _) = vec

        b, seq_img, _ = img.shape
        seq_txt = txt.shape[1]
        heads = module.num_heads

        # Recompute q/k exactly as DoubleStreamBlock.forward does.
        img_modulated = apply_mod(module.img_norm1(img), (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
        img_qkv = module.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(b, seq_img, 3, heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = module.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = apply_mod(module.txt_norm1(txt), (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)
        txt_qkv = module.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(b, seq_txt, 3, heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = module.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)  # (b, heads, seq_txt+seq_img, d)
        k = torch.cat((txt_k, img_k), dim=2)
        if pe is not None:
            q, k = apply_rope(q, k, pe)

        h, w = self._patch_grid()
        if h * w != seq_img:
            h, w = self._spatial_dims(seq_img)
            if h is None:
                return

        keep = min(self.context_size, seq_txt)
        attn = self._attn_img_to_txt(q[:, :, seq_txt:, :], k, keep)
        self._store(attn, keep, h, w, transformer_options, block_key)


class LuminaDaamTracer(JointDiTTracerBase):
    """Captures DAAM heat maps on a Lumina 2 (NextDiT) model.

    The main ``layers`` operate on ``cat(cap_tokens, image_tokens)``. A hook on
    the first ``context_refiner`` layer records the padded caption length each
    forward pass; hooks on every ``layers[i].attention`` (JointAttention) then
    recompute q/k (qkv split, RMS-norm, RoPE, GQA repeat) and extract the
    image->caption attention.
    """

    def __init__(self, diffusion_model, height: int, width: int, context_size: int):
        super().__init__(diffusion_model, height, width, context_size)
        self._cap_len: Optional[int] = None

    def hook(self):
        self.remove()
        layers = getattr(self.diffusion_model, "layers", None)
        refiner = getattr(self.diffusion_model, "context_refiner", None)
        if layers is None or refiner is None or len(refiner) == 0:
            return

        def record_cap_len(module, args, kwargs):
            try:
                self._cap_len = int(args[0].shape[1])
            except Exception:
                pass
            return None

        self._handles.append(refiner[0].register_forward_pre_hook(record_cap_len, with_kwargs=True))

        for i, layer in enumerate(layers):
            attention = getattr(layer, "attention", None)
            if attention is None:
                continue
            handle = attention.register_forward_pre_hook(
                self._make_pre_hook(f"L{i:02d}"), with_kwargs=True
            )
            self._handles.append(handle)

    def _make_pre_hook(self, block_key: str):
        tracer = self

        def pre_hook(module, args, kwargs):
            # JointAttention.forward(x, x_mask, freqs_cis, transformer_options={})
            try:
                x = args[0] if args else kwargs.get("x")
                x_mask = args[1] if len(args) > 1 else kwargs.get("x_mask")
                freqs_cis = args[2] if len(args) > 2 else kwargs.get("freqs_cis")
                if x is None or freqs_cis is None:
                    return None
                tracer._capture(module, x, x_mask, freqs_cis,
                                kwargs.get("transformer_options") or {}, block_key)
            except Exception:
                # Visualization must never break generation.
                pass
            return None

        return pre_hook

    @torch.no_grad()
    def _capture(self, module, x, x_mask, freqs_cis, transformer_options: dict, block_key: str):
        cap_len = self._cap_len
        if not cap_len:
            return
        bsz, seqlen, _ = x.shape
        if seqlen <= cap_len:
            return  # a context_refiner-style call, not the joint sequence

        # Recompute q/k exactly as JointAttention.forward does.
        xq, xk, _xv = torch.split(
            module.qkv(x),
            [
                module.n_local_heads * module.head_dim,
                module.n_local_kv_heads * module.head_dim,
                module.n_local_kv_heads * module.head_dim,
            ],
            dim=-1,
        )
        xq = module.q_norm(xq.view(bsz, seqlen, module.n_local_heads, module.head_dim))
        xk = module.k_norm(xk.view(bsz, seqlen, module.n_local_kv_heads, module.head_dim))
        xq, xk = apply_rope(xq, xk, freqs_cis)

        n_rep = module.n_local_heads // module.n_local_kv_heads
        if n_rep >= 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        q = xq.movedim(1, 2)  # (b, heads, seq, d)
        k = xk.movedim(1, 2)

        h, w = self._patch_grid()
        seq_img = seqlen - cap_len
        if seq_img < h * w:  # image tokens may be padded past h*w, never short
            h, w = self._spatial_dims(seq_img)
            if h is None:
                return

        key_mask = None
        if isinstance(x_mask, torch.Tensor) and x_mask.dtype == torch.bool and x_mask.ndim == 2:
            key_mask = x_mask

        keep = min(self.context_size, cap_len)
        q_img = q[:, :, cap_len:cap_len + h * w, :]
        attn = self._attn_img_to_txt(q_img, k, keep, key_mask=key_mask)
        self._store(attn, keep, h, w, transformer_options, block_key)
