from __future__ import annotations

import re
from itertools import chain
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont

from modules.paths_internal import roboto_ttf_file

__all__ = [
    'expand_image',
    'escape_prompt',
    'calc_context_size',
    'image_overlay_heat_map',
    'PromptAnalyzer',
]


def expand_image(im: torch.Tensor, h: int = 512, w: int = 512, absolute: bool = False, threshold: float = None) -> torch.Tensor:
    im = im.unsqueeze(0).unsqueeze(0)
    im = F.interpolate(im.float().detach(), size=(h, w), mode='bicubic')

    if not absolute:
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)

    if threshold:
        im = (im > threshold).float()

    return im.squeeze()


def _write_on_image(img: Image.Image, caption: str, font_size: int = 32) -> Image.Image:
    ix, iy = img.size
    draw = ImageDraw.Draw(img)
    margin = 2
    fontsize = font_size
    font = ImageFont.truetype(roboto_ttf_file, fontsize)
    text_height = iy - 60
    tx = draw.textbbox((0, 0), caption, font)
    draw.text((int((ix - tx[2]) / 2), text_height + margin), caption, (0, 0, 0), font=font)
    draw.text((int((ix - tx[2]) / 2), text_height - margin), caption, (0, 0, 0), font=font)
    draw.text((int((ix - tx[2]) / 2 + margin), text_height), caption, (0, 0, 0), font=font)
    draw.text((int((ix - tx[2]) / 2 - margin), text_height), caption, (0, 0, 0), font=font)
    draw.text((int((ix - tx[2]) / 2), text_height), caption, (255, 255, 255), font=font)
    return img


def _convert_heat_map_colors(heat_map: torch.Tensor) -> torch.Tensor:
    def get_color(value):
        return np.array(cm.turbo(value / 255)[0:3])

    color_map = np.array([get_color(i) * 255 for i in range(256)])
    color_map = torch.tensor(color_map, device=heat_map.device, dtype=torch.float32)

    heat_map = (heat_map * 255).long()

    return color_map[heat_map]


def image_overlay_heat_map(img, heat_map, word=None, out_file=None, crop=None, alpha=0.5, caption=None, image_scale=1.0):
    # type: (Image.Image, torch.Tensor, str, str, int, float, str, float) -> Image.Image
    assert img is not None

    if heat_map is not None:
        heat_map = _convert_heat_map_colors(heat_map)
        heat_map = heat_map.to('cpu').detach().numpy().copy().astype(np.uint8)
        heat_map_img = Image.fromarray(heat_map)
        img = Image.blend(img, heat_map_img, alpha)
    else:
        img = img.copy()

    if caption:
        img = _write_on_image(img, caption)

    if image_scale != 1.0:
        x, y = img.size
        size = (int(x * image_scale), int(y * image_scale))
        img = img.resize(size, Image.BICUBIC)

    return img


def calc_context_size(token_length: int) -> int:
    len_check = 0 if (token_length - 1) < 0 else token_length - 1
    return (int(len_check // 75) + 1) * 77


def escape_prompt(prompt):
    if isinstance(prompt, str):
        prompt = prompt.lower()
        prompt = re.sub(r"[\(\)\[\]]", "", prompt)
        prompt = re.sub(r":\d+\.*\d*", "", prompt)
        return prompt
    elif isinstance(prompt, list):
        return [escape_prompt(p) for p in prompt]
    return prompt


class PromptAnalyzer:
    """Tokenizes a prompt with Forge Neo's ClassicTextProcessingEngine and maps
    attention words to their token positions in the (chunked) conditioning context.

    `engine` is a ``backend.text_processing.classic_engine.ClassicTextProcessingEngine``
    instance, e.g. ``sd_model.text_processing_engine`` (SD1.x) or
    ``sd_model.text_processing_engine_l`` (SDXL).
    """

    def __init__(self, engine, text: str):
        self.engine = engine

        chunks, token_count = engine.tokenize_line(text)

        self.token_count = token_count
        self.fixes = list(chain.from_iterable(chunk.fixes for chunk in chunks))

        # Each chunk is [id_start] + 75 tokens + [id_end] == 77 entries, matching
        # the per-chunk layout of the cross-attention conditioning context.
        self.tokens: List[int] = list(chain.from_iterable(chunk.tokens for chunk in chunks))
        self.multipliers = list(chain.from_iterable(chunk.multipliers for chunk in chunks))

        # Token dimension of the conditioning context (number of 77-token chunks).
        self.context_size = len(self.tokens) if self.tokens else calc_context_size(token_count)

    def create(self, text: str) -> 'PromptAnalyzer':
        return PromptAnalyzer(self.engine, text)

    def encode(self, text: str) -> List[int]:
        return self.engine.tokenize([text])[0]

    def calc_word_indecies(self, word: str, limit: int = -1, start_pos: int = 0):
        word = word.lower()
        merge_idxs = []

        tokens = self.tokens
        needles = self.encode(word)

        if len(needles) == 0:
            return merge_idxs, 0

        limit_count = 0
        current_pos = 0
        for i, token in enumerate(tokens):
            current_pos = i
            if i < start_pos:
                continue

            if needles[0] == token and len(needles) > 1:
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
                            break

            elif needles[0] == token:
                merge_idxs.append(i)
                if limit > 0:
                    limit_count += 1
                    if limit_count >= limit:
                        break

        return merge_idxs, current_pos
