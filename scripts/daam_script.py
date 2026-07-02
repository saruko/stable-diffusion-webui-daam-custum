from __future__ import annotations

import os

import gradio as gr
import torch
from PIL import Image

import modules.images as images
import modules.scripts as scripts
import modules.shared as shared
from modules import script_callbacks
from modules.processing import StableDiffusionProcessing, fix_seed
from modules.shared import opts

from scripts.daam import trace, utils
from scripts.daam.anima import AnimaDaamTracer, AnimaPromptAnalyzer

before_image_saved_handler = None


def _get_text_engine(sd_model):
    """Return the ClassicTextProcessingEngine for the loaded model.

    SD1.x exposes ``text_processing_engine``; SDXL exposes
    ``text_processing_engine_l`` (CLIP-L), which uses the same token positions
    as the conditioning context.
    """
    for attr in ("text_processing_engine", "text_processing_engine_l"):
        engine = getattr(sd_model, attr, None)
        if engine is not None:
            return engine
    return None


def _get_sd_unet(sd_model):
    """Return the SD/SDXL-style UNet (with input_blocks/output_blocks) or None.

    DAAM's classic cross-attention hook fits the SD U-Net used by SD1.x and SDXL.
    Other DiT-based models (Flux, etc.) expose a different ``diffusion_model``
    without these blocks; anima is handled separately by :func:`_get_anima_dit`.
    """
    try:
        diffusion_model = sd_model.forge_objects.unet.model.diffusion_model
    except AttributeError:
        return None
    if hasattr(diffusion_model, "input_blocks") and hasattr(diffusion_model, "output_blocks"):
        return diffusion_model
    return None


def _get_anima_dit(sd_model):
    """Return the anima (Cosmos-Predict2) DiT ``diffusion_model`` or None.

    anima has no classic ``attn2`` patch point; its ``Block`` modules each hold a
    ``cross_attn`` (SelfCrossAttention) that conditions on the Qwen/T5 text
    embeddings. DAAM hooks those modules directly (see ``AnimaDaamTracer``).
    """
    try:
        diffusion_model = sd_model.forge_objects.unet.model.diffusion_model
    except AttributeError:
        return None
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks and hasattr(blocks[0], "cross_attn"):
        return diffusion_model
    return None


class Script(scripts.Script):

    GRID_LAYOUT_AUTO = "Auto"
    GRID_LAYOUT_PREVENT_EMPTY = "Prevent Empty Spot"
    GRID_LAYOUT_BATCH_LENGTH_AS_ROW = "Batch Length As Row"

    def title(self):
        return "Daam script"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Attention Heatmap", open=False):
                attention_texts = gr.Text(label='Attention texts for visualization. (comma separated)', value='')

                with gr.Row():
                    hide_images = gr.Checkbox(label='Hide heatmap images', value=False)
                    dont_save_images = gr.Checkbox(label='Do not save heatmap images', value=False)
                    hide_caption = gr.Checkbox(label='Hide caption', value=False)

                with gr.Row():
                    use_grid = gr.Checkbox(label='Use grid (output to grid dir)', value=False)
                    grid_layouyt = gr.Dropdown(
                        [Script.GRID_LAYOUT_AUTO, Script.GRID_LAYOUT_PREVENT_EMPTY, Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW],
                        label="Grid layout",
                        value=Script.GRID_LAYOUT_AUTO,
                    )

                with gr.Row():
                    alpha = gr.Slider(label='Heatmap blend alpha', value=0.5, minimum=0, maximum=1, step=0.01)
                    heatmap_image_scale = gr.Slider(label='Heatmap image scale', value=1.0, minimum=0.1, maximum=1, step=0.025)

                with gr.Row():
                    trace_each_layers = gr.Checkbox(label='Trace each layers', value=False)
                    layers_as_row = gr.Checkbox(label='Use layers as row instead of Batch Length', value=False)

        self.tracer = None

        return [attention_texts, hide_images, dont_save_images, hide_caption, use_grid, grid_layouyt, alpha, heatmap_image_scale, trace_each_layers, layers_as_row]

    def process(self,
                p: StableDiffusionProcessing,
                attention_texts: str,
                hide_images: bool,
                dont_save_images: bool,
                hide_caption: bool,
                use_grid: bool,
                grid_layouyt: str,
                alpha: float,
                heatmap_image_scale: float,
                trace_each_layers: bool,
                layers_as_row: bool):

        self.enabled = False  # in case we bail out early

        # Clear any handler left dangling by a previous run that errored out
        # before postprocess, so it can never reference a stale tracer.
        global before_image_saved_handler
        before_image_saved_handler = None

        self.attentions = [s.strip() for s in attention_texts.split(",") if s.strip()]
        if len(self.attentions) == 0:
            return

        if not opts.samples_save:
            print("DAAM: 'Always save all generated images' is disabled; heatmaps cannot be produced. Skipping.")
            return

        self.hide_images = hide_images
        self.dont_save_images = dont_save_images
        self.hide_caption = hide_caption
        self.alpha = alpha
        self.use_grid = use_grid
        self.grid_layouyt = grid_layouyt
        self.heatmap_image_scale = heatmap_image_scale
        self.trace_each_layers = trace_each_layers

        self.images = []
        self.heatmap_images = dict()
        self.attn_captions = []

        self.tracer = None
        self.prompt_analyzer = None
        self.model_kind = None
        self.context_size = 77
        # Global image positions (iteration * batch_size + batch_index) already
        # turned into heatmaps, so auxiliary/grid saves don't double-process.
        self._seen = set()

        self.enabled = True
        fix_seed(p)

    def process_batch(self,
                      p: StableDiffusionProcessing,
                      attention_texts: str,
                      hide_images: bool,
                      dont_save_images: bool,
                      hide_caption: bool,
                      use_grid: bool,
                      grid_layouyt: str,
                      alpha: float,
                      heatmap_image_scale: float,
                      trace_each_layers: bool,
                      layers_as_row: bool,
                      prompts,
                      **kwargs):

        if not getattr(self, "enabled", False):
            return

        clip_engine = _get_text_engine(p.sd_model)
        anima_engine = getattr(p.sd_model, "text_processing_engine_anima", None)

        if clip_engine is not None and _get_sd_unet(p.sd_model) is not None:
            self.model_kind = "clip"
            engine = clip_engine
        elif anima_engine is not None and _get_anima_dit(p.sd_model) is not None:
            self.model_kind = "anima"
            engine = anima_engine
        else:
            print(f"DAAM: unsupported model '{type(p.sd_model).__name__}'. "
                  f"Supported: SD1.x / SDXL (classic CLIP U-Net) and anima (Cosmos-Predict2 DiT). Skipping.")
            self.enabled = False
            return

        styled_prompt = prompts[0]
        try:
            if self.model_kind == "anima":
                self.prompt_analyzer = AnimaPromptAnalyzer(engine, styled_prompt)
            else:
                self.prompt_analyzer = utils.PromptAnalyzer(engine, styled_prompt)
        except Exception as e:
            print(f"DAAM: failed to analyze prompt ({e}). Skipping.")
            self.enabled = False
            return

        self.context_size = self.prompt_analyzer.context_size

        print(f"DAAM: context_size={self.context_size}, token_count={self.prompt_analyzer.token_count}, attentions={self.attentions}")

        global before_image_saved_handler
        before_image_saved_handler = lambda params: self.before_image_saved(params)

    def process_before_every_sampling(self, p: StableDiffusionProcessing, *args, **kwargs):
        if not getattr(self, "enabled", False) or self.prompt_analyzer is None:
            return

        if self.model_kind == "anima":
            # anima's SelfCrossAttention does not honor the attn2 patch dict, so we
            # hook the cross_attn modules on the live diffusion_model directly. The
            # hooks are removed in postprocess.
            diffusion_model = _get_anima_dit(p.sd_model)
            if diffusion_model is None:
                return
            tracer = AnimaDaamTracer(
                diffusion_model=diffusion_model,
                height=p.height,
                width=p.width,
                context_size=self.context_size,
            )
            tracer.reset()
            tracer.hook()
            self.tracer = tracer
            return

        unet = p.sd_model.forge_objects.unet.clone()

        tracer = trace(
            unet_patcher=unet,
            height=p.height,
            width=p.width,
            context_size=self.context_size,
        )
        tracer.reset()
        tracer.hook()

        p.sd_model.forge_objects.unet = unet
        self.tracer = tracer

    def postprocess(self, p, processed,
                    attention_texts: str,
                    hide_images: bool,
                    dont_save_images: bool,
                    hide_caption: bool,
                    use_grid: bool,
                    grid_layouyt: str,
                    alpha: float,
                    heatmap_image_scale: float,
                    trace_each_layers: bool,
                    layers_as_row: bool,
                    **kwargs):
        if not getattr(self, "enabled", False):
            return

        # The classic patched UNet lives only on a clone that Forge discards each
        # run, so there is nothing to detach there. anima hooks the live model, so
        # its forward hooks must be removed explicitly.
        if self.tracer is not None and hasattr(self.tracer, "remove"):
            self.tracer.remove()
        self.tracer = None

        global before_image_saved_handler
        before_image_saved_handler = None

        self.images += processed.images

        if not self.heatmap_images:
            return processed

        if layers_as_row:
            images_list = []
            for i in range(p.batch_size * p.n_iter):
                imgs = []
                for k in sorted(self.heatmap_images.keys()):
                    imgs += [self.heatmap_images[k][len(self.attentions) * i + j] for j in range(len(self.attentions))]
                images_list.append(imgs)
        else:
            images_list = [self.heatmap_images[k] for k in sorted(self.heatmap_images.keys())]

        for img_list in images_list:
            if img_list and self.use_grid:
                grid_layout = self.grid_layouyt
                if grid_layout == Script.GRID_LAYOUT_AUTO:
                    if p.batch_size * p.n_iter == 1:
                        grid_layout = Script.GRID_LAYOUT_PREVENT_EMPTY
                    else:
                        grid_layout = Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW

                if grid_layout == Script.GRID_LAYOUT_PREVENT_EMPTY:
                    grid_img = images.image_grid(img_list)
                elif grid_layout == Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW:
                    if layers_as_row:
                        batch_size = len(self.attentions)
                        rows = len(self.heatmap_images)
                    else:
                        batch_size = p.batch_size
                        rows = p.batch_size * p.n_iter
                    grid_img = images.image_grid(img_list, batch_size=batch_size, rows=rows)
                else:
                    continue

                if not self.dont_save_images:
                    images.save_image(grid_img, p.outpath_grids, "grid_daam", grid=True, p=p)

                if not self.hide_images:
                    processed.images.insert(0, grid_img)
                    processed.index_of_first_image += 1
                    processed.infotexts.insert(0, processed.infotexts[0])
            else:
                if not self.hide_images:
                    processed.images[:0] = img_list
                    processed.index_of_first_image += len(img_list)
                    processed.infotexts[:0] = [processed.infotexts[0]] * len(img_list)

        return processed

    # Filename markers for non-primary saves (face restoration, color correction,
    # masks, grids) that must not be turned into heatmaps.
    _SKIP_SUFFIXES = ("-before-face-restoration", "-before-color-correction", "-mask", "-mask-composite")

    def before_image_saved(self, params: script_callbacks.ImageSaveParams):
        if not getattr(self, "enabled", False) or self.tracer is None or len(self.attentions) == 0:
            return

        # Skip auxiliary saves; only the primary sample image should be processed.
        basename = os.path.basename(params.filename or "")
        if any(marker in basename for marker in self._SKIP_SUFFIXES):
            return

        batch_size = max(1, getattr(params.p, "batch_size", 1))
        batch_pos = int(getattr(params.p, "batch_index", 0))
        if batch_pos < 0 or batch_pos >= batch_size:
            return

        iteration = int(getattr(params.p, "iteration", 0))
        global_pos = iteration * batch_size + batch_pos
        if global_pos in self._seen:
            return
        self._seen.add(global_pos)

        styled_prompt = shared.prompt_styles.apply_styles_to_prompt(params.p.prompt, params.p.styles)

        if self.trace_each_layers:
            block_keys = self.tracer.block_keys
            layers = [[bk] for bk in block_keys]
            captions = list(block_keys)
        else:
            layers = [None]
            captions = [""]

        self.attn_captions = captions

        with torch.no_grad():
            for i, selected in enumerate(layers):
                if i not in self.heatmap_images:
                    self.heatmap_images[i] = []

                global_heat_map = self.tracer.compute_global_heat_map(
                    self.prompt_analyzer, styled_prompt, batch_pos, block_keys=selected
                )
                if global_heat_map is None:
                    print(f"DAAM: no attention captured for image {batch_pos}"
                          + (f" / {captions[i]}" if captions[i] else ""))

                heatmap_images = []
                for attention in self.attentions:
                    img_size = params.image.size

                    caption = None
                    if not self.hide_caption:
                        caption = attention + ((" " + captions[i]) if captions[i] else "")

                    heat_map = global_heat_map.compute_word_heat_map(attention) if global_heat_map is not None else None
                    if heat_map is None:
                        print(f"DAAM: no heatmap for '{attention}'")

                    heat_map_img = utils.expand_image(heat_map, img_size[1], img_size[0]) if heat_map is not None else None
                    img = utils.image_overlay_heat_map(
                        params.image, heat_map_img, alpha=self.alpha, caption=caption, image_scale=self.heatmap_image_scale
                    )

                    fullfn_without_extension, extension = os.path.splitext(params.filename)
                    suffix = "_" + attention + (("_" + captions[i]) if captions[i] else "")
                    full_filename = fullfn_without_extension + suffix + extension

                    heatmap_images.append(img)
                    if not self.use_grid and not self.dont_save_images:
                        img.save(full_filename)

                self.heatmap_images[i] += heatmap_images


def handle_before_image_saved(params: script_callbacks.ImageSaveParams):
    if before_image_saved_handler is not None and callable(before_image_saved_handler):
        before_image_saved_handler(params)


script_callbacks.on_before_image_saved(handle_before_image_saved)
