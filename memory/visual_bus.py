"""
Phase 3: Visual Bus — Episodic Working Memory.

Renders agent history (observations + actions) into a PIL image,
then runs DeepSeek-OCR-2 to compress that image back into text.

The key insight: instead of appending thousands of raw text tokens to the
LLM context each turn, we render history to a compact image and let OCR
extract the semantically meaningful parts. The LLM always receives a
fixed-size compressed summary, not a growing transcript.

Flow per turn:
  history list → render_history() → image on disk
               → compress()       → DeepSeek-OCR-2 reads image
                                  → compressed markdown text returned
"""
from __future__ import annotations

import os
import textwrap
import time
from pathlib import Path
from typing import Optional

from configs.settings import (
    OCR_MODEL_NAME,
    VISUAL_BUS_HISTORY_DIR,
    VISUAL_BUS_BASE_SIZE,
    VISUAL_BUS_IMAGE_SIZE,
    VISUAL_BUS_IMAGE_WIDTH,
    VISUAL_BUS_FONT_SIZE,
    MAX_VISUAL_TILES,
)

# Colours for the rendered history image
_COLOUR_BG = (18, 18, 24)          # dark background
_COLOUR_OBS = (100, 180, 255)      # blue — environment observations
_COLOUR_ACT = (255, 110, 110)      # red  — agent actions
_COLOUR_TURN = (120, 120, 140)     # grey — turn label
_COLOUR_WHITE = (230, 230, 235)    # fallback text


def _find_monospace_font(size: int):
    """Return a PIL ImageFont. Tries common monospace paths, falls back to default."""
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",                    # macOS
        "C:/Windows/Fonts/consola.ttf",                       # Windows
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_history(
    history: list[dict],
    image_path: str,
    *,
    max_turns: int = MAX_VISUAL_TILES,
    image_width: int = VISUAL_BUS_IMAGE_WIDTH,
    font_size: int = VISUAL_BUS_FONT_SIZE,
) -> str:
    """
    Render conversation history to a colour-coded PIL image.

    Args:
        history: list of {"role": "user"|"assistant", "content": "..."}
        image_path: where to save the PNG
        max_turns: keep the most recent N turns (prevents unbounded height)
        image_width: pixel width of the output image
        font_size: monospace font size in pt

    Returns:
        image_path (for chaining)
    """
    from PIL import Image, ImageDraw

    font = _find_monospace_font(font_size)

    # Approximate character width/height for the chosen font
    try:
        # PIL >= 10
        bbox = font.getbbox("M")
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]
    except AttributeError:
        char_w, char_h = font.getsize("M")

    padding = 12
    line_spacing = int(char_h * 1.4)
    usable_width = image_width - 2 * padding
    chars_per_line = max(20, usable_width // max(1, char_w))

    # Truncate to last max_turns entries (each entry = one obs or action)
    recent = history[-max_turns * 2:] if len(history) > max_turns * 2 else history

    # Pre-compute wrapped lines
    rendered_lines: list[tuple[str, tuple[int, int, int]]] = []
    turn_num = max(0, len(history) // 2 - len(recent) // 2)

    for i, msg in enumerate(recent):
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            label = f"── Turn {turn_num} | OBS "
            colour = _COLOUR_OBS
        else:
            label = f"── Turn {turn_num} | ACT "
            colour = _COLOUR_ACT
            turn_num += 1

        rendered_lines.append((label + "─" * max(0, chars_per_line - len(label)), _COLOUR_TURN))

        wrapped = textwrap.wrap(content, width=chars_per_line) or ["(empty)"]
        for line in wrapped:
            rendered_lines.append((line, colour))

    # Calculate image height
    total_lines = len(rendered_lines)
    img_height = max(200, padding * 2 + total_lines * line_spacing + padding)

    img = Image.new("RGB", (image_width, img_height), _COLOUR_BG)
    draw = ImageDraw.Draw(img)

    y = padding
    for text, colour in rendered_lines:
        draw.text((padding, y), text, font=font, fill=colour)
        y += line_spacing

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    img.save(image_path, format="PNG")
    return image_path


class VisualBus:
    """
    Episodic working memory via visual compression.

    Renders agent history to an image each turn, then uses DeepSeek-OCR-2
    to extract a compact text summary. The reasoning LLM receives this
    compressed summary instead of the raw growing transcript.
    """

    def __init__(self, history_dir: str = VISUAL_BUS_HISTORY_DIR):
        self.history_dir = history_dir
        self._ocr_model = None
        self._ocr_tokenizer = None
        self._task_id: str = "task"
        self._turn: int = 0
        os.makedirs(history_dir, exist_ok=True)

    def reset(self, task_id: str = "task"):
        self._task_id = task_id.replace(" ", "_")[:40]
        self._turn = 0

    # ------------------------------------------------------------------
    # OCR model — lazy-loaded once, shared across all compress() calls
    # ------------------------------------------------------------------
    _ocr_unavailable: bool = False  # class-level flag: stop retrying after first failure

    @staticmethod
    def _patch_llama_flash_attn():
        """
        Stub ``LlamaFlashAttention2`` for transformers versions that removed it.

        DeepSeek-OCR-2's remote code imports ``LlamaFlashAttention2`` at module
        level, but only uses it when ``_attn_implementation='flash_attention_2'``.
        We load with ``'eager'`` so the stub is imported but never called.

        Root cause: vLLM ≥0.15 needs transformers ≥4.50 (for Gemma3Config),
        but transformers ≥4.48 removed LlamaFlashAttention2. Monkey-patching
        bridges the gap without pinning transformers to an older version.
        """
        try:
            import transformers.models.llama.modeling_llama as llama_mod
            if not hasattr(llama_mod, "LlamaFlashAttention2"):
                # Use SdpaAttention as a stand-in; it's never called because
                # we always load with _attn_implementation='eager'.
                fallback_cls = getattr(
                    llama_mod, "LlamaSdpaAttention",
                    getattr(llama_mod, "LlamaAttention", object),
                )
                llama_mod.LlamaFlashAttention2 = fallback_cls
                print(
                    "[VisualBus] Patched LlamaFlashAttention2 stub "
                    f"(using {fallback_cls.__name__})",
                    flush=True,
                )
        except Exception as patch_err:
            print(f"[VisualBus] WARNING: could not apply LlamaFlashAttention2 patch: {patch_err}", flush=True)

    def _load_ocr(self) -> bool:
        """Load DeepSeek-OCR-2. Returns False if unavailable (OOM, missing deps, etc.)."""
        if self._ocr_model is not None:
            return True
        if VisualBus._ocr_unavailable:
            return False

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            # Must patch before the model's remote code is imported
            self._patch_llama_flash_attn()

            print(f"[VisualBus] Loading OCR model ({OCR_MODEL_NAME}) …", flush=True)
            t0 = time.time()
            self._ocr_tokenizer = AutoTokenizer.from_pretrained(
                OCR_MODEL_NAME, trust_remote_code=True
            )
            # Use 'eager' so LlamaFlashAttention2 (stubbed above) is never instantiated
            self._ocr_model = AutoModel.from_pretrained(
                OCR_MODEL_NAME,
                _attn_implementation="eager",
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
            ).eval().cuda()
            print(f"[VisualBus] OCR model ready in {time.time() - t0:.1f}s", flush=True)
            return True

        except Exception as e:
            print(
                f"[VisualBus] WARNING: OCR model failed to load ({e}).\n"
                f"  Falling back to truncated text history for this run.",
                flush=True,
            )
            VisualBus._ocr_unavailable = True
            return False

    def _text_fallback(self, history: list[dict]) -> str:
        """
        Fallback when OCR is unavailable: return a plain-text summary of the
        most recent MAX_VISUAL_TILES turns. Much cheaper than full history but
        still grows slowly — useful for debugging, not the real Visual Bus.
        """
        recent = history[-(MAX_VISUAL_TILES * 2):]
        lines = []
        turn = max(0, len(history) // 2 - len(recent) // 2)
        for msg in recent:
            role = "OBS" if msg["role"] == "user" else "ACT"
            if role == "ACT":
                turn += 1
            lines.append(f"[Turn {turn} {role}] {msg['content'][:200]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compress(self, history: list[dict]) -> str:
        """
        Render history to an image and compress via DeepSeek-OCR-2.

        Falls back to truncated plain text if OCR is unavailable
        (e.g. transformers version mismatch). In either case the agent
        continues running — the fallback just won't flatten the token curve.

        Returns:
            Compressed text summary of the visual history.
            Empty string if history is empty.
        """
        if not history:
            return ""

        self._turn += 1

        # If OCR can't load, degrade gracefully rather than crashing
        if not self._load_ocr():
            return self._text_fallback(history)

        img_name = f"{self._task_id}_t{self._turn:03d}"
        img_path = os.path.join(self.history_dir, f"{img_name}.png")
        ocr_out_dir = os.path.join(self.history_dir, "ocr_output")
        os.makedirs(ocr_out_dir, exist_ok=True)

        # Step 1: render history to image
        t0 = time.time()
        render_history(history, img_path)
        render_ms = (time.time() - t0) * 1000

        # Step 2: run DeepSeek-OCR-2 on the image
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        t1 = time.time()
        try:
            self._ocr_model.infer(
                self._ocr_tokenizer,
                prompt=prompt,
                image_file=img_path,
                output_path=ocr_out_dir,
                base_size=VISUAL_BUS_BASE_SIZE,
                image_size=VISUAL_BUS_IMAGE_SIZE,
                crop_mode=True,
                save_results=True,
            )
        except Exception as e:
            print(f"[VisualBus] WARNING: OCR inference failed ({e}). Using text fallback.", flush=True)
            return self._text_fallback(history)
        ocr_ms = (time.time() - t1) * 1000

        # Step 3: read OCR output file
        result_file = os.path.join(ocr_out_dir, f"{img_name}.md")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                compressed = f.read().strip()
        else:
            print(f"[VisualBus] WARNING: OCR output not found at {result_file}. Using text fallback.", flush=True)
            return self._text_fallback(history)

        print(
            f"[VisualBus] render={render_ms:.0f}ms  ocr={ocr_ms:.0f}ms  "
            f"compressed={len(compressed)} chars",
            flush=True,
        )
        return compressed
