"""
Phase 3: Visual Bus — Episodic Working Memory.

Renders agent history (observations + actions) into a PIL image,
then runs GLM-OCR (0.9B) to compress that image back into text.

The key insight: instead of appending thousands of raw text tokens to the
LLM context each turn, we render history to a compact image and let OCR
extract the semantically meaningful parts. The LLM always receives a
fixed-size compressed summary, not a growing transcript.

Flow per turn:
  history list → render_history() → image on disk
               → compress()       → GLM-OCR reads image
                                  → compressed text returned
"""
from __future__ import annotations

import os
import textwrap
import time

from configs.settings import (
    OCR_MODEL_NAME,
    VISUAL_BUS_HISTORY_DIR,
    VISUAL_BUS_IMAGE_WIDTH,
    VISUAL_BUS_FONT_SIZE,
    MAX_VISUAL_TILES,
)

# Colours for the rendered history image
_COLOUR_BG = (18, 18, 24)          # dark background
_COLOUR_OBS = (100, 180, 255)      # blue — environment observations
_COLOUR_ACT = (255, 110, 110)      # red  — agent actions
_COLOUR_TURN = (120, 120, 140)     # grey — turn label


def _find_monospace_font(size: int):
    """Return a PIL ImageFont. Tries common monospace paths, falls back to default."""
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:/Windows/Fonts/consola.ttf",
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

    try:
        bbox = font.getbbox("M")
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]
    except AttributeError:
        char_w, char_h = font.getsize("M")

    padding = 12
    line_spacing = int(char_h * 1.4)
    usable_width = image_width - 2 * padding
    chars_per_line = max(20, usable_width // max(1, char_w))

    recent = history[-max_turns * 2:] if len(history) > max_turns * 2 else history

    rendered_lines: list[tuple[str, tuple[int, int, int]]] = []
    turn_num = max(0, len(history) // 2 - len(recent) // 2)

    for msg in recent:
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

    img_height = max(200, padding * 2 + len(rendered_lines) * line_spacing + padding)
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

    Renders agent history to an image each turn, then uses GLM-OCR (0.9B)
    to extract a compact text summary. The reasoning LLM receives this
    compressed summary instead of the raw growing transcript.

    GLM-OCR uses standard HuggingFace APIs (AutoProcessor +
    AutoModelForImageTextToText) with no trust_remote_code — no compatibility
    patches required.
    """

    _ocr_unavailable: bool = False  # class-level: stop retrying after first failure

    def __init__(self, history_dir: str = VISUAL_BUS_HISTORY_DIR):
        self.history_dir = history_dir
        self._ocr_model = None
        self._ocr_processor = None
        self._task_id: str = "task"
        self._turn: int = 0
        os.makedirs(history_dir, exist_ok=True)

    def reset(self, task_id: str = "task"):
        self._task_id = task_id.replace(" ", "_")[:40]
        self._turn = 0

    # ------------------------------------------------------------------
    # OCR model — lazy-loaded once, shared across all compress() calls
    # ------------------------------------------------------------------

    def _load_ocr(self) -> bool:
        """Load GLM-OCR. Returns False if unavailable (OOM, missing deps, etc.)."""
        if self._ocr_model is not None:
            return True
        if VisualBus._ocr_unavailable:
            return False

        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText

            print(f"[VisualBus] Loading OCR model ({OCR_MODEL_NAME}) …", flush=True)
            t0 = time.time()
            self._ocr_processor = AutoProcessor.from_pretrained(OCR_MODEL_NAME)
            self._ocr_model = AutoModelForImageTextToText.from_pretrained(
                OCR_MODEL_NAME,
                torch_dtype="auto",
                device_map="auto",
            )
            self._ocr_model.eval()
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

    def _run_ocr(self, image_path: str) -> str:
        """Run GLM-OCR on a rendered history image. Returns extracted text."""
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ]

        inputs = self._ocr_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._ocr_model.device)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            generated_ids = self._ocr_model.generate(**inputs, max_new_tokens=8192)

        output_text = self._ocr_processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )
        return output_text.strip()

    def _text_fallback(self, history: list[dict]) -> str:
        """
        Fallback when OCR is unavailable: return a plain-text summary of the
        most recent MAX_VISUAL_TILES turns. Grows slowly — for debugging only,
        not the real Visual Bus.
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
        Render history to an image and compress via GLM-OCR.

        Falls back to truncated plain text if OCR is unavailable.
        In either case the agent continues running.

        Returns:
            Compressed text summary of the visual history.
            Empty string if history is empty.
        """
        if not history:
            return ""

        self._turn += 1

        if not self._load_ocr():
            return self._text_fallback(history)

        img_name = f"{self._task_id}_t{self._turn:03d}"
        img_path = os.path.join(self.history_dir, f"{img_name}.png")

        # Step 1: render history to image
        t0 = time.time()
        render_history(history, img_path)
        render_ms = (time.time() - t0) * 1000

        # Step 2: run GLM-OCR on the image
        t1 = time.time()
        try:
            compressed = self._run_ocr(img_path)
        except Exception as e:
            print(f"[VisualBus] WARNING: OCR inference failed ({e}). Using text fallback.", flush=True)
            return self._text_fallback(history)
        ocr_ms = (time.time() - t1) * 1000

        print(
            f"[VisualBus] render={render_ms:.0f}ms  ocr={ocr_ms:.0f}ms  "
            f"compressed={len(compressed)} chars",
            flush=True,
        )
        return compressed
