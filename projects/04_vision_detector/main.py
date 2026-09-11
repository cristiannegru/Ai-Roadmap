"""Project 04 — Vision detector (NumPy + PIL, zero heavy CV deps).

Category 2 (Intermediate) in docs/projects-roadmap.md.

Classical pipeline: grayscale → threshold → connected components (flood
fill) → bounding boxes → annotated PNG. Teaches what YOLO replaces with a
single learned pass.

Usage:
    python main.py                  # demo: generate → detect → save outputs/
    python main.py --show           # also print an ASCII preview

Production swap: replace detect_boxes() with Ultralytics YOLOv8
(`pip install ultralytics`, `YOLO("yolov8n.pt")`) — the boxes + annotated
image outputs keep the same shapes.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent

import numpy as np
from PIL import Image, ImageDraw

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 (x1/y1 exclusive)


def make_synthetic_image(size: int = 128, seed: int = 7) -> np.ndarray:
    """Dark background + 2 bright rectangles + 1 bright disc + noise."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)
    img[20:45, 15:55] = 220.0  # rect 1
    img[70:110, 75:115] = 200.0  # rect 2
    yy, xx = np.ogrid[:size, :size]
    img[(yy - 90) ** 2 + (xx - 30) ** 2 <= 12**2] = 210.0  # disc
    img += rng.normal(0, 8, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def detect_boxes(image: np.ndarray, *, threshold: int = 128, min_area: int = 50) -> list[Box]:
    """Threshold + 4-connected components → bounding boxes (area-filtered)."""
    if image.ndim != 2:
        raise ValueError(f"expected grayscale (H, W), got {image.shape}")
    mask = image > threshold
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    boxes: list[Box] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            queue, visited[y, x] = deque([(y, x)]), True
            xs, ys = [], []
            while queue:
                cy, cx = queue.popleft()
                xs.append(cx)
                ys.append(cy)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(xs) >= min_area:
                boxes.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    return sorted(boxes)


def draw_boxes(image: np.ndarray, boxes: list[Box]) -> Image.Image:
    """Return a PIL image with red boxes + labels drawn."""
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 0, 0), width=2)
        draw.text((x0 + 2, max(y0 - 10, 0)), f"obj{i}", fill=(255, 0, 0))
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vision detector demo (synthetic image)")
    parser.add_argument("--out", type=Path, default=HERE / "outputs" / "detection.png")
    parser.add_argument("--show", action="store_true", help="print ASCII preview")
    args = parser.parse_args(argv)

    img = make_synthetic_image()
    boxes = detect_boxes(img)
    print(f"detected {len(boxes)} objects:")
    for b in boxes:
        print(f"  box x0={b[0]} y0={b[1]} x1={b[2]} y1={b[3]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    draw_boxes(img, boxes).save(args.out)
    print(f"saved: {args.out}")
    if args.show:
        small = np.array(Image.fromarray(img).resize((32, 32))) > 128
        for row in small:
            print("".join("#" if v else "." for v in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
