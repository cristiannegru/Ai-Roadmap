# 04 — Vision Detector (Intermediate)

Classical detection with NumPy + PIL: threshold → connected components →
bounding boxes. No OpenCV, no weights download.

```bash
cd projects/04_vision_detector
pip install -r requirements.txt
python main.py --show
```

Detects the 3 synthetic objects (2 plates + 1 disc) and saves an annotated
`outputs/detection.png`.

## Go production

`pip install ultralytics`, load `YOLO("yolov8n.pt")`, and replace
`detect_boxes()` — CLI outputs (boxes + PNG) stay the same shape.
