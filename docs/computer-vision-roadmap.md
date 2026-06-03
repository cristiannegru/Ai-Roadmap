# 👁️ Computer Vision Roadmap

Computer Vision (CV) enables computers to see, identify, and process images and videos in the same way that human vision does. This guide maps out the journey from basic image manipulation to state-of-the-art object detection and generative vision.

---

## 🗺️ Computer Vision Journey
```
[OpenCV & Image Prep] ──> [Object Detection: YOLO] ──> [Segmentation & Generative Vision]
          │                             │                              │
          ├─ Filtering & Thresholds     ├─ YOLOv8 Custom Training      ├─ U-Net & SAM
          └─ Edge Detection (Canny)     └─ Multi-Object Tracking       └─ Diffusion & GANs
```

---

## 📌 Phase 5A: Computer Vision Specialization

### 1. Classical Image Processing & OpenCV
Learn the fundamentals of pixel manipulation, colorspaces (RGB/HSV), image filtering, thresholds, morphological operations, and edge detection.
- 📺 **OpenCV Complete Course (freeCodeCamp)**: [Watch Video](https://youtu.be/oXlwWbU8l2o)
- 📺 **OpenCV Python Tutorial for Beginners**: [Watch Playlist](https://www.youtube.com/watch?v=oXlwWbU8l2o)

### 2. Object Detection & YOLO
Object detection goes beyond classification by finding the coordinates of specific objects. YOLO (You Only Look Once) is the industry standard for real-time object detection.
- **Topics**: Intersection over Union (IoU), Non-Max Suppression, Bounding Boxes, YOLOv8 custom dataset training.
- 📺 **YOLOv8 Crash Course (freeCodeCamp)**: [Watch Video](https://youtu.be/m9fH9OWn8YM)
- 📺 **YOLOv8 Custom Object Detection Tutorial**: [Watch Video](https://www.youtube.com/watch?v=m9fH9OWn8YM)

### 3. Image Segmentation
Pixel-level classification where every pixel is assigned to an object category.
- **Topics**: U-Net, Mask R-CNN, Segment Anything Model (SAM) by Meta.
- 📺 **U-Net Image Segmentation Tutorial**: [Watch Video](https://www.youtube.com/watch?v=ZoOuNv8TXLs)
- 📺 **Segment Anything Model (SAM) Guide**: [Watch Video](https://www.youtube.com/watch?v=D-D6ZmadzPE)

### 4. Generative Vision (GANs & Diffusion)
Creating new visual content from noise or textual prompts.
- **Topics**: Generative Adversarial Networks (GANs), Diffusion Models, Stable Diffusion.
- 📺 **GANs Explained (PyTorch implementation)**: [Watch Video](https://youtu.be/OljTVUVzPpM)
- 📺 **How Diffusion Models Work**: [Watch Video](https://youtu.be/yTAMrHVG1ew)

---

## 💻 OpenCV Code Sample: Real-Time Edge Detection
```python
import cv2

# Capture video stream from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Run Canny Edge Detector
    edges = cv2.Canny(gray, 100, 200)
    
    # Display the result
    cv2.imshow('Canny Edges', edges)
    
    # Stop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🛠️ Computer Vision Milestones & Projects
1. **Virtual Paintbrush using OpenCV (Color Masking)**
   - 📺 **Tutorial**: [Watch Video](https://www.youtube.com/watch?v=oXlwWbU8l2o)
2. **PPE Detection on Construction Site (YOLOv8 Custom Training)**
   - 📺 **Tutorial**: [Watch Video](https://www.youtube.com/watch?v=m9fH9OWn8YM)
3. **Automatic Number Plate Recognition (ANPR)**
   - 📺 **Tutorial**: [Watch Video](https://www.youtube.com/watch?v=m9fH9OWn8YM)
