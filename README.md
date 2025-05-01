# Clicknext Internship – Technical Test (2025)

This project detects and tracks a cat in a video using YOLOv8 and OpenCV. The task was part of the Clicknext internship assessment.

## Features
- Detects only the **cat** class
- Draws a blue bounding box on the cat
- Adds a tracking line for movement
- Shows a watermark at the top-right
- Real-time only, no video saving

## Files
- `yolo_detector.py` – main script
- `yolov8n.pt` – YOLOv8 model weights
- `CatZoomies.mp4` – input video
- `requirements.txt` – list of Python packages

## How to Run

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the script:

    ```bash
    python yolo_detector.py
    ```

Make sure all files are in the same folder.

---

_Poompakorn Kamphangphet_  
Clicknext Internship Submission – 2025

