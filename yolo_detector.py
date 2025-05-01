import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO("yolov8n.pt")
cat_class_name = "cat"

track_points = []

def draw_boxes_and_tracking(frame, boxes):
    global track_points
    annotator = Annotator(frame, line_width=2)

    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name != cat_class_name:
            continue

        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)

        annotator.box_label(box=xyxy, label=class_name, color=(255, 0, 0))

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if track_points:
            prev_cx, prev_cy = track_points[-1]
            smoothed = ((cx + prev_cx) // 2, (cy + prev_cy) // 2)
            track_points.append(smoothed)
        else:
            track_points.append((cx, cy))

        if len(track_points) > 60:
            track_points.pop(0)

    recent_points = track_points[-30:]
    for i in range(1, len(recent_points)):
        cv2.line(frame, recent_points[i - 1], recent_points[i], (200, 100, 100), 2)

    if track_points:
        cv2.circle(frame, track_points[-1], 4, (255, 0, 0), -1)

    watermark = "Poompakorn-Clicknext-Internship-2025"
    (text_width, _), _ = cv2.getTextSize(watermark, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(
        frame,
        watermark,
        (frame.shape[1] - text_width - 20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    return annotator.result()


def detect_object(frame):
    results = model(frame)
    for result in results:
        frame = draw_boxes_and_tracking(frame, result.boxes)
    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture("/Users/poompakornkamphangphet/Desktop/ClickNext Internship/CatZoomies.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = detect_object(frame)

        cv2.imshow("Detection", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
