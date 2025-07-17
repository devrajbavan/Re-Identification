from ultralytics import YOLO
import cv2

def detect_players(video_path, model_path, player_class_ids=[0]):
    """
    Detect players in a video using a YOLO model.

    Args:
        video_path (str): Path to the input video.
        model_path (str): Path to the YOLO model weights.
        player_class_ids (list): List of class IDs considered as players (default [0] for COCO person).
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Process results: filter player class only
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            xyxy = box.xyxy[0].tolist()

            if cls_id in player_class_ids:
                # Draw bounding box and label on frame
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Player {cls_id} {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

        # Show annotated frame
        cv2.imshow("Player Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Testing with default YOLOv8n model for pipeline verification
    detect_players("/home/devraj-bavan/Documents/assignment/Re-Id Verification/data/videos/15sec_input_720p.mp4", "yolov8n.pt")

