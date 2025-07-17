import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def main(video_path, output_path):
    """
    Player Re-Identification with YOLO detection and DeepSORT tracking.

    Args:
        video_path (str): Path to input video.
        output_path (str): Path to save the output tracked video.
    """
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')

    # Initialize DeepSORT tracker
    tracker = DeepSort(max_age=30)  # adjust max_age for re-entry tracking duration

    # Initialize video capture and writer
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Prepare detections for DeepSORT
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())

            if cls == 0:  # assuming class 0 is player
                bbox = [x1, y1, x2 - x1, y2 - y1]  # convert to [x, y, w, h]
                detections.append((bbox, conf, 'player'))

        # Update tracker with current frame detections
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw tracked objects
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}",
                        (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

        # Display frame
        cv2.imshow("Player Re-ID Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example execution
    main(video_path="/home/devraj-bavan/Documents/assignment/Re-Id Verification/data/videos/15sec_input_720p.mp4",
         output_path="/home/devraj-bavan/Documents/assignment/Re-Id Verification/results/output_tracked.mp4")

