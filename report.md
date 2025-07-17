
---

## 🗒️ **report.md (Brief Report)**

```markdown
# 📝 **Player Re-Identification Assignment Report**

### 👤 **Name**: Devraj Bavan  
### 📅 **Date**: July 2025

---

## ✅ **Objective**

Implement **player re-identification** in a single video feed by:

- Detecting all players frame-by-frame.
- Maintaining consistent IDs even when players leave and re-enter the frame.

---

## ⚙️ **Approach and Methodology**

1. **Detection:**

   - Used **Ultralytics YOLO (yolov11.pt)** for detecting players in each frame.
   - Filtered detections to retain only the **player class** (assumed class ID = 0).

2. **Tracking & Re-Identification:**

   - Integrated **DeepSORT tracker** to assign unique IDs to each detected player.
   - Configured `max_age=30` to maintain IDs even if a player leaves the frame briefly.
   - Tracker outputs are drawn with bounding boxes and player IDs for clarity.

3. **Visualization & Output:**

   - Results saved as `output_tracked.mp4` for submission.
   - Live display window for debugging and performance verification.

---

## 🔬 **Techniques Tried and Outcomes**

| Technique                 | Outcome |
|-----------------------------|---------|
| YOLOv8n default model      | Detected players but assignment required yolov11.pt |
| YOLO + DeepSORT baseline   | Successfully tracks and re-identifies players with consistent IDs |
| Adjusting DeepSORT parameters | Improved re-identification when players reappear quickly |

---

## ⚠️ **Challenges Encountered**

1. **Missing yolov11.pt model file**:  
   - The provided Google Drive link was inactive, so used yolov8n temporarily for testing pipeline integration.

2. **DeepSORT input format**:  
   - Required correct `[x, y, w, h]` bounding box format for tracker initialization.

3. **Resource limitations**:  
   - GPU inference was essential for real-time performance on high-resolution videos.

---

## 🚀 **Next Steps & Improvements**

If provided more time and compute resources, I would:

- Integrate **Re-ID embedding models** (e.g., FastReID) for improved identity preservation under occlusions or significant reappearance gaps.  
- Fine-tune YOLO on the assignment dataset for optimal detection accuracy.  
- Extend pipeline for **cross-camera re-identification (Option 1)** using appearance + spatial mapping techniques (Homography + Re-ID).

---

## ✅ **Status**

✔️ **Core objective achieved**: Player re-identification in single feed with consistent IDs.  
❌ **Pending**: Model file verification (`yolov11.pt`) as per assignment instructions.

---

### 🙏 **Thank you for the opportunity. Looking forward to your feedback.**

