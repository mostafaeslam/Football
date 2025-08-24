# Football Task Notebook Documentation

## Cell-by-Cell Explanation

### Configuration Cell

```python
CONFIG = {
    "paths": {
        "video": "data/raw/CV_Task.mkv",
        "weights": "models/yolov8n.pt",
        "outputs_dir": "outputs",
        "out_video_name": ""  # auto-named with timestamp if empty
    },
    "yolo": {
        "model_type": "YOLOv8",
        "imgsz": 640,
        "conf_person": 0.25,
        "conf_ball": 0.20,
        "iou_nms": 0.45,
        "device": "auto"
    },
    "ball": {
        "prefer_lower_half": True,
        "min_rel_area": 5e-6,
        "max_rel_area": 8e-4
    },
    "teams": {
        "min_players_for_cluster": 4,
        "ema_alpha": 0.2  # exp. moving avg for team colors
    },
    "ocr": {
        "enabled": True,
        "engine": "easyocr",
        "roi": {
            "top": 0.15,
            "bottom": 0.60,
            "left": 0.15,
            "right": 0.85
        },
        "valid_len": [1, 2],  # valid number of digits in jersey number
        "sample_every_n_frames": 5,
        "min_bbox_h": 60,
        "keep_numeric_only": True
    },
    "class_map": {
        0: "player",
        1: "referee",
        2: "ball"
    },
    "tracking": {             # Tracking configuration
        "enabled": True,      # Enable object tracking
        "iou_match": 0.4,     # IOU threshold for matching tracks
        "max_age": 15         # Max frames to keep lost tracks
    },
    "runtime": {              # Runtime configuration
        "preview_frames": 150,  # Number of frames to process in preview mode
        "full_run": False,     # Set to True to process the entire video
        "process_every_n": 1    # Process every nth frame (for speed)
    },
    "viz": {
        "draw_thickness": 2,
        "font_scale": 0.6
    }
}
```

This cell defines the configuration dictionary (CONFIG) with parameters for all aspects of the video processing pipeline.

### Environment & Imports Cell

```python
# ==== ENVIRONMENT & IMPORTS ====
import os, sys, math, time, yaml, importlib, types
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

try:
    from ultralytics import YOLO
except Exception as e:
    print("Ultralytics not available:", e)

print("Imports loaded.")
```

This cell imports all necessary libraries for the project, including OpenCV for image processing, NumPy for numerical operations, Matplotlib for visualization, and YOLO for object detection.

### Path and Video Setup Cell

```python
# ==== SANITY CHECKS (auto-pick video: mp4/mkv/avi/mov) ====
from pathlib import Path

video_path = Path(CONFIG["paths"]["video"])
if not video_path.exists():
    raw_dir = Path("data/raw")
    # Auto-find first video in raw data dir
    videos = list(raw_dir.glob("*.mp4")) + list(raw_dir.glob("*.mkv")) + \
             list(raw_dir.glob("*.avi")) + list(raw_dir.glob("*.mov"))
    if videos:
        video_path = videos[0]
        CONFIG["paths"]["video"] = str(video_path)
        print(f"Using auto-found video: {video_path}")
    else:
        raise FileNotFoundError(f"No video found in paths or {raw_dir}")
        
print(f"Input video: {video_path}")
print(f"Using weights: {CONFIG['paths']['weights']}")
```

This cell verifies that the video file exists and automatically selects the first available video file if the specified path doesn't exist.

### Helper Functions Cell

```python
# ==== HELPER FUNCTIONS ====
def next_versioned_path(directory, prefix, ext=".mp4"):
    """Generate next available filename with pattern: prefix_vNN.ext"""
    directory = Path(directory)
    i = 1
    while True:
        path = directory / f"{prefix}_v{i:02d}{ext}"
        if not path.exists():
            return path
        i += 1

def box_center(box):
    """Return center point (x,y) of box [x1,y1,x2,y2]"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def box_iou(box1, box2):
    """Calculate IOU between two boxes [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / float(box1_area + box2_area - intersection)

def draw_bbox(img, box, label=None, color=(0,255,0), thickness=2):
    """Draw a bounding box with label on image"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = CONFIG["viz"]["font_scale"]
        text_thickness = max(1, int(thickness/2))
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
        cv2.rectangle(img, (x1, y1-th-5), (x1+tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1-5), font, font_scale, (255,255,255), text_thickness)
```

This cell defines utility functions for file naming, box operations, and drawing bounding boxes on images.

### YOLO Model Loading Cell

```python
# ==== YOLO MODEL LOADING ====
def safe_load_yolo(weights_path):
    """Safely load YOLO model with error handling"""
    try:
        model = YOLO(weights_path)
        print(f"Model loaded: {weights_path}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        # Create a minimal model stub for testing without YOLO
        class YOLOStub:
            def predict(self, **kwargs):
                from types import SimpleNamespace
                # Return empty detection to allow code to run without crashing
                return [SimpleNamespace(boxes=None)]
        print("Warning: Using YOLOStub (no real detections)")
        return YOLOStub()
```

This cell defines a function to safely load the YOLOv8 model with error handling in case the model isn't available.

### Pitch Detection Functions Cell

```python
# ==== PITCH DETECTION ====
def pitch_mask(img):
    """Extract binary mask of the grass pitch"""
    # Convert to HSV and extract green channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Green color range for typical football pitch
    lower = np.array([35, 20, 20])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Morphological operations to clean up mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def inner_pitch_mask(pitch_mask, shrink_px=30):
    """Get the inner pitch area (exclude touchlines/sidelines)"""
    # Erode the pitch mask to get the inner area
    kernel = np.ones((shrink_px, shrink_px), np.uint8)
    inner = cv2.erode(pitch_mask, kernel)
    return inner

def bbox_on_inner_pitch(inner_mask, box, min_overlap_ratio=0.5):
    """Check if bounding box is on the inner pitch area"""
    x1, y1, x2, y2 = map(int, box)
    h, w = inner_mask.shape
    
    # Clip to image bounds
    x1 = max(0, min(w-1, x1))
    y1 = max(0, min(h-1, y1))
    x2 = max(0, min(w-1, x2))
    y2 = max(0, min(h-1, y2))
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    # Get box mask and calculate overlap
    box_area = (x2-x1) * (y2-y1)
    if box_area == 0:
        return False
        
    # Bottom half of the box (feet area)
    feet_y1 = y1 + int((y2-y1) * 0.6)
    feet_mask = inner_mask[feet_y1:y2, x1:x2]
    if feet_mask.size == 0:
        return False
        
    overlap = np.sum(feet_mask > 0) / float(feet_mask.size)
    return overlap >= min_overlap_ratio
```

This cell contains functions for detecting the football pitch and determining if objects are on the playing field.

### Team Differentiation Cell

```python
# ==== TEAM DIFFERENTIATION ====
import numpy as np
from sklearn.cluster import KMeans

class Team2Clustering:
    """
    Class for clustering players into two teams based on jersey color
    Uses exponential moving average for stable team colors
    """
    def __init__(self, alpha=0.2):
        self.cents = None
        self.alpha = alpha
        self.valid = False
        
    def fit_predict(self, feats):
        """Fit K-means and predict team assignments"""
        if feats.shape[0] < 2:
            return np.zeros(feats.shape[0], dtype=int)
            
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feats)
        
        # Update centroids with EMA
        if self.cents is None:
            self.cents = kmeans.cluster_centers_.copy()
            self.valid = True
        else:
            # Match current centroids to previous (could be swapped)
            dists = []
            for i in range(2):
                for j in range(2):
                    d = np.linalg.norm(kmeans.cluster_centers_[i] - self.cents[j])
                    dists.append((d, i, j))
            dists.sort()
            _, i0, j0 = dists[0]
            _, i1, j1 = dists[-1]
            
            # Update with EMA
            self.cents[j0] = (1-self.alpha) * self.cents[j0] + self.alpha * kmeans.cluster_centers_[i0]
            self.cents[j1] = (1-self.alpha) * self.cents[j1] + self.alpha * kmeans.cluster_centers_[i1]
            
            # Remap labels to be consistent with previous centroids
            if i0 == 1 and j0 == 0:  # centroids were swapped
                labels = 1 - labels
                
        return labels
        
    def assign(self, feats):
        """Assign features to closest centroid"""
        if not self.valid:
            return np.zeros(feats.shape[0], dtype=int)
            
        labels = []
        for feat in feats:
            d0 = np.linalg.norm(feat - self.cents[0])
            d1 = np.linalg.norm(feat - self.cents[1])
            labels.append(0 if d0 < d1 else 1)
        return np.array(labels)
```

This cell implements the team differentiation algorithm using K-means clustering on jersey colors with exponential moving average for stability.

### Jersey Feature Extraction Cell

```python
# ==== JERSEY FEATURE EXTRACTION ====
def jersey_feature_hsv(frame, box):
    """Extract HSV color feature from jersey area"""
    roi = crop_jersey_roi(frame, box)
    if roi.size == 0:
        return None
        
    # Convert to HSV and get mean color
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Only consider reasonably saturated and bright pixels
    mask = (s > 30) & (v > 30)
    if np.sum(mask) < 10:
        return None
        
    # Get mean HSV on masked pixels
    h_mean = np.mean(h[mask]) / 180.0
    s_mean = np.mean(s[mask]) / 255.0
    v_mean = np.mean(v[mask]) / 255.0
    
    return np.array([h_mean, s_mean, v_mean], dtype=np.float32)

def far_from_teams(feat, cents, tau=30.0):
    """Check if feature is far from both team centroids"""
    if cents is None or feat is None:
        return False
        
    d0 = np.linalg.norm(feat - cents[0])
    d1 = np.linalg.norm(feat - cents[1])
    return min(d0, d1) > tau

def referee_color_heuristic(feat):
    """Simple heuristic to identify referee by color (black/white/yellow)"""
    if feat is None:
        return False
        
    h, s, v = feat
    
    # Black: low V
    if v < 0.25 and s < 0.3:
        return True
        
    # White: high V, low S
    if v > 0.7 and s < 0.15:
        return True
        
    # Yellow referee: H around 0.15-0.2, high S
    h_norm = h * 360
    if (25 < h_norm < 65) and s > 0.6 and v > 0.7:
        return True
        
    return False
```

This cell contains functions for extracting jersey color features and identifying referee uniforms.

### Object Tracking Cell

```python
# ==== TRACKING CLASS ====
import numpy as np
from collections import defaultdict

class IOUTracker:
    """
    Simple IOU-based tracker for players and referees
    """
    def __init__(self, iou_thr=0.5, max_age=30):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}  # id -> {"box": bbox, "age": int, "class": str, "conf": float}
        
    def _iou(self, box1, box2):
        """Calculate IOU between two boxes [x1,y1,x2,y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / float(box1_area + box2_area - intersection)
        
    def update(self, boxes, classes=None, confs=None):
        """
        Update tracker with new detections
        
        Args:
            boxes: list of [x1,y1,x2,y2] bounding boxes
            classes: list of class labels
            confs: list of confidence scores
            
        Returns:
            Dictionary of track_id -> {"box": box, "class": class, "conf": conf}
        """
        # Fill missing attributes with defaults
        if classes is None:
            classes = ["unknown"] * len(boxes)
        if confs is None:
            confs = [1.0] * len(boxes)
        
        # If no tracks yet, initialize with current detections
        if not self.tracks:
            for box, cls, conf in zip(boxes, classes, confs):
                self.tracks[self.next_id] = {
                    "box": box,
                    "class": cls,
                    "conf": conf,
                    "age": 0
                }
                self.next_id += 1
            return self.get_active_tracks()
            
        # Calculate IOU between existing tracks and new detections
        matched_tracks = set()
        matched_detections = set()
        
        # For each track, find best matching detection
        for track_id, track in self.tracks.items():
            best_iou = self.iou_thr
            best_detection = -1
            
            for i, box in enumerate(boxes):
                if i in matched_detections:
                    continue
                    
                iou = self._iou(track["box"], box)
                if iou > best_iou:
                    best_iou = iou
                    best_detection = i
            
            if best_detection >= 0:
                # Update track with new detection
                self.tracks[track_id]["box"] = boxes[best_detection]
                if classes is not None:
                    self.tracks[track_id]["class"] = classes[best_detection]
                if confs is not None:
                    self.tracks[track_id]["conf"] = confs[best_detection]
                self.tracks[track_id]["age"] = 0
                
                matched_tracks.add(track_id)
                matched_detections.add(best_detection)
        
        # Create new tracks for unmatched detections
        for i, box in enumerate(boxes):
            if i not in matched_detections:
                self.tracks[self.next_id] = {
                    "box": box,
                    "class": classes[i] if classes is not None else "unknown",
                    "conf": confs[i] if confs is not None else 1.0,
                    "age": 0
                }
                self.next_id += 1
        
        # Update age for unmatched tracks
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id]["age"] += 1
        
        # Remove old tracks
        self.tracks = {k:v for k,v in self.tracks.items() if v["age"] <= self.max_age}
        
        return self.get_active_tracks()
        
    def get_active_tracks(self):
        """Return only active (not lost) tracks"""
        return {k:v for k,v in self.tracks.items() if v["age"] == 0}
```

This cell implements an IOU-based tracking algorithm to maintain object identity across video frames.

### Ball Helper Functions Cell

```python
# ==== BALL HELPER FUNCTIONS ====
def near_any_player_feet(ball_box, player_boxes, max_px=120):
    """Check if ball is near any player's feet"""
    if not player_boxes:
        return False
        
    bx, by = box_center(ball_box)
    
    for box in player_boxes:
        # Check distance to bottom center of player box (feet area)
        px = (box[0] + box[2]) / 2
        py = box[3]  # bottom y
        
        dist = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
        if dist < max_px:
            return True
            
    return False

def preprocess_for_ocr(img):
    """Preprocess image for better OCR results"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Try inverting to see if it gives better results
    inv = cv2.bitwise_not(thresh)
    
    # Return both options
    return [thresh, inv]
```

This cell defines helper functions for ball detection and image preprocessing for OCR.

### Preview Loop Cell

```python
# ==== PREVIEW LOOP ====
# Prepare model and trackers
model = safe_load_yolo(CONFIG["paths"]["weights"])
team_mgr = Team2Clustering(alpha=CONFIG["teams"]["ema_alpha"])
tracker = IOUTracker(iou_thr=CONFIG["tracking"]["iou_match"], max_age=CONFIG["tracking"]["max_age"]) if CONFIG["tracking"]["enabled"] else None

# IO
in_path = CONFIG["paths"]["video"]
videos_dir = Path(CONFIG["paths"]["outputs_dir"]) / "videos"
stem = Path(in_path).stem
out_path = str(next_versioned_path(videos_dir, stem + "_preview"))

cap = cv2.VideoCapture(in_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

# motion background (for ball gating)
prev_gray = None
last_ball = None
frame_idx = 0

while frame_idx < CONFIG["runtime"]["preview_frames"]:
    ok, frame = cap.read()
    if not ok or frame is None: break
    frame_idx += 1

    # pitch & inner pitch (excludes touchline/sideline)
    pmask = pitch_mask(frame)
    inner = inner_pitch_mask(pmask, shrink_px=28)

    det = detect_roles(frame)
    players, balls = det["players"], det["balls"]

    # keep only boxes on inner pitch
    players = [(b,p) for (b,p) in players if bbox_on_inner_pitch(inner, b, min_overlap_ratio=0.20)]

    # jersey features
    feats, pboxes = [], []
    for (b, p) in players:
        f = jersey_feature_hsv(frame, b)
        if f is not None:
            feats.append(f); pboxes.append(b)
    feats = np.array(feats, dtype=np.float32) if feats else np.empty((0,3), dtype=np.float32)

    # teams K=2
    labels = None
    if feats.shape[0] >= CONFIG["teams"]["min_players_for_cluster"]:
        labels = team_mgr.fit_predict(feats) if not team_mgr.valid else team_mgr.assign(feats)

    vis = frame.copy()

    # tracking (for number memory)
    tracks = tracker.update([b for (b, p) in players]) if tracker is not None else {}

    # classify each player box as Team A / Team B / Referee
    for idx, b in enumerate(pboxes):
        lab_txt, color = "player", (0,255,0)
        feat = feats[idx]
        ref_like = False
        if team_mgr.valid:
            if far_from_teams(feat, team_mgr.cents, tau=45.0) or referee_color_heuristic(feat):
                ref_like = True
        if labels is not None and not ref_like:
            lab = labels[idx]
            if lab == 0:
                lab_txt, color = "Team A", (0,128,255)
            else:
                lab_txt, color = "Team B", (255,128,0)
        else:
            # if ref-like and on-pitch → Referee
            lab_txt, color = ("Referee", (0,0,0)) if ref_like else ("player", (0,255,0))

        # find best track id to attach number memory
        tid_best, best_iou = None, 0
        for tid, trk in tracks.items():
            iou = box_iou(b, trk["box"])
            if iou > best_iou:
                tid_best, best_iou = tid, iou

        # OCR every N frames (optional)
        if CONFIG["ocr"]["enabled"] and frame_idx % CONFIG["ocr"]["sample_every_n_frames"] == 0:
            if (b[3]-b[1]) >= CONFIG["ocr"]["min_bbox_h"]:
                roi = crop_jersey_roi(frame, b)
                if roi.size:
                    # quality gate: blur measure
                    fm = cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    if fm > 80:  # not too blurry
                        def preprocess(c):
                            g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
                            g1 = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
                            g2 = cv2.bitwise_not(g1)
                            return [g1, g2]
                        candidates = preprocess(roi)
                        num = None
                        try:
                            import easyocr, torch, types, importlib
                            if not hasattr(torch, "version"):
                                torch.version = types.SimpleNamespace()
                                torch.version.__version__ = torch.__version__
                            if not hasattr(torch, "_utils"):
                                torch._utils = importlib.import_module("torch.utils")
                            reader = easyocr.Reader(['en'], gpu=False)
                            for img in candidates:
                                res = reader.readtext(img, detail=0, allowlist="0123456789")
                                if res:
                                    txt = max(res, key=len)
                                    txt = "".join(ch for ch in txt if ch.isdigit())
                                    if txt:
                                        lo, hi = CONFIG["ocr"]["valid_len"]
                                        if lo <= len(txt) <= hi:
                                            num = txt; break
                        except Exception:
                            pass
                        if num and tid_best is not None:
                            tracks[tid_best]["number"] = num

        if tid_best is not None and tracks[tid_best].get("number"):
            lab_txt += f" #{tracks[tid_best]['number']}"

        draw_bbox(vis, b, lab_txt, color=color, thickness=CONFIG["viz"]["draw_thickness"])

    # Ball detection and tracking
    # ...code continues for ball detection...

cap.release()
writer.release()
print(f"[OK] Preview written to: {out_path}")
```

This cell implements the preview processing loop that handles a limited number of frames to verify the pipeline works correctly.

### Full Video Processing Cell

```python
# ==== FULL VIDEO PROCESSING ====
def process_video(video_path, model, output_name=None, process_every_n=1):
    """
    Process full video with the complete pipeline
    
    Args:
        video_path: Path to input video
        model: YOLO model for detection
        output_name: Path for output video (or auto-generated if None)
        process_every_n: Process every nth frame (for speed)
        
    Returns:
        Path to the output video
    """
    # Initialize trackers
    team_mgr = Team2Clustering(alpha=CONFIG["teams"]["ema_alpha"])
    tracker = IOUTracker(iou_thr=CONFIG["tracking"]["iou_match"], max_age=CONFIG["tracking"]["max_age"]) if CONFIG["tracking"]["enabled"] else None
    
    # Set up paths
    video_path = Path(video_path)
    if output_name is None:
        videos_dir = Path(CONFIG["paths"]["outputs_dir"]) / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        stem = video_path.stem
        output_name = str(next_versioned_path(videos_dir, stem + "_full"))
    
    # Initialize video reader and writer
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = output_name if output_name.endswith(".mp4") else output_name + ".mp4"
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    
    # Initialize variables
    prev_gray = None
    last_ball = None
    frame_count = 0
    
    # OCR reader initialization
    ocr_reader = None
    if CONFIG["ocr"]["enabled"]:
        try:
            if CONFIG["ocr"]["engine"].lower() == "easyocr":
                import easyocr
                print("Initializing EasyOCR (this may take a moment)...")
                ocr_reader = easyocr.Reader(['en'], gpu=False)
                print("EasyOCR ready ✓")
        except Exception as e:
            print(f"OCR initialization error: {e}")
    
    # Process each frame
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {out_path}")
    
    start_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
            
        frame_count += 1
        if frame_count % process_every_n != 0:
            continue
            
        # Full processing pipeline
        # ...code for processing each frame...
    
    cap.release()
    writer.release()
    
    duration = time.time() - start_time
    print(f"✓ Processing complete!")
    print(f"  - Frames: {frame_count}")
    print(f"  - Time: {duration:.2f}s ({frame_count/duration:.2f} fps)")
    print(f"  - Output: {out_path}")
    
    return out_path

# Run full processing if requested
if CONFIG["runtime"]["full_run"]:
    output_path = process_video(
        video_path=CONFIG["paths"]["video"],
        model=model,
        output_name=CONFIG["paths"]["out_video_name"],
        process_every_n=CONFIG["runtime"]["process_every_n"]
    )
    
    # Display completion message
    print("\nTask Completed Successfully! ✓")
```

This cell implements the full video processing function that handles the entire video file.

### Team Analysis Cell

```python
# ==== TEAM ANALYSIS ====
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans

def extract_team_colors(frame, player_boxes):
    """
    Extract team colors using K-means clustering
    
    Args:
        frame: Input video frame
        player_boxes: List of player bounding boxes
        
    Returns:
        team_assignments: Team assignment for each player
        team_colors: BGR color tuple for each team
        valid_boxes: The boxes used for clustering
    """
    jersey_colors = []
    valid_boxes = []
    
    # Extract jersey colors from player bounding boxes
    for box in player_boxes:
        # Extract jersey ROI (center portion of the box)
        roi = crop_jersey_roi(frame, box)
        
        if roi.size == 0:
            continue
        
        # Get dominant color
        dominant_color = get_dominant_color(roi, k=1)
        
        # Skip if color is too dark (likely referee)
        hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_BGR2HSV)[0][0]
        if hsv[1] < 40 or hsv[2] < 40:
            continue
            
        jersey_colors.append(dominant_color)
        valid_boxes.append(box)
    
    if len(jersey_colors) < 2:
        return [], [], []
        
    # Convert to numpy array for clustering
    colors_array = np.array(jersey_colors)
    
    # Use KMeans to find the two team clusters
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    team_assignments = kmeans.fit_predict(colors_array)
    
    # Get representative color for each team
    team_colors = kmeans.cluster_centers_.astype(int)
    
    # Convert back to BGR tuples
    team_colors_bgr = [tuple(map(int, color)) for color in team_colors]
    
    return team_assignments, team_colors_bgr, valid_boxes
```

This cell implements team color analysis to visualize and verify the team differentiation.

### Jersey Number Detection Cell

```python
# ==== JERSEY NUMBER DETECTION USING OCR ====
import re
import easyocr

def init_ocr():
    """Initialize the OCR reader based on config"""
    if CONFIG["ocr"]["engine"].lower() == "easyocr":
        print("Initializing EasyOCR (this may take a moment)...")
        reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR ready ✓")
        return reader
    else:
        # Fallback to Tesseract if configured
        import pytesseract
        print("Using Tesseract OCR")
        return None  # No initialization needed for pytesseract

def detect_jersey_number(img, reader):
    """
    Detect jersey number in the given ROI image.
    
    Args:
        img: The cropped jersey ROI
        reader: OCR reader object
        
    Returns:
        detected_number: The detected jersey number as string, or None
    """
    if img.size == 0 or img.shape[0] < CONFIG["ocr"]["min_bbox_h"]:
        return None
    
    # Preprocess the image for better OCR results
    processed = preprocess_for_ocr(img)
    
    # Detect text
    if CONFIG["ocr"]["engine"].lower() == "easyocr":
        results = reader.readtext(processed)
        
        # Process results
        if len(results) == 0:
            return None
            
        best_text = None
        max_conf = 0
        
        for (_, text, conf) in results:
            # Keep only numeric text
            if CONFIG["ocr"]["keep_numeric_only"]:
                text = re.sub(r'[^0-9]', '', text)
            
            # Check if valid length and higher confidence
            if text and CONFIG["ocr"]["valid_len"][0] <= len(text) <= CONFIG["ocr"]["valid_len"][1]:
                if conf > max_conf:
                    best_text = text
                    max_conf = conf
        
        return best_text
```

This cell implements jersey number detection using OCR (Optical Character Recognition).

### Final Cell with Result Display

```python
# ==== RESULTS & DISPLAY ====
import matplotlib.pyplot as plt

# Display a sample frame with annotations
# Code to display results and metrics

print("Football video analysis complete!")
print(f"- Teams differentiated: 2")
print(f"- Players detected: {len(players)}")
print(f"- Jersey numbers detected: {len([t for t in tracks.values() if 'number' in t])}")
print(f"- Preview video saved to: {out_path}")

# Show the last processed frame with annotations
plt.figure(figsize=(12,8))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.title("Football Video Analysis Results")
plt.axis("off")
plt.show()
```

This final cell displays results and metrics from the video analysis pipeline.
