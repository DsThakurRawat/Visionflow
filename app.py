"""TrafficFlow Pro — Intelligent Traffic Telemetry & Directional Flow Platform.

Enterprise-grade computer vision application for automated video analysis:
- Dual-stream video comparison (Input vs. Output)
- Multi-class directional vehicle counting (In / Out per class)
- Calibrated speed estimation & trajectory tracking
- License plate recognition & CSV telemetry export
"""

import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO

# ==========================================
# PAGE CONFIG & PROFESSIONAL ENTERPRISE UI
# ==========================================
st.set_page_config(
    page_title="TrafficFlow Pro — Traffic Intelligence Platform",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ENTERPRISE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-base: #0B0F17;
        --bg-surface: #111827;
        --bg-surface-elevated: #1F2937;
        --border-subtle: #374151;
        --border-focus: #3B82F6;
        --text-primary: #F9FAFB;
        --text-secondary: #9CA3AF;
        --text-muted: #6B7280;
        --accent-blue: #2563EB;
        --accent-cyan: #0284C7;
        --accent-green: #10B981;
        --accent-red: #EF4444;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background-color: var(--bg-base);
    }

    /* Header styling */
    .platform-header {
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 18px;
        margin-bottom: 28px;
    }

    .platform-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.025em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .platform-subtitle {
        font-size: 0.95rem;
        color: var(--text-secondary);
        margin-top: 6px;
    }

    /* Structured Panels */
    .panel-box {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 24px;
    }

    .panel-title {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 14px;
    }

    /* Metric Cards */
    .metric-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 16px;
        height: 100%;
    }

    .metric-header {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--text-muted);
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-primary);
        line-height: 1.1;
    }

    .metric-footer {
        display: flex;
        gap: 14px;
        margin-top: 10px;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        padding-top: 8px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }

    .count-in {
        color: var(--accent-green);
        display: flex;
        align-items: center;
        gap: 4px;
    }

    .count-out {
        color: var(--accent-red);
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Primary Generate Button */
    .stButton>button {
        background-color: var(--accent-blue) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.75rem !important;
        transition: background-color 0.15s ease !important;
    }

    .stButton>button:hover {
        background-color: #1D4ED8 !important;
    }

    /* Download Buttons */
    .stDownloadButton>button {
        background-color: var(--bg-surface-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
    }

    .stDownloadButton>button:hover {
        border-color: var(--border-focus) !important;
        background-color: #2D3748 !important;
    }

    /* Video container label */
    .video-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
</style>
"""
st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)

# ==========================================
# CONSTANTS & VEHICLE SPECIFICATIONS
# ==========================================
VEHICLE_CONFIGS = {
    2: {"name": "car", "label": "Cars", "real_width_m": 1.9},
    3: {"name": "motorcycle", "label": "Motorcycles", "real_width_m": 0.9},
    5: {"name": "bus", "label": "Buses", "real_width_m": 2.55},
    7: {"name": "truck", "label": "Trucks", "real_width_m": 2.50},
}
VEHICLE_CLASS_IDS = list(VEHICLE_CONFIGS.keys())
DEFAULT_MODEL = "yolo11s.pt"
DEFAULT_PLATE_MODEL = "license-plate-finetune-v1s.pt"


# ==========================================
# DATA BOOK & TELEMETRY
# ==========================================
class VehicleBook:
    """Maintains vehicle trajectories, speeds, OCR plate voting, and tripwire direction."""

    def __init__(self, fps: float, maxlen: int) -> None:
        self._fps = fps
        self._lines: dict[int, deque[tuple]] = defaultdict(lambda: deque(maxlen=maxlen))
        self._plate_votes: dict[int, Counter[str]] = defaultdict(Counter)
        self._direction: dict[int, str] = defaultdict(lambda: "In-Transit")

    def record(self, tracker_id: int, frame_index: int, anchor: np.ndarray, width_px: float, class_id: int) -> None:
        self._lines[tracker_id].append((frame_index, tuple(anchor), width_px, class_id))

    def mark_crossing(self, tracker_id: int, direction: str) -> None:
        self._direction[tracker_id] = direction

    def vote_plate(self, tracker_id: int, text: str) -> None:
        if tracker_id >= 0 and text:
            self._plate_votes[tracker_id][text] += 1

    def winner_plate(self, tracker_id: int) -> str:
        votes = self._plate_votes.get(tracker_id)
        return votes.most_common(1)[0][0] if votes else ""

    def speed_kmh(self, tracker_id: int, real_width_m: float) -> float | None:
        history = self._lines.get(tracker_id)
        if history is None or len(history) < 2:
            return None
        first, last = history[0], history[-1]
        if last[0] - first[0] < self._fps * 0.5:
            return None
        widths = [np.float64(e[2]) for e in history if e[2] > 20]
        if not widths:
            return None
        m_per_px = real_width_m / float(np.median(widths))
        distance_m = np.hypot(last[1][0] - first[1][0], last[1][1] - first[1][1]) * m_per_px
        elapsed_s = (last[0] - first[0]) / self._fps
        return (distance_m / elapsed_s) * 3.6 if elapsed_s > 0 else None

    def to_dataframe(self, class_names: dict[int, str]) -> pd.DataFrame:
        records = []
        for tracker_id, history in sorted(self._lines.items()):
            if not history:
                continue
            class_id = int(history[-1][3])
            real_width = VEHICLE_CONFIGS.get(class_id, {}).get("real_width_m", 2.0)
            speed = self.speed_kmh(tracker_id, real_width)
            records.append({
                "Tracker ID": f"#{tracker_id}",
                "Classification": class_names.get(class_id, "Vehicle").capitalize(),
                "Speed (km/h)": round(speed, 1) if speed is not None else None,
                "License Plate": self.winner_plate(tracker_id) or "—",
                "Direction": self._direction.get(tracker_id, "In-Transit"),
                "First Frame": history[0][0],
                "Last Frame": history[-1][0],
                "Duration (Frames)": len(history),
            })
        return pd.DataFrame(records)


def transcode_to_h264(input_path: str, output_path: str) -> bool:
    """Converts the video file to standard browser-playable H.264 (yuv420p)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    try:
        command = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", input_path,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_path,
        ]
        subprocess.run(command, check=True)
        return True
    except Exception:
        return False


def read_plate_text(ocr, frame: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> str:
    """Reads license plate characters using RapidOCR."""
    crop = frame[max(0, y0 - 1): y1 + 1, max(0, x0 - 1): x1 + 1]
    if crop.size == 0 or ocr is None:
        return ""
    try:
        results, _ = ocr(crop)
        if not results:
            return ""
        best_text, best_score = "", 0.0
        for _, text, score in results:
            cleaned = "".join(ch for ch in str(text).upper() if ch.isalnum())
            if 4 <= len(cleaned) <= 12 and score > best_score:
                best_text, best_score = cleaned, score
        return best_text if best_score >= 0.4 else ""
    except Exception:
        return ""


def match_plate_to_vehicle(plate_xyxy: np.ndarray, vehicles_xyxy: np.ndarray) -> int:
    """Associates a detected license plate crop to its parent vehicle."""
    if len(vehicles_xyxy) == 0:
        return -1
    cx = (plate_xyxy[0] + plate_xyxy[2]) / 2
    cy = (plate_xyxy[1] + plate_xyxy[3]) / 2
    inside = (
        (vehicles_xyxy[:, 0] <= cx) & (cx <= vehicles_xyxy[:, 2]) &
        (vehicles_xyxy[:, 1] <= cy) & (cy <= vehicles_xyxy[:, 3])
    )
    if not inside.any():
        return -1
    areas = (vehicles_xyxy[:, 2] - vehicles_xyxy[:, 0]) * (vehicles_xyxy[:, 3] - vehicles_xyxy[:, 1])
    areas[~inside] = -1
    return int(np.argmax(areas))


def draw_tripwire_preview(
    video_path: str,
    y_pct: int = 72,
    tilt_pct: int = 0,
    x_start_pct: int = 0,
    x_end_pct: int = 100,
) -> np.ndarray | None:
    """Extract a sample video frame and render the calibrated tripwire counting line."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frame_idx = min(30, max(0, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    h, w = frame.shape[:2]

    y1 = int(h * np.clip((y_pct - tilt_pct / 2.0) / 100.0, 0.05, 0.95))
    y2 = int(h * np.clip((y_pct + tilt_pct / 2.0) / 100.0, 0.05, 0.95))
    x1 = int(w * (x_start_pct / 100.0))
    x2 = int(w * (x_end_pct / 100.0))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tripwire glowing line (Cyan #00E5FF)
    cv2.line(rgb, (x1, y1), (x2, y2), (0, 229, 255), 4, cv2.LINE_AA)
    # Endpoint handles
    cv2.circle(rgb, (x1, y1), 8, (255, 215, 0), -1, cv2.LINE_AA)
    cv2.circle(rgb, (x1, y1), 10, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(rgb, (x2, y2), 8, (255, 215, 0), -1, cv2.LINE_AA)
    cv2.circle(rgb, (x2, y2), 10, (0, 0, 0), 2, cv2.LINE_AA)

    # Direction badges
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2

    # IN Badge (Green)
    cv2.rectangle(rgb, (mid_x - 70, max(10, mid_y - 36)), (mid_x + 70, max(38, mid_y - 8)), (16, 185, 129), -1)
    cv2.putText(rgb, 'IN (Away) ^', (mid_x - 55, max(30, mid_y - 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # OUT Badge (Red)
    cv2.rectangle(rgb, (mid_x - 70, min(h - 38, mid_y + 8)), (mid_x + 70, min(h - 10, mid_y + 36)), (239, 68, 68), -1)
    cv2.putText(rgb, 'v OUT (Near)', (mid_x - 60, min(h - 18, mid_y + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return rgb


# ==========================================
# PLATFORM HEADER
# ==========================================
st.markdown(
    """
    <div class="platform-header">
        <h1 class="platform-title">🚦 TrafficFlow Pro — Vision Analytics Engine</h1>
        <div class="platform-subtitle">
            Autonomous multi-class vehicle detection, directional passage counting (In/Out), speed telemetry, and license plate recognition.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# MAIN SECTION: UPLOAD & CONFIGURATION
# ==========================================
st.markdown("### 1. Video Ingestion")

uploaded_file = st.file_uploader(
    "Select a traffic surveillance video to process (.mp4, .mov, .avi, .mkv):",
    type=["mp4", "mov", "avi", "mkv"],
    help="Upload your video file to initiate automated vehicle counting and telemetry analysis.",
)

if uploaded_file is None:
    st.info("Upload a video file above to proceed.")
    st.stop()

# Reset state if a different video is uploaded
if uploaded_file is not None and st.session_state.get("active_filename") != uploaded_file.name:
    st.session_state["active_filename"] = uploaded_file.name
    st.session_state["results_ready"] = False
    # Auto-seed preset if recognized
    if "input2" in uploaded_file.name.lower():
        st.session_state["calib_line_offset"] = 55
    else:
        st.session_state["calib_line_offset"] = 75
    st.session_state["calib_line_tilt"] = 0
    st.session_state["calib_lane_span"] = (0, 100)

# Save uploaded file to a deterministic temporary path
temp_dir = tempfile.gettempdir()
input_video_path = os.path.join(temp_dir, f"upload_{uploaded_file.name}")
if not os.path.exists(input_video_path) or os.path.getsize(input_video_path) != uploaded_file.size:
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

st.success(f"Video uploaded successfully: **{uploaded_file.name}** ({uploaded_file.size / (1024 * 1024):.1f} MB)")

# Initialize default calibration values in session state
if "calib_line_offset" not in st.session_state:
    st.session_state["calib_line_offset"] = 75
if "calib_line_tilt" not in st.session_state:
    st.session_state["calib_line_tilt"] = 0
if "calib_lane_span" not in st.session_state:
    st.session_state["calib_lane_span"] = (0, 100)

# ==========================================
# 2. COUNTING LINE CALIBRATION (LIVE PREVIEW)
# ==========================================
st.markdown("### 2. Interactive Counting Line Calibration")
st.caption("Visually align the cyan tripwire across the roadway where vehicles travel to ensure 100% counting accuracy.")

calib_col_preview, calib_col_controls = st.columns([1.35, 1.0])

with calib_col_controls:
    st.markdown("**Road Camera Angle Presets:**")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        if st.button("🛣️ Lower Road (75%)", use_container_width=True, help="Optimal for low camera angles like input.mp4"):
            st.session_state["calib_line_offset"] = 75
            st.session_state["calib_line_tilt"] = 0
            st.rerun()
    with p_col2:
        if st.button("🌉 Mid-Way (55%)", use_container_width=True, help="Optimal for overpass cameras like input2.mp4"):
            st.session_state["calib_line_offset"] = 55
            st.session_state["calib_line_tilt"] = 0
            st.rerun()
    with p_col3:
        if st.button("🔭 Horizon (40%)", use_container_width=True, help="Optimal for high distance cameras"):
            st.session_state["calib_line_offset"] = 40
            st.session_state["calib_line_tilt"] = 0
            st.rerun()

    line_offset = st.slider(
        "Line Height Position (% from top):",
        min_value=10, max_value=95,
        value=st.session_state["calib_line_offset"],
        step=2,
        key="slider_offset",
        help="Slide up or down to place the counting line directly through moving traffic.",
    )
    st.session_state["calib_line_offset"] = line_offset

    line_tilt = st.slider(
        "Line Angle / Tilt (%):",
        min_value=-25, max_value=25,
        value=st.session_state["calib_line_tilt"],
        step=1,
        key="slider_tilt",
        help="Tilt the tripwire if the roadway is diagonal or slanted.",
    )
    st.session_state["calib_line_tilt"] = line_tilt

    lane_span = st.slider(
        "Lane Coverage (% Width):",
        min_value=0, max_value=100,
        value=st.session_state["calib_lane_span"],
        step=5,
        key="slider_span",
        help="Restrict tripwire to specific lanes if desired.",
    )
    st.session_state["calib_lane_span"] = lane_span

    with st.expander("⚙️ Advanced Detection Parameters", expanded=False):
        conf_thresh = st.slider("Detection Confidence", 0.15, 0.85, 0.30, 0.05)
        trigger_anchor = st.selectbox(
            "Trigger Anchor",
            ["Center (Recommended for Vehicles)", "Bottom Center (Road Contact)", "All 4 Box Corners (Strict)"],
            index=0,
            help="Center point eliminates boundary straddling dropouts.",
        )
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            enable_speed = st.checkbox("Estimate Speeds", value=True)
            enable_traces = st.checkbox("Draw Motion Traces", value=True)
        with col_c2:
            enable_plates = st.checkbox("License Plates", value=True)

with calib_col_preview:
    preview_img = draw_tripwire_preview(
        input_video_path,
        y_pct=line_offset,
        tilt_pct=line_tilt,
        x_start_pct=lane_span[0],
        x_end_pct=lane_span[1],
    )
    if preview_img is not None:
        st.image(
            preview_img,
            caption=f"Tripwire Placement — Position: {line_offset}% | Tilt: {line_tilt}% | Span: {lane_span[0]}%–{lane_span[1]}%",
            use_container_width=True,
        )
    else:
        st.warning("Could not render frame preview.")

# Single Action Button
generate_btn = st.button("🚀 Generate Traffic Analysis & Metrics", use_container_width=True)

if generate_btn:
    os.makedirs("data", exist_ok=True)
    video_stem = Path(uploaded_file.name).stem
    temp_raw_output = f"data/raw_{video_stem}_output.mp4"
    final_output_path = f"data/output_{video_stem}.mp4"
    summary_csv_path = f"data/summary_{video_stem}.csv"

    # Status & Progress UI
    status_text = st.empty()
    progress_bar = st.progress(0.0)

    # Initialize OCR
    ocr = None
    if enable_plates:
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR()
        except Exception:
            ocr = None

    status_text.markdown("⏳ **Loading detection neural network and tracker...**")
    model = YOLO(DEFAULT_MODEL if Path(DEFAULT_MODEL).is_file() else "yolo11n.pt")
    plate_model = YOLO(DEFAULT_PLATE_MODEL) if (enable_plates and Path(DEFAULT_PLATE_MODEL).is_file()) else None

    video_info = sv.VideoInfo.from_video_path(input_video_path)
    total_frames = video_info.total_frames or 1
    w, h = video_info.resolution_wh

    # Establish Calibrated LineZone
    y1 = int(h * np.clip((line_offset - line_tilt / 2.0) / 100.0, 0.05, 0.95))
    y2 = int(h * np.clip((line_offset + line_tilt / 2.0) / 100.0, 0.05, 0.95))
    x1 = int(w * (lane_span[0] / 100.0))
    x2 = int(w * (lane_span[1] / 100.0))

    start_pt = sv.Point(x1, y1)
    end_pt = sv.Point(x2, y2)

    if trigger_anchor == "Center (Recommended for Vehicles)":
        chosen_anchors = [sv.Position.CENTER]
    elif trigger_anchor == "Bottom Center (Road Contact)":
        chosen_anchors = [sv.Position.BOTTOM_CENTER]
    else:
        chosen_anchors = (
            sv.Position.TOP_LEFT,
            sv.Position.TOP_RIGHT,
            sv.Position.BOTTOM_LEFT,
            sv.Position.BOTTOM_RIGHT,
        )

    line_zone = sv.LineZone(
        start=start_pt,
        end=end_pt,
        triggering_anchors=chosen_anchors,
    )
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=2,
        text_scale=0.5,
        text_thickness=1,
        color=sv.Color.from_hex("#38BDF8"),
    )

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = 0.55 * sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
    vehicle_label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK,
    )
    plate_label_annotator = sv.LabelAnnotator(
        text_scale=0.35,
        text_thickness=1,
        text_position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.INDEX,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(video_info.fps * 1.5),
        position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK,
    )

    byte_track = ByteTrackTracker(frame_rate=video_info.fps, track_activation_threshold=conf_thresh)
    smoother = sv.DetectionsSmoother(length=3)
    book = VehicleBook(video_info.fps, maxlen=int(video_info.fps))

    start_time = time.time()

    with sv.VideoSink(temp_raw_output, video_info) as sink:
        for frame_index, frame in enumerate(sv.get_video_frames_generator(source_path=input_video_path)):
            # 1. Detection & Tracking
            results = model(frame, conf=conf_thresh, iou=0.70, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = byte_track.update(detections)
            detections = detections[detections.tracker_id != -1]
            detections = smoother.update_with_detections(detections)

            # Filter Vehicles
            is_vehicle = np.isin(detections.class_id, VEHICLE_CLASS_IDS)
            vehicles = detections[is_vehicle]

            # 2. LineZone In/Out Crossing
            crossed_in, crossed_out = line_zone.trigger(vehicles)
            for idx, c_in in enumerate(crossed_in):
                if c_in:
                    book.mark_crossing(int(vehicles.tracker_id[idx]), "IN")
            for idx, c_out in enumerate(crossed_out):
                if c_out:
                    book.mark_crossing(int(vehicles.tracker_id[idx]), "OUT")

            # 3. Speed Tracking
            if enable_speed and len(vehicles):
                anchors = vehicles.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                for i, tracker_id in enumerate(vehicles.tracker_id):
                    class_id = int(vehicles.class_id[i])
                    box = vehicles.xyxy[i]
                    book.record(tracker_id, frame_index, anchors[i], float(box[2] - box[0]), class_id)

            # 4. License Plate Detection & OCR
            plate_boxes, plate_texts = [], []
            if plate_model is not None and ocr is not None and frame_index % 3 == 0:
                plate_result = plate_model(frame, conf=0.35, iou=0.7, verbose=False)[0]
                plate_detections = sv.Detections.from_ultralytics(plate_result)
                v_xyxy = vehicles.xyxy if len(vehicles) else np.empty((0, 4), dtype=float)
                for plate_box in plate_detections.xyxy.astype(int):
                    matched_idx = match_plate_to_vehicle(plate_box, v_xyxy)
                    plate_boxes.append(plate_box)
                    plate_texts.append("")
                    if matched_idx >= 0:
                        text = read_plate_text(ocr, frame, *plate_box)
                        t_id = int(vehicles.tracker_id[matched_idx])
                        if text:
                            book.vote_plate(t_id, text)
                            plate_texts[-1] = book.winner_plate(t_id)

            # 5. Annotation
            annotated = frame.copy()
            if enable_traces:
                annotated = trace_annotator.annotate(scene=annotated, detections=vehicles)
            if len(detections):
                annotated = box_annotator.annotate(scene=annotated, detections=detections)

            vehicle_labels = []
            for tracker_id, class_id in zip(vehicles.tracker_id, vehicles.class_id):
                v_cfg = VEHICLE_CONFIGS.get(int(class_id), {"name": "vehicle", "real_width_m": 2.0})
                label = f"#{int(tracker_id)} {v_cfg['name']}"
                if enable_speed:
                    speed_val = book.speed_kmh(int(tracker_id), v_cfg["real_width_m"])
                    if speed_val is not None:
                        label += f" ≈{speed_val:.0f}km/h"
                if enable_plates and book.winner_plate(int(tracker_id)):
                    label += f" [{book.winner_plate(int(tracker_id))}]"
                vehicle_labels.append(label)

            if len(vehicles):
                annotated = vehicle_label_annotator.annotate(
                    scene=annotated, detections=vehicles, labels=vehicle_labels
                )

            if plate_boxes:
                plate_dets = sv.Detections(xyxy=np.array(plate_boxes, dtype=np.float32))
                annotated = plate_label_annotator.annotate(
                    scene=annotated, detections=plate_dets, labels=plate_texts
                )

            annotated = line_zone_annotator.annotate(annotated, line_counter=line_zone)
            sink.write_frame(annotated)

            if frame_index % 12 == 0:
                prog = min(1.0, (frame_index + 1) / total_frames)
                progress_bar.progress(prog)
                status_text.markdown(
                    f"⚡ **Processing frame {frame_index + 1}/{total_frames}** "
                    f"| Total In: **{line_zone.in_count}** | Total Out: **{line_zone.out_count}**"
                )

    progress_bar.progress(1.0)
    status_text.markdown("🎬 **Finalizing video stream (H.264 encoding)...**")
    encoded_ok = transcode_to_h264(temp_raw_output, final_output_path)
    if not encoded_ok:
        shutil.copy(temp_raw_output, final_output_path)
    if os.path.exists(temp_raw_output) and os.path.exists(final_output_path):
        os.remove(temp_raw_output)

    df_summary = book.to_dataframe({cid: cfg["name"] for cid, cfg in VEHICLE_CONFIGS.items()})
    df_summary.to_csv(summary_csv_path, index=False)

    elapsed = time.time() - start_time
    status_text.success(f"Processing complete: {total_frames} frames processed in {elapsed:.1f}s ({total_frames / elapsed:.1f} FPS).")

    # Persist in session state
    st.session_state["results_ready"] = True
    st.session_state["input_video_path"] = input_video_path
    st.session_state["output_video_path"] = final_output_path
    st.session_state["summary_csv_path"] = summary_csv_path
    st.session_state["df_summary"] = df_summary
    st.session_state["in_per_class"] = {int(k): int(v) for k, v in line_zone.in_count_per_class.items() if k is not None}
    st.session_state["out_per_class"] = {int(k): int(v) for k, v in line_zone.out_count_per_class.items() if k is not None}
    st.session_state["total_in"] = int(line_zone.in_count)
    st.session_state["total_out"] = int(line_zone.out_count)


# ==========================================
# RESULTS SECTION: SIDE-BY-SIDE VIDEO PLAYERS
# ==========================================
if st.session_state.get("results_ready"):
    input_v_path = st.session_state["input_video_path"]
    output_v_path = st.session_state["output_video_path"]
    summary_c_path = st.session_state["summary_csv_path"]
    df_summary = st.session_state["df_summary"]
    in_per_class = st.session_state["in_per_class"]
    out_per_class = st.session_state["out_per_class"]

    st.markdown("---")
    st.markdown("### 2. Side-by-Side Video Analysis")

    col_raw, col_proc = st.columns(2)

    with col_raw:
        st.markdown('<div class="video-label">📹 Raw Input Video Stream</div>', unsafe_allow_html=True)
        st.video(input_v_path)

    with col_proc:
        st.markdown('<div class="video-label">🎯 Processed & Annotated Video Stream</div>', unsafe_allow_html=True)
        st.video(output_v_path)

    # Export Row Directly Below Videos
    btn_col1, btn_col2, _ = st.columns([1.5, 1.5, 3])
    with btn_col1:
        with open(output_v_path, "rb") as vf:
            st.download_button(
                label="📥 Download Output Video (.mp4)",
                data=vf.read(),
                file_name=Path(output_v_path).name,
                mime="video/mp4",
                use_container_width=True,
            )
    with btn_col2:
        with open(summary_c_path, "rb") as cf:
            st.download_button(
                label="📊 Download Telemetry CSV",
                data=cf.read(),
                file_name=Path(summary_c_path).name,
                mime="text/csv",
                use_container_width=True,
            )

    # ==========================================
    # METRICS SECTION (ALL BELOW VIDEOS)
    # ==========================================
    st.markdown("---")
    st.markdown("### 3. Directional Passage Telemetry (In / Out)")

    # 5 KPI Metric Cards
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    with m_col1:
        tot_in = st.session_state["total_in"]
        tot_out = st.session_state["total_out"]
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-header"><span>All Vehicles</span><span>Total</span></div>
                <div class="metric-value">{tot_in + tot_out}</div>
                <div class="metric-footer">
                    <span class="count-in">↑ In: {tot_in}</span>
                    <span class="count-out">↓ Out: {tot_out}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        car_in = in_per_class.get(2, 0)
        car_out = out_per_class.get(2, 0)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-header"><span>Cars</span><span>Class 2</span></div>
                <div class="metric-value">{car_in + car_out}</div>
                <div class="metric-footer">
                    <span class="count-in">↑ In: {car_in}</span>
                    <span class="count-out">↓ Out: {car_out}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        truck_in = in_per_class.get(7, 0)
        truck_out = out_per_class.get(7, 0)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-header"><span>Trucks</span><span>Class 7</span></div>
                <div class="metric-value">{truck_in + truck_out}</div>
                <div class="metric-footer">
                    <span class="count-in">↑ In: {truck_in}</span>
                    <span class="count-out">↓ Out: {truck_out}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col4:
        bus_in = in_per_class.get(5, 0)
        bus_out = out_per_class.get(5, 0)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-header"><span>Buses</span><span>Class 5</span></div>
                <div class="metric-value">{bus_in + bus_out}</div>
                <div class="metric-footer">
                    <span class="count-in">↑ In: {bus_in}</span>
                    <span class="count-out">↓ Out: {bus_out}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col5:
        moto_in = in_per_class.get(3, 0)
        moto_out = out_per_class.get(3, 0)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-header"><span>Motorcycles</span><span>Class 3</span></div>
                <div class="metric-value">{moto_in + moto_out}</div>
                <div class="metric-footer">
                    <span class="count-in">↑ In: {moto_in}</span>
                    <span class="count-out">↓ Out: {moto_out}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Detailed Charts & Tables
    st.markdown("<br>", unsafe_allow_html=True)
    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Directional Flow Distribution</div>', unsafe_allow_html=True)
        chart_data = pd.DataFrame([
            {"Class": "Cars", "Direction": "In", "Count": in_per_class.get(2, 0)},
            {"Class": "Cars", "Direction": "Out", "Count": out_per_class.get(2, 0)},
            {"Class": "Trucks", "Direction": "In", "Count": in_per_class.get(7, 0)},
            {"Class": "Trucks", "Direction": "Out", "Count": out_per_class.get(7, 0)},
            {"Class": "Buses", "Direction": "In", "Count": in_per_class.get(5, 0)},
            {"Class": "Buses", "Direction": "Out", "Count": out_per_class.get(5, 0)},
            {"Class": "Motorcycles", "Direction": "In", "Count": in_per_class.get(3, 0)},
            {"Class": "Motorcycles", "Direction": "Out", "Count": out_per_class.get(3, 0)},
        ])
        flow_chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Class:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Count:Q", title="Crossed Count"),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(domain=["In", "Out"], range=["#10B981", "#EF4444"]),
                legend=alt.Legend(title="Direction", orient="top"),
            ),
            xOffset="Direction:N",
            tooltip=["Class", "Direction", "Count"],
        ).properties(height=280)
        st.altair_chart(flow_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_chart2:
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Speed Distribution Spectrum</div>', unsafe_allow_html=True)
        speed_series = df_summary["Speed (km/h)"].dropna()
        if len(speed_series) > 0:
            hist_chart = alt.Chart(df_summary.dropna(subset=["Speed (km/h)"])).mark_bar(
                color="#3B82F6", cornerRadiusTopLeft=4, cornerRadiusTopRight=4
            ).encode(
                x=alt.X("Speed (km/h):Q", bin=alt.Bin(maxbins=12), title="Speed Interval (km/h)"),
                y=alt.Y("count()", title="Vehicle Count"),
                tooltip=["count()"],
            ).properties(height=280)
            st.altair_chart(hist_chart, use_container_width=True)
        else:
            st.info("No speed metrics registered for current crossing threshold.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Full Data Table
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Comprehensive Vehicle Telemetry Records</div>', unsafe_allow_html=True)
    st.dataframe(
        df_summary,
        use_container_width=True,
        column_config={
            "Speed (km/h)": st.column_config.ProgressColumn(
                "Speed (km/h)",
                format="%d km/h",
                min_value=0,
                max_value=120,
            ),
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)
