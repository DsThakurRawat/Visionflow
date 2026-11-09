"""Traffic analytics pipeline: detect, track, estimate speeds, and read number plates.

Extends ``process_video.py`` into a full traffic tool:

  1. YOLO over the full COCO class list (everything the camera sees, annotated).
  2. ByteTrack tracker IDs + path traces for road vehicles (car/bus/truck/motorcycle).
  3. Speed per vehicle, self-calibrated from the vehicle's pixel width and a
     real-world width prior (no manual road-plane calibration); labelled ``≈``.
  4. Number plates via a finetuned YOLO11 plate detector, OCR'd with RapidOCR,
     majority-voted across frames and matched to the containing vehicle.
  5. Per-vehicle summary written to CSV.

Usage:

    uv run python traffic_pipeline.py
    uv run python traffic_pipeline.py data/input.mp4 data/traffic.mp4
    uv run python traffic_pipeline.py --no-plates --no-speed

Dependencies: ``uv pip install ultralytics trackers rapidocr-onnxruntime``.
"""

import argparse
import csv
import shutil
import subprocess
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO

DEFAULT_INPUT = "data/input.mp4"
DEFAULT_OUTPUT = "data/traffic.mp4"
DEFAULT_MODEL = "yolo11s.pt"
DEFAULT_PLATE_MODEL = "license-plate-finetune-v1s.pt"
DEFAULT_SUMMARY = "data/summary.csv"

# COCO class ids treated as road vehicles (tracked, sped, plate-matched).
VEHICLE_CLASS_IDS = {2: 1.9, 3: 0.9, 5: 2.55, 7: 2.5}  # car, motorcycle, bus, truck -> real width in meters
CONFIDENCE_THRESHOLD = 0.30
IOU_THRESHOLD = 0.70
OCR_EVERY_N_FRAMES = 3  # OCR is CPU-bound; a few reads per second are plenty for voting
MIN_HISTORY_FRACTION = 0.5  # need at least half a second of track history before trusting speed
MIN_PLATE_TEXT_LENGTH = 4
MAX_PLATE_TEXT_LENGTH = 12
MIN_OCR_SCORE = 0.4
MIN_VEHICLE_PIXEL_WIDTH = 20  # below this the width-prior calibration is noise


class ViewTransformer:
    """Map pixel-space ground points to metric coordinates via a 4-point homography.

    Same pattern as ``examples/speed_estimation``: ``source`` quad in image pixels
    (near-left, near-right, far-right, far-left), ``target`` metric rectangle.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """Build the perspective transform from 4 correspondences."""
        import cv2  # lazy: heavy dep, needed only in --plane mode

        self.matrix = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform ``(N, 2)`` pixel points to metric ground coordinates."""
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(reshaped, self.matrix).reshape(-1, 2)


def parse_plane(spec: str) -> np.ndarray:
    """Parse ``--plane "x1,y1 x2,y2 x3,y3 x4,y4"`` into a ``(4, 2)`` float array."""
    parts = [p for chunk in spec.split() for p in chunk.split(",")]
    if len(parts) != 8:
        raise argparse.ArgumentTypeError("--plane needs 4 points as x1,y1 x2,y2 x3,y3 x4,y4")
    try:
        values = np.array([float(p) for p in parts])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--plane values must be numbers: {exc}") from exc
    return values.reshape(4, 2)


class VehicleBook:
    """Per-vehicle running history, speed votes and plate-text votes."""

    def __init__(self, fps: float, maxlen: int, metric: bool = False) -> None:
        self._fps = fps
        self._metric = metric
        self._lines: dict[int, deque[tuple]] = defaultdict(lambda: deque(maxlen=maxlen))
        self._plate_votes: dict[int, Counter[str]] = defaultdict(Counter)
        self._direction: dict[int, str] = defaultdict(lambda: "In-Transit")

    def record(
        self, tracker_id: int, frame_index: int, anchor: np.ndarray,
        width_px: float, class_id: int,
    ) -> None:
        """Append ``(frame, anchor, width, class)`` to a vehicle's history."""
        self._lines[tracker_id].append((frame_index, tuple(anchor), width_px, class_id))

    def mark_crossing(self, tracker_id: int, direction: str) -> None:
        """Mark directional tripwire crossing ('IN' or 'OUT')."""
        self._direction[tracker_id] = direction

    def vote_plate(self, tracker_id: int, text: str) -> None:
        """Add an OCR result to a vehicle's plate-text vote."""
        if tracker_id >= 0 and text:
            self._plate_votes[tracker_id][text] += 1

    def winner_plate(self, tracker_id: int) -> str:
        """Return the majority-voted plate text, or an empty string."""
        votes = self._plate_votes.get(tracker_id)
        return votes.most_common(1)[0][0] if votes else ""

    def speed_kmh(self, tracker_id: int, real_width_m: float) -> float | None:
        """Estimate speed from the vehicle's history and its width-based scale prior.

        Self-calibrating: meters-per-pixel is ``real_width / median_pixel_width`` over the
        history window, so speed stays scale-consistent without camera calibration. Window
        is 1 s; estimates are approximate (labelled ``≈``) because pixel width changes with
        distance and lateral motion is folded in as-is.
        """
        history = self._lines.get(tracker_id)
        if history is None or len(history) < 2:
            return None
        first, last = history[0], history[-1]
        if last[0] - first[0] < self._fps * MIN_HISTORY_FRACTION:
            return None
        if self._metric:
            # Homography mode: anchors are already in meters; width prior is unused.
            distance_m = np.hypot(last[1][0] - first[1][0], last[1][1] - first[1][1])
        else:
            widths = [np.float64(entry[2]) for entry in history if entry[2] > MIN_VEHICLE_PIXEL_WIDTH]
            if not widths:
                return None
            meters_per_pixel = real_width_m / float(np.median(widths))
            distance_m = np.hypot(last[1][0] - first[1][0], last[1][1] - first[1][1]) * meters_per_pixel
        elapsed_s = (last[0] - first[0]) / self._fps
        return (distance_m / elapsed_s) * 3.6

    def rows_for_summary(self, class_names: dict[int, str]) -> list[list]:
        """Build one CSV row per tracked vehicle id: class, speed, plate, presence."""
        rows = []
        for tracker_id, history in sorted(self._lines.items()):
            if not history:
                continue
            class_id = int(history[-1][3])
            speed = self.speed_kmh(tracker_id, VEHICLE_CLASS_IDS[class_id]) if class_id in VEHICLE_CLASS_IDS else None
            rows.append(
                [
                    tracker_id,
                    class_names.get(class_id, "unknown"),
                    f"{speed:.0f}" if speed is not None else "",
                    self.winner_plate(tracker_id) or "-",
                    self._direction.get(tracker_id, "In-Transit"),
                    history[0][0],
                    history[-1][0],
                    len(history),
                ]
            )
        return rows


def transcode_to_h264(video_path: str) -> None:
    """Re-encode ``video_path`` in place to H.264 so it opens in every player.

    ``VideoSink`` writes MPEG-4 Part 2 via OpenCV's default ``mp4v`` codec, which browsers
    accept poorly; system ``ffmpeg`` produces the widely supported H.264.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg not found: keeping the 'mp4v' codec; extracted data may not play in browsers.")
        return

    fixed_path = str(Path(video_path).with_suffix("")) + ".h264.mp4"
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-i", video_path,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                fixed_path,
            ],
            check=True,
        )
        Path(fixed_path).replace(video_path)
    except Exception as exc:
        print(f"Warning: ffmpeg transcoding failed: {exc}")
        if Path(fixed_path).exists():
            Path(fixed_path).unlink()


def read_plate_text(ocr, frame: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[str, float]:
    """OCR one padded plate crop; return the best cleaned alphanumeric string and its score."""
    crop = frame[max(0, y0 - 1): y1 + 1, max(0, x0 - 1): x1 + 1]
    if crop.size == 0:
        return "", 0.0
    results, _ = ocr(crop)
    if not results:
        return "", 0.0

    best_text, best_score = "", 0.0
    for _, text, score in results:
        cleaned = "".join(ch for ch in str(text).upper() if ch.isalnum())
        if MIN_PLATE_TEXT_LENGTH <= len(cleaned) <= MAX_PLATE_TEXT_LENGTH and score >= MIN_OCR_SCORE:
            if score > best_score:
                best_text, best_score = cleaned, score
    return best_text, best_score


def match_plate_to_vehicle(plate_xyxy: np.ndarray, vehicles_xyxy: np.ndarray) -> int:
    """Return the vehicle row index whose box contains the plate center (largest wins)."""
    if len(vehicles_xyxy) == 0:
        return -1
    cx = (plate_xyxy[0] + plate_xyxy[2]) / 2
    cy = (plate_xyxy[1] + plate_xyxy[3]) / 2
    inside = (
        (vehicles_xyxy[:, 0] <= cx) & (cx <= vehicles_xyxy[:, 2])
        & (vehicles_xyxy[:, 1] <= cy) & (cy <= vehicles_xyxy[:, 3])
    )
    if not inside.any():
        return -1
    areas = (vehicles_xyxy[:, 2] - vehicles_xyxy[:, 0]) * (vehicles_xyxy[:, 3] - vehicles_xyxy[:, 1])
    areas[~inside] = -1
    return int(np.argmax(areas))


def write_summary(path: str, rows: list[list]) -> None:
    """Write the per-vehicle summary CSV with a header row."""
    with open(path, mode="w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["tracker_id", "class", "speed_kmh_approx", "plate_text", "direction", "first_frame", "last_frame", "frames"])
        writer.writerows(rows)


def play_video_popup(video_path: str) -> None:
    """Launch the annotated video in a standalone popup player window."""
    import os
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("No graphical display detected ($DISPLAY / $WAYLAND_DISPLAY not set); skipping popup player.")
        return

    resolved_path = str(Path(video_path).resolve())
    title = f"Traffic Analysis - {Path(video_path).name}"

    players = [
        ["mpv", f"--title={title}", "--loop=inf", resolved_path],
        ["ffplay", "-window_title", title, "-loop", "0", resolved_path],
        ["xdg-open", resolved_path],
    ]

    for cmd in players:
        if shutil.which(cmd[0]):
            print(f"Opening output in popup window with {cmd[0]}...")
            try:
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return
            except Exception as exc:
                print(f"Failed to launch {cmd[0]}: {exc}")
                continue
    print("No graphical video player found (tried mpv, ffplay, xdg-open).")


def run_pipeline(input_path: str, output_path: str, model_path: str, plate_model_path: str,
                 summary_path: str, plates: bool, speed: bool,
                 plane: np.ndarray | None = None, plane_width_m: float = 11.25,
                 plane_height_m: float = 60.0, line_offset: float | None = 0.75,
                 line_tilt: float = 0.0, play: bool = True) -> None:
    """Detect + track + speed + plate-OCR the video and write annotated output + summary."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    except Exception:  # noqa: BLE001 — OCR must never kill detection; degrade to "-".
        ocr = None

    view_transformer = None
    if plane is not None:
        # Metric rectangle: across = --plane-width m at BOTH depths (trav lanes are
        # parallel in ground truth), along = --plane-height m between the two depths.
        target = np.array(
            [[0.0, 0.0], [plane_width_m, 0.0], [plane_width_m, plane_height_m], [0.0, plane_height_m]],
        )
        view_transformer = ViewTransformer(plane, target)

    model = YOLO(model_path)
    plate_model = YOLO(plate_model_path) if plates else None

    video_info = sv.VideoInfo.from_video_path(input_path)

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    # Cap text scale: on distant/small-vehicle footage the "optimal" scale dwarfs the boxes.
    text_scale = 0.6 * sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
    vehicle_label_annotator = sv.LabelAnnotator(
        text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK,
    )
    plate_label_annotator = sv.LabelAnnotator(
        text_scale=0.35, text_thickness=1, text_position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.INDEX,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness, trace_length=int(video_info.fps * 1.5), position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK,
    )

    line_zone = None
    line_zone_annotator = None
    if line_offset is not None:
        w, h = video_info.resolution_wh
        y1 = int(h * np.clip(line_offset - line_tilt / 2.0, 0.05, 0.95))
        y2 = int(h * np.clip(line_offset + line_tilt / 2.0, 0.05, 0.95))
        line_zone = sv.LineZone(
            start=sv.Point(0, y1),
            end=sv.Point(w, y2),
            triggering_anchors=[sv.Position.CENTER],
        )
        line_zone_annotator = sv.LineZoneAnnotator(
            thickness=thickness,
            text_scale=0.5,
            text_thickness=1,
            color=sv.Color.from_hex("#38BDF8"),
        )

    byte_track = ByteTrackTracker(
        frame_rate=video_info.fps, track_activation_threshold=CONFIDENCE_THRESHOLD,
    )
    smoother = sv.DetectionsSmoother(length=3)
    book = VehicleBook(video_info.fps, maxlen=int(video_info.fps), metric=view_transformer is not None)

    summary_rows: list[list] = []
    with sv.VideoSink(output_path, video_info) as sink:
        for frame_index, frame in enumerate(
            sv.get_video_frames_generator(source_path=input_path)
        ):
            result = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = byte_track.update(detections)
            detections = detections[detections.tracker_id != -1]
            detections = smoother.update_with_detections(detections)

            is_vehicle = np.isin(detections.class_id, list(VEHICLE_CLASS_IDS))
            vehicles = detections[is_vehicle]

            if line_zone is not None:
                crossed_in, crossed_out = line_zone.trigger(vehicles)
                for idx, c_in in enumerate(crossed_in):
                    if c_in:
                        book.mark_crossing(int(vehicles.tracker_id[idx]), "IN")
                for idx, c_out in enumerate(crossed_out):
                    if c_out:
                        book.mark_crossing(int(vehicles.tracker_id[idx]), "OUT")

            if speed:
                anchors = vehicles.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                if view_transformer is not None:
                    anchors = view_transformer.transform_points(anchors)
                for i, tracker_id in enumerate(vehicles.tracker_id):
                    class_id = int(vehicles.class_id[i])
                    box = vehicles.xyxy[i]
                    book.record(tracker_id, frame_index, anchors[i], float(box[2] - box[0]), class_id)

            plate_boxes, plate_texts = [], []
            if plate_model is not None and ocr is not None and frame_index % OCR_EVERY_N_FRAMES == 0:
                plate_result = plate_model(frame, conf=0.35, iou=0.7, verbose=False)[0]
                plate_detections = sv.Detections.from_ultralytics(plate_result)
                vehicles_xyxy = vehicles.xyxy if len(vehicles) else np.empty((0, 4), dtype=float)
                for plate_xyxy in plate_detections.xyxy.astype(int):
                    vehicle_index = match_plate_to_vehicle(plate_xyxy, vehicles_xyxy)
                    plate_boxes.append(plate_xyxy)
                    plate_texts.append("")
                    if vehicle_index < 0:
                        continue
                    text, _ = read_plate_text(ocr, frame, *plate_xyxy)
                    tracker_id = int(vehicles.tracker_id[vehicle_index])
                    if text:
                        book.vote_plate(tracker_id, text)
                        plate_texts[-1] = book.winner_plate(tracker_id)

            annotated = frame.copy()
            annotated = trace_annotator.annotate(scene=annotated, detections=vehicles)
            if len(detections):
                annotated = box_annotator.annotate(scene=annotated, detections=detections)

            vehicle_labels = []
            speed_is_approx = view_transformer is None
            for tracker_id, class_id in zip(vehicles.tracker_id, vehicles.class_id):
                label = f"#{int(tracker_id)} {model.names[int(class_id)]}"
                if speed:
                    real_width = VEHICLE_CLASS_IDS[int(class_id)]
                    speed_kmh = book.speed_kmh(int(tracker_id), real_width)
                    if speed_kmh is not None:
                        label += (" ≈" if speed_is_approx else " ") + f"{speed_kmh:.0f} km/h"
                    if plates and book.winner_plate(int(tracker_id)):
                        label += f" [{book.winner_plate(int(tracker_id))}]"
                vehicle_labels.append(label)
            if len(vehicles):
                annotated = vehicle_label_annotator.annotate(
                    scene=annotated, detections=vehicles, labels=vehicle_labels,
                )
            if plate_boxes:
                plate_dets = sv.Detections(
                    xyxy=np.array(plate_boxes, dtype=np.float32),
                )
                annotated = plate_label_annotator.annotate(
                    scene=annotated, detections=plate_dets, labels=plate_texts,
                )
            if line_zone is not None:
                annotated = line_zone_annotator.annotate(frame=annotated, line_counter=line_zone)
            sink.write_frame(annotated)

    summary_rows = book.rows_for_summary(model.names)
    write_summary(summary_path, summary_rows)
    transcode_to_h264(output_path)
    print(f"Done. Annotated video saved to: {output_path}")
    print(f"Per-vehicle summary saved to:   {summary_path}")
    if line_zone is not None:
        print(f"Directional Counts: In={line_zone.in_count}, Out={line_zone.out_count}")
    if play:
        play_video_popup(output_path)


def resolve_io_paths(
    input_name: str,
    output_path: str | None = None,
    summary_path: str | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve input video path and compute default output and summary paths in data/.

    Accepts bare names (e.g. 'input', 'input.mp4') or paths (e.g. 'data/input.mp4').
    Defaults output to 'data/output_<name>.mp4' and summary to 'data/summary_<name>.csv'.
    """
    candidate = Path(input_name)
    data_dir = Path("data")

    if candidate.is_file():
        resolved_input = candidate
    elif (data_dir / candidate.name).is_file():
        resolved_input = data_dir / candidate.name
    elif candidate.with_suffix(".mp4").is_file():
        resolved_input = candidate.with_suffix(".mp4")
    elif (data_dir / f"{candidate.stem}.mp4").is_file():
        resolved_input = data_dir / f"{candidate.stem}.mp4"
    else:
        raise FileNotFoundError(
            f"Input video not found: tried '{candidate}', 'data/{candidate.name}', "
            f"and 'data/{candidate.stem}.mp4'."
        )

    stem = resolved_input.stem
    resolved_output = Path(output_path) if output_path else data_dir / f"output_{stem}.mp4"
    resolved_summary = Path(summary_path) if summary_path else data_dir / f"summary_{stem}.csv"

    return resolved_input, resolved_output, resolved_summary


def main() -> int:
    """Parse CLI args, validate paths, run the traffic pipeline."""
    parser = argparse.ArgumentParser(description="Traffic detection, tracking, speed and number plates")
    parser.add_argument(
        "input", nargs="?", default="input",
        help="input video name or path (e.g. 'input', 'input2', 'data/input.mp4'; default: %(default)s)",
    )
    parser.add_argument(
        "output", nargs="?", default=None,
        help="output video path (default: data/output_<name>.mp4)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="COCO detection model (default: %(default)s)")
    parser.add_argument("--plate-model", default=DEFAULT_PLATE_MODEL, help="plate detector weights (default: %(default)s)")
    parser.add_argument(
        "--summary", default=None,
        help="CSV summary output path (default: data/summary_<name>.csv)",
    )
    parser.add_argument("--no-plates", action="store_true", help="skip plate detection + OCR")
    parser.add_argument("--no-speed", action="store_true", help="skip speed estimation")
    parser.add_argument("--no-play", action="store_true", help="skip automatically playing output video in a popup window")
    parser.add_argument("--line-offset", type=float, default=0.75, help="horizontal counting line offset fraction 0.0-1.0 (default: 0.75)")
    parser.add_argument("--line-tilt", type=float, default=0.0, help="line tilt fraction (e.g. -0.2 to +0.2; default: 0.0)")
    parser.add_argument("--no-line", action="store_true", help="disable directional counting line")
    parser.add_argument(
        "--plane", type=parse_plane, default=None,
        help='ground calibration: 4 pixels "x1,y1 x2,y2 x3,y3 x4,y4" (near-left near-right far-right far-left)',
    )
    parser.add_argument("--plane-width", type=float, default=11.25, help="metric across-road width of the --plane quad (default: %(default)s)")
    parser.add_argument("--plane-height", type=float, default=60.0, help="metric along-road length of the --plane quad (default: %(default)s)")
    args = parser.parse_args()

    try:
        input_video, output_video, summary_file = resolve_io_paths(args.input, args.output, args.summary)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    if output_video.resolve() == input_video.resolve():
        print("Output path must differ from input path.")
        return 1
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found; YouTube-style playback may be affected.")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    run_pipeline(
        str(input_video), str(output_video), args.model, args.plate_model, str(summary_file),
        plates=not args.no_plates, speed=not args.no_speed,
        plane=args.plane, plane_width_m=args.plane_width, plane_height_m=args.plane_height,
        line_offset=None if args.no_line else args.line_offset,
        line_tilt=args.line_tilt,
        play=not args.no_play,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
