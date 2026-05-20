"""
rtsp_service.py - RTSP realtime parking counter with Telegram MP4 notifications.

What it does:
- Reads RTSP stream (keeps only latest frame; no backlog).
- Runs OpenVINO object detection + tracking (ParkingCounter) to detect gate events: IN / OUT.
- Maintains persistent "free parking spots" state (based on events) across restarts.
- On each detected event sends ONE Telegram message: video clip (MP4/H.264) + caption (current free spots).

Key env vars (.env):
- VIDEO_SOURCE=rtsp://...
- MODEL_XML=path/to/model.xml
- TELEGRAM_BOT_TOKEN=...
- TELEGRAM_CHAT_ID=...

Parking state:
- PARKING_TOTAL_SPOTS=20
- PARKING_INITIAL_FREE_SPOTS=12          # used only if no state file exists yet
- PARKING_STATE_PATH=parking_state.json  # optional
- PARKING_DB_PATH=parking_stats.db       # optional (stores IN/OUT history)

Video:
- PRE_EVENT_SEC=5
- POST_EVENT_SEC=5
- BUFFER_FPS=10
- VIDEO_FPS=10
- VIDEO_MAX_WIDTH=960
- VIDEO_CRF=23
- VIDEO_PRESET=veryfast

Zone polygon:
- ZONE_POLYGON_JSON=[ [x,y], ... ] or
- zone_polygon.json file in project root

Note:
OpenCV/FFmpeg can be noisy on stderr. We redirect native stderr before importing cv2.
"""

from __future__ import annotations

import json
import os
import shutil
import datetime
import subprocess
import sys
import threading
import time
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple

from dotenv import load_dotenv

# ---- Keep Python exceptions visible even if stderr is redirected
def _excepthook(exctype, value, tb):
    import traceback
    traceback.print_exception(exctype, value, tb, file=sys.stdout)

sys.excepthook = _excepthook


# ---- Redirect native FFmpeg spam (printed to stderr) BEFORE importing cv2
def _redirect_stderr() -> None:
    mode = os.getenv("FFMPEG_STDERR_MODE", "file").strip().lower()
    if mode in ("off", "0", "false", "no"):
        return
    try:
        if mode == "nul":
            target = "NUL" if os.name == "nt" else os.devnull
            f = open(target, "w", buffering=1)
        else:
            # default: file
            f = open("ffmpeg_stderr.log", "a", buffering=1, encoding="utf-8", errors="ignore")
        os.dup2(f.fileno(), 2)
    except Exception:
        # best-effort only
        return


_redirect_stderr()

# Reduce OpenCV logs (best-effort; depends on build)
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402

from parking_counter import ParkingCounter, CounterConfig  # noqa: E402
from openvino_detector import OpenVINOObjectDetector, OVDetectorConfig  # noqa: E402
from storage import ParkingStore, ParkingState  # noqa: E402


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def env_bool(name: str, default: str = "0") -> bool:
    return bool(int(os.getenv(name, default).strip() or "0"))


def env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default).strip())


def env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default).strip())


@dataclass(frozen=True)
class ServiceConfig:
    video_source: str
    model_xml: str

    telegram_token: str
    telegram_chat_id: str

    show: bool
    draw: bool

    # Clip handling
    clip_annotate: bool   # overlay zone + detections for Telegram clip
    save_raw_clip: bool   # save clean (no overlays) clip to CLIPS_DIR


    detect_fps: float
    buffer_fps: float
    pre_event_sec: float
    post_event_sec: float
    health_every_sec: float

    num_zones: int
    in_direction: str

    conf_threshold: float
    min_box_area: int
    min_box_width: int
    min_box_height: int
    max_box_area: Optional[int]
    lost_track_buffer_sec: float
    min_track_frames: int

    clips_dir: Path

    video_fps: float
    video_max_width: int
    video_crf: str
    video_preset: str

    parking_total_spots: int
    parking_initial_free_spots: Optional[int]


def load_config() -> ServiceConfig:
    load_dotenv()

    video_source = os.getenv("VIDEO_SOURCE", "").strip()
    model_xml = os.getenv("MODEL_XML", "").strip()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    show = env_bool("SHOW", "0")
    draw = env_bool("DRAW", "1")

    clip_annotate = env_bool("CLIP_ANNOTATE", "1")
    save_raw_clip = env_bool("SAVE_RAW_CLIP", "1")

    detect_fps = env_float("DETECT_FPS", "10")
    buffer_fps = env_float("BUFFER_FPS", "10")
    pre_event_sec = env_float("PRE_EVENT_SEC", "5")
    post_event_sec = env_float("POST_EVENT_SEC", "5")
    health_every_sec = env_float("HEALTH_EVERY_SEC", "60")

    num_zones = env_int("NUM_ZONES", "5")
    in_direction = os.getenv("IN_DIRECTION", "bottom_to_top").strip()

    conf_threshold = env_float("CONF_THRESHOLD", "0.45")
    min_box_area = env_int("MIN_BOX_AREA", "2200")
    min_box_width = env_int("MIN_BOX_WIDTH", "55")
    min_box_height = env_int("MIN_BOX_HEIGHT", "45")
    max_box_area_raw = os.getenv("MAX_BOX_AREA", "0").strip()
    max_box_area = int(max_box_area_raw) if max_box_area_raw and int(max_box_area_raw) > 0 else None
    lost_track_buffer_sec = env_float("LOST_TRACK_BUFFER_SEC", "8")
    min_track_frames = env_int("MIN_TRACK_FRAMES", "6")

    clips_dir = Path(os.getenv("CLIPS_DIR", "clips").strip())
    clips_dir.mkdir(parents=True, exist_ok=True)

    video_fps = env_float("VIDEO_FPS", str(buffer_fps))
    video_max_width = env_int("VIDEO_MAX_WIDTH", "960")
    video_crf = os.getenv("VIDEO_CRF", "23").strip()
    video_preset = os.getenv("VIDEO_PRESET", "veryfast").strip()

    # Backward-compatible: support TOTAL_SPOTS if PARKING_TOTAL_SPOTS is not set
    parking_total_spots = env_int("PARKING_TOTAL_SPOTS", "0")
    if parking_total_spots <= 0:
        try:
            parking_total_spots = int(os.getenv("TOTAL_SPOTS", "0").strip() or "0")
        except Exception:
            parking_total_spots = 0
    init_free_raw = os.getenv("PARKING_INITIAL_FREE_SPOTS", "").strip()
    parking_initial_free_spots = int(init_free_raw) if init_free_raw else None

    return ServiceConfig(
        video_source=video_source,
        model_xml=model_xml,
        telegram_token=token,
        telegram_chat_id=chat_id,
        show=show,
        draw=draw,
        clip_annotate=clip_annotate,
        save_raw_clip=save_raw_clip,
        detect_fps=detect_fps,
        buffer_fps=buffer_fps,
        pre_event_sec=pre_event_sec,
        post_event_sec=post_event_sec,
        health_every_sec=health_every_sec,
        num_zones=num_zones,
        in_direction=in_direction,
        conf_threshold=conf_threshold,
        min_box_area=min_box_area,
        min_box_width=min_box_width,
        min_box_height=min_box_height,
        max_box_area=max_box_area,
        lost_track_buffer_sec=lost_track_buffer_sec,
        min_track_frames=min_track_frames,
        clips_dir=clips_dir,
        video_fps=video_fps,
        video_max_width=video_max_width,
        video_crf=video_crf,
        video_preset=video_preset,
        parking_total_spots=parking_total_spots,
        parking_initial_free_spots=parking_initial_free_spots,
    )


# ------------------ Telegram ------------------

def _tg_url(token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{token}/{method}"


def tg_send_text(token: str, chat_id: str, text: str) -> None:
    if not token or not chat_id:
        return
    try:
        requests.post(_tg_url(token, "sendMessage"), json={"chat_id": chat_id, "text": text}, timeout=15)
    except Exception as e:
        log(f"telegram sendMessage failed: {e}")


def tg_send_video(token: str, chat_id: str, mp4_path: str, caption: str = "") -> None:
    """
    Send MP4 as Telegram video with caption (ONE message).
    """
    if not token or not chat_id:
        return

    p = Path(mp4_path)
    if not p.exists():
        log(f"telegram sendVideo: file not found: {p}")
        return

    try:
        with p.open("rb") as f:
            files = {"video": (p.name, f, "video/mp4")}
            data = {
                "chat_id": chat_id,
                "caption": caption[:1024],
                "supports_streaming": True,
            }
            r = requests.post(_tg_url(token, "sendVideo"), data=data, files=files, timeout=60)
            if not r.ok:
                log(f"telegram sendVideo failed: {r.status_code} {r.text[:500]}")
    except Exception as e:
        log(f"telegram sendVideo failed: {e}")


# ------------------ Zone polygon ------------------

DEFAULT_ZONE_POLYGON = np.array(
    [
        [856, 1055],
        [1150, 1078],
        [1153, 761],
        [875, 742],
    ],
    dtype=np.int32,
)


def load_zone_polygon() -> np.ndarray:
    """
    Priority:
    1) ZONE_POLYGON_JSON env
    2) ./zone_polygon.json file
    3) DEFAULT_ZONE_POLYGON
    """
    raw = os.getenv("ZONE_POLYGON_JSON", "").strip()
    if raw:
        try:
            pts = json.loads(raw)
            return np.array(pts, dtype=np.int32)
        except Exception:
            log("Invalid ZONE_POLYGON_JSON; falling back to zone_polygon.json / default")

    p = Path("zone_polygon.json")
    if p.exists():
        try:
            pts = json.loads(p.read_text(encoding="utf-8"))
            return np.array(pts, dtype=np.int32)
        except Exception:
            log("Invalid zone_polygon.json; falling back to default polygon")

    return DEFAULT_ZONE_POLYGON.copy()


def _poly_in_bounds(poly: np.ndarray, w: int, h: int) -> bool:
    xs = poly[:, 0]
    ys = poly[:, 1]
    return (xs.min() >= 0) and (ys.min() >= 0) and (xs.max() < w) and (ys.max() < h)


# ------------------ Clip annotation (zone + detections) ------------------

class ClipAnnotator:
    """
    Annotates frames for the Telegram event clip:
    - draws the configured zone polygon
    - draws detected vehicle bounding boxes (OpenVINO detector, ROI-cropped for speed)

    Notes:
    - This is intentionally *detector-only* (no tracking), so it won't affect counting logic.
    - Uses the same MODEL_XML as the main counter.
    """

    def __init__(self, zone_polygon: np.ndarray, model_xml: str, device: str = "CPU", conf_threshold: float = 0.35,
                 min_box_area: int = 2500, min_box_width: int = 60, min_box_height: int = 45, roi_pad: int = 10):
        self.zone_polygon = zone_polygon.astype(np.int32)
        self.detector = OpenVINOObjectDetector(
            OVDetectorConfig(
                model_xml=model_xml,
                device=device,
                conf_threshold=conf_threshold,
                allowed_labels=(0,),  # vehicles only for clip overlay
                min_box_area=min_box_area,
                min_box_width=min_box_width,
                min_box_height=min_box_height,
            )
        )
        self.roi_pad = int(roi_pad)

    def _roi_from_polygon(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        xs = self.zone_polygon[:, 0]
        ys = self.zone_polygon[:, 1]
        x1 = max(0, int(xs.min()) - self.roi_pad)
        y1 = max(0, int(ys.min()) - self.roi_pad)
        x2 = min(w, int(xs.max()) + self.roi_pad)
        y2 = min(h, int(ys.max()) + self.roi_pad)
        # fallback if something is off
        if x2 <= x1 or y2 <= y1:
            return 0, 0, w, h
        return x1, y1, x2, y2

    @staticmethod
    def _anchor_inside(poly: np.ndarray, x: int, y: int) -> bool:
        # pointPolygonTest expects float32 polygon
        return cv2.pointPolygonTest(poly.astype(np.float32), (float(x), float(y)), False) >= 0

    def annotate(self, frame_bgr: np.ndarray) -> np.ndarray:
        out = frame_bgr.copy()

        # 1) zone polygon
        cv2.polylines(out, [self.zone_polygon.reshape((-1, 1, 2))], True, (0, 255, 255), 2)

        # 2) detections (within ROI)
        x1, y1, x2, y2 = self._roi_from_polygon(out)
        roi = out[y1:y2, x1:x2]
        if roi.size == 0:
            return out

        xyxy, conf, lab = self.detector.infer(roi)
        if xyxy.shape[0] == 0:
            return out

        h_in, w_in = self.detector.input_hw
        roi_h, roi_w = roi.shape[:2]
        sx = roi_w / float(w_in)
        sy = roi_h / float(h_in)

        for (bx1, by1, bx2, by2), c in zip(xyxy, conf):
            # map from model input coords -> ROI coords -> full frame coords
            fx1 = int(round(bx1 * sx)) + x1
            fy1 = int(round(by1 * sy)) + y1
            fx2 = int(round(bx2 * sx)) + x1
            fy2 = int(round(by2 * sy)) + y1

            # sanity clamp
            fx1 = max(0, min(fx1, out.shape[1] - 1))
            fx2 = max(0, min(fx2, out.shape[1] - 1))
            fy1 = max(0, min(fy1, out.shape[0] - 1))
            fy2 = max(0, min(fy2, out.shape[0] - 1))
            if fx2 <= fx1 or fy2 <= fy1:
                continue

            # keep only boxes whose bottom-center anchor is inside polygon
            ax = int((fx1 + fx2) / 2)
            ay = int(fy2)
            if not self._anchor_inside(self.zone_polygon, ax, ay):
                continue

            cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
            cv2.circle(out, (ax, ay), 3, (0, 255, 0), -1)
            cv2.putText(out, f"car {float(c):.2f}", (fx1, max(0, fy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return out


# ------------------ Video buffering ------------------

@dataclass
class BufferedFrame:
    ts: float
    frame: np.ndarray


class RingBuffer:
    def __init__(self, seconds: float, fps: float):
        self.maxlen = max(1, int(round(seconds * fps)))
        self.buf: Deque[BufferedFrame] = deque(maxlen=self.maxlen)
        self._lock = threading.Lock()

    def push(self, ts: float, frame: np.ndarray) -> None:
        with self._lock:
            self.buf.append(BufferedFrame(ts=ts, frame=frame))

    def get_window(self, start_ts: float, end_ts: float) -> List[BufferedFrame]:
        with self._lock:
            return [bf for bf in list(self.buf) if start_ts <= bf.ts <= end_ts]


class LatestFrameGrabber:
    """
    Continuously grab RTSP frames and keep only the latest frame.
    This avoids backlog buildup when detector FPS is lower than stream FPS.
    """

    def __init__(self, source: str):
        self.source = source
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts: float = 0.0
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._alive = False

    def start(self):
        self._thr.start()

    def stop(self):
        self._stop.set()
        self._thr.join(timeout=2.0)

    def get_latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._ts

    @property
    def alive(self) -> bool:
        return self._alive

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _run(self):
        cap = self._open()
        self._alive = cap.isOpened()

        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                time.sleep(0.5)
                cap = self._open()
                self._alive = cap.isOpened()
                continue

            ok = cap.grab()
            if not ok:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                self._alive = False
                time.sleep(0.5)
                continue

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue

            ts = time.monotonic()
            with self._lock:
                self._frame = frame
                self._ts = ts
            self._alive = True

        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


# ------------------ MP4 encoding ------------------

def _downscale_max_width(frame: np.ndarray, max_w: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    scale = max_w / float(w)
    new_w = max_w
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def find_ffmpeg() -> Optional[str]:
    # 1) explicit env
    p = os.getenv("FFMPEG_BIN", "").strip()
    if p and Path(p).exists():
        return p
    # 2) PATH
    p = shutil.which("ffmpeg")
    if p:
        return p
    # 3) imageio-ffmpeg
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore
        p = get_ffmpeg_exe()
        if p and Path(p).exists():
            return p
    except Exception:
        pass
    return None


def write_mp4_h264(frames_bgr: List[np.ndarray], out_path: str, fps: float, max_w: int, crf: str, preset: str) -> bool:
    """
    Encode frames (BGR) into MP4 (H.264, yuv420p, faststart). Telegram-friendly.
    """
    if not frames_bgr:
        return False

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        log("ffmpeg not found. Install ffmpeg or set FFMPEG_BIN.")
        return False

    frames = [_downscale_max_width(fr, max_w) for fr in frames_bgr]
    h, w = frames[0].shape[:2]

    cmd = [
        ffmpeg,
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ]

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log(f"ffmpeg start failed: {e}")
        return False

    try:
        for fr in frames:
            if fr.shape[0] != h or fr.shape[1] != w:
                fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
            assert proc.stdin is not None
            proc.stdin.write(fr.tobytes())
        assert proc.stdin is not None
        proc.stdin.close()
        rc = proc.wait(timeout=120)
        if rc != 0:
            log(f"ffmpeg exited with code {rc}")
            return False
    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        log(f"ffmpeg encoding failed: {e}")
        return False

    p = Path(out_path)
    return p.exists() and p.stat().st_size > 0


# ------------------ Service loop ------------------

@dataclass(frozen=True)
class PendingEvent:
    ev: str
    ts: float
    state: ParkingState


def format_caption(event: str, state: ParkingState) -> str:
    """Build Telegram caption for a single event clip (video caption).

    Requirements:
    - Must include which event happened (IN/OUT) as 'В'їзд/Виїзд'.
    - Must emphasize current free spots.
    - Do NOT include explicit timestamp (Telegram message already has it).
    """
    event_ua = {"IN": "В'їзд", "OUT": "Виїзд"}.get(event, event)
    return (
        f"Подія: {event_ua}\n"
        f"Вільні місця: {state.free_spots}/{state.total_spots}\n"
    )



def format_status(state: ParkingState) -> str:
    """Short status text for non-event messages (startup/health)."""
    return (
        f"Вільні місця: {state.free_spots}/{state.total_spots}"
    )

def main() -> None:
    cfg = load_config()

    if not cfg.video_source:
        raise SystemExit("VIDEO_SOURCE is empty in .env")
    if not cfg.model_xml:
        raise SystemExit("MODEL_XML is empty in .env (path to OpenVINO .xml)")
    if cfg.parking_total_spots <= 0:
        raise SystemExit("PARKING_TOTAL_SPOTS is empty/0 in .env (total parking capacity)")

    store = ParkingStore()
    state = store.load_or_init_state(cfg.parking_total_spots, cfg.parking_initial_free_spots)

    grabber = LatestFrameGrabber(cfg.video_source)
    grabber.start()

    # wait for first frame
    frame: Optional[np.ndarray] = None
    fts = 0.0
    for _ in range(200):
        frame, fts = grabber.get_latest()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        raise SystemExit("No frames received from RTSP")

    H, W = frame.shape[:2]
    poly = load_zone_polygon()

    log(f"stream frame_size={W}x{H}")
    log(f"zone_polygon={poly.tolist()}")

    if not _poly_in_bounds(poly, W, H):
        raise SystemExit("ZONE_POLYGON is out of bounds. Recalibrate on the same RTSP stream (main/sub must match).")

    counter_cfg = CounterConfig(
        model_xml=cfg.model_xml,
        frame_rate=float(cfg.detect_fps),
        num_zones=int(cfg.num_zones),
        in_direction=str(cfg.in_direction),
        conf_threshold=float(cfg.conf_threshold),
        min_box_area=int(cfg.min_box_area),
        min_box_width=int(cfg.min_box_width),
        min_box_height=int(cfg.min_box_height),
        max_box_area=cfg.max_box_area,
        lost_track_buffer_sec=float(cfg.lost_track_buffer_sec),
        min_track_frames=int(cfg.min_track_frames),
        draw=bool(cfg.draw),
    )
    counter = ParkingCounter(poly, counter_cfg)

    clip_annotator = ClipAnnotator(
        zone_polygon=poly,
        model_xml=cfg.model_xml,
        device=counter_cfg.device,
        conf_threshold=counter_cfg.conf_threshold,
        min_box_area=counter_cfg.min_box_area,
        min_box_width=counter_cfg.min_box_width,
        min_box_height=counter_cfg.min_box_height,
    )

    ring = RingBuffer(seconds=(cfg.pre_event_sec + cfg.post_event_sec + 2.0), fps=cfg.buffer_fps)

    last_buffer = 0.0
    last_detect = 0.0
    last_health = 0.0
    pending: Deque[PendingEvent] = deque()

    # daily reset tracking
    _last_reset_day: Optional[int] = None

    if cfg.telegram_token and cfg.telegram_chat_id:
        tg_send_text(cfg.telegram_token, cfg.telegram_chat_id, f"RTSP сервіс запущено.\n{format_status(state)}")

    win = "RTSP (OpenVINO) - press q"

    while True:
        now = time.monotonic()

        # --- Daily reset at 00:01 ---
        _wall = datetime.datetime.now()
        if _wall.hour == 0 and _wall.minute == 1:
            today = _wall.toordinal()
            if _last_reset_day != today:
                _last_reset_day = today
                new_st = ParkingState(
                    total_spots=state.total_spots,
                    free_spots=0,
                    updated_ts=time.time(),
                )
                store.save_state(new_st)
                state = new_st
                log(f"[DAILY RESET] All spots marked busy: {state.free_spots}/{state.total_spots}")
                if cfg.telegram_token and cfg.telegram_chat_id:
                    tg_send_text(
                        cfg.telegram_token,
                        cfg.telegram_chat_id,
                        f"🔄 Щоденне скидання (00:01) — всі місця зайняті\n{format_status(state)}",
                    )
        frame, fts = grabber.get_latest()
        if frame is None:
            time.sleep(0.01)
            continue

        # buffer frames for clip creation
        if (now - last_buffer) >= (1.0 / max(0.1, cfg.buffer_fps)):
            ring.push(now, frame.copy())
            last_buffer = now

        # detection tick
        if (now - last_detect) >= (1.0 / max(0.1, cfg.detect_fps)):
            last_detect = now

            if cfg.show:
                events, dbg = counter.process_frame_debug(frame, ts_sec=now)
                cv2.imshow(win, dbg.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                events = counter.process_frame(frame, ts_sec=now)

            # apply each event to persistent state
            if events:
                for ev in events:
                    state = store.apply_event(state, ev, ts=time.time())
                    pending.append(PendingEvent(ev=ev, ts=now, state=state))
                    log(f"EVENT {ev} -> free={state.free_spots}/{state.total_spots}")

        # send ready pending events (after post window)
        while pending and now >= (pending[0].ts + cfg.post_event_sec):
            pe = pending.popleft()

            start_ts = pe.ts - cfg.pre_event_sec
            end_ts = pe.ts + cfg.post_event_sec
            frames = [bf.frame for bf in ring.get_window(start_ts, end_ts)]

            if not frames:
                log("No buffered frames for event clip (skipped).")
                continue

            # downsample to target video FPS
            stride = max(1, int(round(cfg.buffer_fps / max(1.0, cfg.video_fps))))
            vid_frames = frames[::stride]

            mp4_name = f"event_{pe.ev}_{int(pe.ts)}.mp4"
            raw_path = str(cfg.clips_dir / mp4_name)

            # 1) Save RAW (clean) clip for future tests (no overlays)
            ok_raw = False
            if cfg.save_raw_clip:
                ok_raw = write_mp4_h264(
                    vid_frames,
                    raw_path,
                    fps=min(cfg.video_fps, cfg.buffer_fps),
                    max_w=cfg.video_max_width,
                    crf=cfg.video_crf,
                    preset=cfg.video_preset,
                )
                if not ok_raw:
                    log("RAW MP4 was not created (ffmpeg missing or failed).")

            # 2) Prepare clip for Telegram (annotated or raw)
            send_path = raw_path
            ok_send = ok_raw
            cleanup_send_file = False

            if cfg.clip_annotate:
                # create annotated clip in a temp file so CLIPS_DIR stays clean
                try:
                    send_frames = [clip_annotator.annotate(fr) for fr in vid_frames]
                except Exception as e:
                    log(f"clip annotation failed: {e}")
                    send_frames = vid_frames  # fallback to raw

                tmp = tempfile.NamedTemporaryFile(prefix="tg_", suffix=".mp4", delete=False)
                send_path = tmp.name
                tmp.close()
                cleanup_send_file = True

                ok_send = write_mp4_h264(
                    send_frames,
                    send_path,
                    fps=min(cfg.video_fps, cfg.buffer_fps),
                    max_w=cfg.video_max_width,
                    crf=cfg.video_crf,
                    preset=cfg.video_preset,
                )

            elif not cfg.save_raw_clip:
                # no RAW saved, but we still need to send something -> encode raw into temp file
                tmp = tempfile.NamedTemporaryFile(prefix="tg_", suffix=".mp4", delete=False)
                send_path = tmp.name
                tmp.close()
                cleanup_send_file = True

                ok_send = write_mp4_h264(
                    vid_frames,
                    send_path,
                    fps=min(cfg.video_fps, cfg.buffer_fps),
                    max_w=cfg.video_max_width,
                    crf=cfg.video_crf,
                    preset=cfg.video_preset,
                )

            if ok_send:
                caption = format_caption(pe.ev, pe.state)
                tg_send_video(cfg.telegram_token, cfg.telegram_chat_id, send_path, caption=caption)
            else:
                log("MP4 was not created (ffmpeg missing or failed).")

            if cleanup_send_file:
                try:
                    os.remove(send_path)
                except Exception:
                    pass

        # health log
        if (now - last_health) >= cfg.health_every_sec:
            last_health = now
            age = now - fts
            det_n = int(counter._last_det.xyxy.shape[0]) if hasattr(counter, "_last_det") else -1
            trk_n = int(len(counter._tracks)) if hasattr(counter, "_tracks") else -1
            log(f"health: alive={grabber.alive} age={age:.2f}s det={det_n} tracks={trk_n} detect_fps={cfg.detect_fps} free={state.free_spots}/{state.total_spots}")

        time.sleep(0.002)

    grabber.stop()
    if cfg.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
