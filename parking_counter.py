import os
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import cv2
import numpy as np
from dotenv import load_dotenv

import supervision as sv

from openvino_detector import OpenVINOObjectDetector, OVDetectorConfig


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two xyxy arrays."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    # intersection
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    # areas
    area_a = np.maximum(0.0, (ax2 - ax1)) * np.maximum(0.0, (ay2 - ay1))
    area_b = np.maximum(0.0, (bx2 - bx1)) * np.maximum(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


@dataclass
class CounterConfig:
    # --- OpenVINO model ---
    model_xml: str = ""  # if empty, read from env MODEL_XML
    device: str = "CPU"
    conf_threshold: float = 0.35
    # person-vehicle-bike-detection-2002 labels: 0=vehicle,1=person,2=bike
    allowed_labels: Tuple[int, ...] = (0, 1)

    # --- Tracker ---
    frame_rate: float = 10.0
    track_activation_threshold: float = 0.30
    minimum_matching_threshold: float = 0.80
    lost_track_buffer_sec: float = 8.0

    # --- Zone / counting ---
    num_zones: int = 5
    # "IN" is bottom (gate) -> top (yard). Flip if your camera is opposite.
    in_direction: str = "top_to_bottom"  # IN: upper->lower (towards gate)

    # How close to extremes we require
    low_zone_threshold: int = 2   # zones >= this considered "near gate" (for 5 zones: 3,4)
    high_zone_threshold: int = 2  # zones <= this considered "deep" (0,1)

    min_track_frames: int = 4        # require track to be present N frames
    min_zone_spread: int = 2         # require visiting at least this many zone indices

    # bbox size filters (in pixels in original frame)
    min_box_area: int = 2500
    min_box_width: int = 60
    min_box_height: int = 45
    max_box_area: Optional[int] = None

    # separate (looser) filters for person class, used only for suppression
    min_person_box_area: int = 900
    min_person_box_width: int = 20
    min_person_box_height: int = 45

    # Global debounce between events of the same type (protects against track ID flicker)
    global_event_cooldown_sec: float = 0.5

    # Person suppression: if a 'vehicle' box overlaps a person box strongly, drop it
    suppress_person_iou: float = 0.45

    # debug
    draw: bool = True


@dataclass
class FrameDebug:
    frame: np.ndarray
    detections: sv.Detections
    zones: List[np.ndarray]


class ParkingCounter:
    """Vehicle counter based on OpenVINO detector + ByteTrack.

    Interface compatibility goal:
      - process_frame(frame, ts_sec) -> list[str] events
      - process_frame_debug(...) -> (events, FrameDebug)

    Zone polygon is a quadrilateral defining the gate ROI.
    Counting is done by mapping the anchor point (bottom-center of bbox)
    into a rectified coordinate system (homography) and splitting Y into zones.
    """

    def __init__(self, zone_polygon: np.ndarray, cfg: Optional[CounterConfig] = None):
        load_dotenv()
        self.cfg = cfg or CounterConfig()

        if not self.cfg.model_xml:
            self.cfg.model_xml = os.getenv("MODEL_XML", "").strip()
        if not self.cfg.model_xml:
            raise ValueError("MODEL_XML is empty. Put it into .env or CounterConfig.model_xml")

        # detector
        det_cfg = OVDetectorConfig(
            model_xml=self.cfg.model_xml,
            device=self.cfg.device,
            conf_threshold=self.cfg.conf_threshold,
            allowed_labels=self.cfg.allowed_labels,
            min_box_area=self.cfg.min_box_area,
            min_box_width=self.cfg.min_box_width,
            min_box_height=self.cfg.min_box_height,
            max_box_area=self.cfg.max_box_area,
        )
        self.detector = OpenVINOObjectDetector(det_cfg)

        # tracker
        lost_buf = int(round(self.cfg.lost_track_buffer_sec * float(self.cfg.frame_rate)))
        self.tracker = sv.ByteTrack(
            track_activation_threshold=float(self.cfg.track_activation_threshold),
            minimum_matching_threshold=float(self.cfg.minimum_matching_threshold),
            lost_track_buffer=int(max(1, lost_buf)),
            frame_rate=int(max(1, round(self.cfg.frame_rate))),
        )

        # geometry
        self.zone_polygon = np.asarray(zone_polygon, dtype=np.int32)
        if self.zone_polygon.shape != (4, 2):
            raise ValueError("zone_polygon must be 4x2 (quadrilateral)")

        # Polygon points are provided as BL, BR, TR, TL (as in your rtsp_service.py)
        bl, br, tr, tl = self.zone_polygon.astype(np.float32)
        src = np.array([tl, tr, br, bl], dtype=np.float32)
        self._warp_w = 1000.0
        self._warp_h = 1000.0
        dst = np.array([[0, 0], [self._warp_w, 0], [self._warp_w, self._warp_h], [0, self._warp_h]], dtype=np.float32)
        self._H = cv2.getPerspectiveTransform(src, dst)

        # Pre-crop bbox (speeds up inference)
        x, y, w, h = cv2.boundingRect(self.zone_polygon)
        pad = 8
        self._roi_xyxy = (
            max(0, x - pad),
            max(0, y - pad),
            x + w + pad,
            y + h + pad,
        )

        # zones (in original image for drawing only)
        self._zones_poly = self._build_zone_polygons(self.zone_polygon, self.cfg.num_zones)

        # track state
        self._tracks: Dict[int, Dict] = {}
        self._last_det: sv.Detections = sv.Detections.empty()
        self._last_event_ts = {'IN': -1e9, 'OUT': -1e9}

    # ----------------------------
    # Geometry helpers
    # ----------------------------

    def _warp_point(self, x: float, y: float) -> Tuple[float, float]:
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    @staticmethod
    def _build_zone_polygons(poly: np.ndarray, n: int) -> List[np.ndarray]:
        """Split quad into N horizontal bands for drawing."""
        bl, br, tr, tl = poly.astype(np.float32)
        zones: List[np.ndarray] = []
        for i in range(n):
            a0 = i / n
            a1 = (i + 1) / n
            # interpolate along left edge (tl->bl) and right edge (tr->br)
            left0 = tl + (bl - tl) * a0
            left1 = tl + (bl - tl) * a1
            right0 = tr + (br - tr) * a0
            right1 = tr + (br - tr) * a1
            z = np.array([left1, right1, right0, left0], dtype=np.int32)
            zones.append(z)
        return zones

    def _zone_index(self, x: float, y: float) -> int:
        _, wy = self._warp_point(x, y)
        wy = max(0.0, min(wy, self._warp_h - 1.0))
        z = int((wy / self._warp_h) * self.cfg.num_zones)
        return int(max(0, min(self.cfg.num_zones - 1, z)))

    # ----------------------------
    # Detection
    # ----------------------------


    def _detect_vehicles(self, frame: np.ndarray) -> sv.Detections:
        """Detect vehicles inside ROI.

        We run the model on a pre-crop ROI, map boxes back to original coordinates,
        keep only objects whose bottom-center anchor is inside the gate polygon,
        then apply vehicle-specific size filters.

        Additionally, we *also* keep person boxes (with looser filters) and suppress
        vehicle boxes that strongly overlap persons (helps against pedestrian false positives).
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._roi_xyxy
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return sv.Detections.empty()

        crop = frame[y1:y2, x1:x2]
        xyxy_in, conf, lab = self.detector.infer(crop)
        if xyxy_in.shape[0] == 0:
            return sv.Detections.empty()

        # Map boxes back to original coordinates.
        h_in, w_in = self.detector.input_hw
        scale_x = (x2 - x1) / float(w_in)
        scale_y = (y2 - y1) / float(h_in)

        xyxy = xyxy_in.copy()
        xyxy[:, 0] = xyxy[:, 0] * scale_x + x1
        xyxy[:, 2] = xyxy[:, 2] * scale_x + x1
        xyxy[:, 1] = xyxy[:, 1] * scale_y + y1
        xyxy[:, 3] = xyxy[:, 3] * scale_y + y1

        ww = (xyxy[:, 2] - xyxy[:, 0])
        hh = (xyxy[:, 3] - xyxy[:, 1])
        area = ww * hh

        # Anchor inside polygon
        cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        ay = xyxy[:, 3]  # bottom
        inside = np.zeros((xyxy.shape[0],), dtype=bool)
        for i in range(xyxy.shape[0]):
            inside[i] = (cv2.pointPolygonTest(self.zone_polygon, (float(cx[i]), float(ay[i])), False) >= 0)

        xyxy = xyxy[inside]
        conf = conf[inside]
        lab = lab[inside]
        ww = ww[inside]
        hh = hh[inside]
        area = area[inside]

        if xyxy.shape[0] == 0:
            return sv.Detections.empty()

        is_vehicle = (lab == 0)
        is_person = (lab == 1)

        # Vehicle filters
        keep_v = is_vehicle.copy()
        keep_v &= (ww >= self.cfg.min_box_width) & (hh >= self.cfg.min_box_height) & (area >= self.cfg.min_box_area)
        if self.cfg.max_box_area is not None:
            keep_v &= (area <= int(self.cfg.max_box_area))

        # Person filters (looser; only for suppression)
        keep_p = is_person.copy()
        keep_p &= (ww >= self.cfg.min_person_box_width) & (hh >= self.cfg.min_person_box_height) & (area >= self.cfg.min_person_box_area)

        veh_boxes = xyxy[keep_v]
        veh_conf = conf[keep_v]

        # Suppress vehicles that overlap persons strongly
        if self.cfg.suppress_person_iou > 0:
            per_boxes = xyxy[keep_p]
            if veh_boxes.shape[0] > 0 and per_boxes.shape[0] > 0:
                iou = _iou_matrix(veh_boxes.astype(np.float32), per_boxes.astype(np.float32))
                max_iou = iou.max(axis=1)
                keep = max_iou < float(self.cfg.suppress_person_iou)
                veh_boxes = veh_boxes[keep]
                veh_conf = veh_conf[keep]

        if veh_boxes.shape[0] == 0:
            return sv.Detections.empty()

        # All returned detections are vehicles (class_id=0)
        class_id = np.zeros((veh_boxes.shape[0],), dtype=int)
        return sv.Detections(
            xyxy=veh_boxes.astype(np.float32),
            confidence=veh_conf.astype(np.float32),
            class_id=class_id,
        )

    # ----------------------------
    # Tracking & counting
    # ----------------------------

    def _update_track_state(self, detections: sv.Detections, ts_sec: float) -> None:
        if detections.tracker_id is None:
            return

        for i, tid in enumerate(detections.tracker_id):
            tid = int(tid)
            x0, y0, x1, y1 = detections.xyxy[i].tolist()
            ax = (x0 + x1) * 0.5
            ay = y1
            zone = self._zone_index(ax, ay)

            st = self._tracks.get(tid)
            if st is None:
                st = {
                    "first_ts": ts_sec,
                    "last_ts": ts_sec,
                    "frames": 0,
                    "zones": deque(maxlen=120),
                    "counted": False,
                }
                self._tracks[tid] = st

            st["last_ts"] = ts_sec
            st["frames"] += 1

            zq: Deque[int] = st["zones"]
            if not zq or zq[-1] != zone:
                zq.append(zone)

    def _maybe_cleanup_tracks(self, ts_sec: float) -> None:
        # remove stale tracks
        ttl = max(2.0, self.cfg.lost_track_buffer_sec + 2.0)
        dead = [tid for tid, st in self._tracks.items() if (ts_sec - float(st["last_ts"])) > ttl]
        for tid in dead:
            self._tracks.pop(tid, None)

    def _track_event(self, st: Dict) -> Optional[str]:
        if st.get("counted"):
            return None
        if int(st.get("frames", 0)) < int(self.cfg.min_track_frames):
            return None

        zones: List[int] = list(st.get("zones", []))
        if len(zones) < 2:
            return None

        zmin, zmax = min(zones), max(zones)
        if (zmax - zmin) < int(self.cfg.min_zone_spread):
            return None

        # find first time we hit low/high thresholds
        low_thr = int(self.cfg.low_zone_threshold)
        high_thr = int(self.cfg.high_zone_threshold)

        first_low = next((i for i, z in enumerate(zones) if z >= low_thr), None)
        first_high = next((i for i, z in enumerate(zones) if z <= high_thr), None)
        if first_low is None or first_high is None:
            return None
        # if thresholds overlap (e.g., both=2), require we actually moved across at least one index
        if first_low == first_high and (max(zones) - min(zones)) < 2:
            return None

        # direction: compare median of first half vs last half
        mid = max(1, len(zones) // 2)
        a = float(np.median(zones[:mid]))
        b = float(np.median(zones[mid:]))
        trend = b - a  # positive = moving down (towards gate)

        if self.cfg.in_direction == "bottom_to_top":
            # IN: low (bottom) happens before high (top), and trend is negative (up)
            if first_low < first_high and trend < 0:
                st["counted"] = True
                return "IN"
            # OUT: high happens before low, trend positive
            if first_high < first_low and trend > 0:
                st["counted"] = True
                return "OUT"
        else:
            # flipped camera
            if first_high < first_low and trend > 0:
                st["counted"] = True
                return "IN"
            if first_low < first_high and trend < 0:
                st["counted"] = True
                return "OUT"

        return None

    def process_frame(self, frame: np.ndarray, ts_sec: Optional[float] = None) -> List[str]:
        ts = float(ts_sec) if ts_sec is not None else float(cv2.getTickCount() / cv2.getTickFrequency())

        det = self._detect_vehicles(frame)
        det = self.tracker.update_with_detections(det)

        self._last_det = det

        self._update_track_state(det, ts)
        self._maybe_cleanup_tracks(ts)

        events: List[str] = []
        # evaluate events for tracks updated recently
        for tid, st in list(self._tracks.items()):
            # only consider tracks that were seen very recently
            if (ts - float(st["last_ts"])) > 0.8:
                continue
            ev = self._track_event(st)
            if ev is not None:
                # global debounce (protects against track-id flicker creating duplicates)
                if (ts - float(self._last_event_ts.get(ev, -1e9))) >= float(self.cfg.global_event_cooldown_sec):
                    self._last_event_ts[ev] = ts
                    events.append(ev)

        return events

    def process_frame_debug(self, frame: np.ndarray, ts_sec: Optional[float] = None) -> Tuple[List[str], FrameDebug]:
        events = self.process_frame(frame, ts_sec=ts_sec)

        det = self._last_det
        dbg_frame = frame.copy()
        if self.cfg.draw:
            cv2.polylines(dbg_frame, [self.zone_polygon], True, (0, 255, 0), 2)
            for i, zpoly in enumerate(self._zones_poly):
                cv2.polylines(dbg_frame, [zpoly], True, (80, 80, 80), 1)
                # label zone index on left edge
                p = zpoly[0]
                cv2.putText(dbg_frame, str(i), (int(p[0]) + 3, int(p[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

            box_anno = sv.BoxAnnotator(thickness=2)
            label_anno = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
            dbg_frame = box_anno.annotate(scene=dbg_frame, detections=det)
            if det.tracker_id is not None:
                labels = [f"id={int(t)}" for t in det.tracker_id]
                dbg_frame = label_anno.annotate(scene=dbg_frame, detections=det, labels=labels)

            if events:
                cv2.putText(dbg_frame, f"EVENT: {events}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        return events, FrameDebug(frame=dbg_frame, detections=det, zones=self._zones_poly)
