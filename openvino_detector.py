import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


@dataclass
class OVDetectorConfig:
    model_xml: str
    device: str = "CPU"
    conf_threshold: float = 0.45
    # For person-vehicle-bike-detection-2002: 0=vehicle,1=person,2=bike
    allowed_labels: Tuple[int, ...] = (0,)
    # Optional bbox filters (in pixels, in *original frame* coordinates)
    min_box_area: int = 1800
    min_box_width: int = 45
    min_box_height: int = 35
    max_box_area: Optional[int] = None


class OpenVINOObjectDetector:
    """OpenVINO IR object detector with SSD-like output 1x1xNx7.

    Output row format: [image_id, label, conf, x_min, y_min, x_max, y_max]
    where coords are normalized to input size.
    """

    def __init__(self, cfg: OVDetectorConfig):
        self.cfg = cfg
        self._compiled = None
        self._input_name = None
        self._input_hw = None  # (H,W)
        self._output_index = 0

        # lazy import (keeps module importable without OpenVINO installed)
        try:
            from openvino.runtime import Core  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenVINO is not installed. Install: pip install openvino openvino-dev"
            ) from e

        if not cfg.model_xml or not os.path.exists(cfg.model_xml):
            raise FileNotFoundError(f"MODEL_XML not found: {cfg.model_xml}")

        core = Core()
        model = core.read_model(cfg.model_xml)
        self._compiled = core.compile_model(model, cfg.device)

        inp = self._compiled.input(0)
        self._input_name = inp.any_name
        shape = list(inp.shape)
        # Expect NCHW
        if len(shape) != 4:
            raise ValueError(f"Unexpected input shape: {shape}")
        h, w = int(shape[2]), int(shape[3])
        self._input_hw = (h, w)

        # pick first output
        _ = self._compiled.output(0)

    @property
    def input_hw(self) -> Tuple[int, int]:
        assert self._input_hw is not None
        return self._input_hw

    def infer(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (xyxy, conf, label) in input image coordinates (pixel)."""
        h_in, w_in = self.input_hw
        img = cv2.resize(bgr, (w_in, h_in), interpolation=cv2.INTER_AREA)
        blob = img.transpose(2, 0, 1)[None, ...].astype(np.float32)

        res = self._compiled([blob])
        out = list(res.values())[0]
        det = np.array(out)
        det = det.reshape(-1, 7)

        xyxy: List[List[float]] = []
        conf: List[float] = []
        lab: List[int] = []

        for row in det:
            image_id, label, score, x0, y0, x1, y1 = row.tolist()
            if score < self.cfg.conf_threshold:
                continue
            label = int(label)
            if self.cfg.allowed_labels and label not in self.cfg.allowed_labels:
                continue

            # Some models return -1 when no detections
            if image_id < 0:
                continue

            x0p = float(x0) * w_in
            y0p = float(y0) * h_in
            x1p = float(x1) * w_in
            y1p = float(y1) * h_in

            x0p = max(0.0, min(x0p, w_in - 1.0))
            y0p = max(0.0, min(y0p, h_in - 1.0))
            x1p = max(0.0, min(x1p, w_in - 1.0))
            y1p = max(0.0, min(y1p, h_in - 1.0))

            if x1p <= x0p or y1p <= y0p:
                continue

            xyxy.append([x0p, y0p, x1p, y1p])
            conf.append(float(score))
            lab.append(label)

        if not xyxy:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        return (
            np.asarray(xyxy, dtype=np.float32),
            np.asarray(conf, dtype=np.float32),
            np.asarray(lab, dtype=np.int32),
        )
