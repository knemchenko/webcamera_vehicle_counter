import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from dotenv import load_dotenv, find_dotenv

from parking_counter import ParkingCounter, CounterConfig

# Paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIXTURES = HERE / "fixtures"
OUTDIR = HERE / "_out"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ROI polygon: BL, BR, TR, TL
ZONE_POLYGON = np.array(
    [[871, 1061], [1144, 1070], [1155, 708], [894, 699]],
    dtype=np.int32,
)
def _ensure_model_or_skip() -> str:
    # Ensure .env is loaded from project root when running from elsewhere
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        # try root/.env
        load_dotenv(ROOT / ".env")

    model_xml = (os.getenv("MODEL_XML") or "").strip()
    if not model_xml:
        pytest.skip("MODEL_XML is not set. Put it into .env or env vars.")
    if not Path(model_xml).exists():
        pytest.skip(f"MODEL_XML not found: {model_xml}")
    return model_xml


def draw_debug(vis_frame: np.ndarray) -> np.ndarray:
    # Already annotated by ParkingCounter.process_frame_debug
    return vis_frame


def run_clip(path: str, sample_fps: float = 10.0, show: bool = False, save_debug: bool = False):
    _ensure_model_or_skip()

    cfg = CounterConfig(frame_rate=float(sample_fps))
    counter = ParkingCounter(ZONE_POLYGON, cfg)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if src_fps <= 1:
        src_fps = 25.0

    step = max(1, int(round(src_fps / sample_fps)))
    dt = 1.0 / float(sample_fps)
    ts = 0.0

    got_in = 0
    got_out = 0

    writer = None
    out_path = None
    if save_debug:
        out_path = str(OUTDIR / (Path(path).stem + "_debug.mp4"))

    while True:
        # fast-forward
        for _ in range(step - 1):
            if not cap.grab():
                cap.release()
                break

        ret, frame = cap.read()
        if not ret:
            break

        events, dbg = counter.process_frame_debug(frame, ts_sec=ts)
        for ev in events:
            if ev == "IN":
                got_in += 1
            elif ev == "OUT":
                got_out += 1

        vis = draw_debug(dbg.frame)

        if save_debug:
            if writer is None:
                h, w = vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, float(sample_fps), (w, h))
            writer.write(vis)

        if show:
            cv2.imshow("debug", vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        ts += dt

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    return got_in, got_out, out_path


@pytest.mark.parametrize(
    "fname, exp_in, exp_out",
    [
        # existing
        ("day_in_1.mp4", 1, 0),
        ("night_in.mp4", 1, 0),
        ("day_out_1.mp4", 0, 1),
        ("day_out_2.mp4", 0, 1),
        ("day_out_3.mp4", 0, 1),
        ("night_out.mp4", 0, 1),
        ("walking_1.mp4", 0, 0),

        # day IN
        ("day_in_2.mp4", 1, 0),
        ("day_in_3.mp4", 1, 0),
        ("day_in_4.mp4", 1, 0),
        ("day_in_5.mp4", 1, 0),
        ("day_in_6.mp4", 1, 0),
        ("day_in_7.mp4", 1, 0),
        ("day_in_8.mp4", 1, 0),
        ("day_in_9.mp4", 2, 0),

        # walking
        ("walking_2.mp4", 0, 0),
        ("walking_3.mp4", 0, 0),
        ("walking_4.mp4", 0, 0),
        ("walking_5.mp4", 0, 0),

        # night OUT
        ("night_out_2.mp4", 0, 1),
        ("night_out_3.mp4", 0, 1),
    ],
)
def test_scenarios(fname, exp_in, exp_out, pytestconfig):
    # look in tests/fixtures first, then project root
    p1 = FIXTURES / fname
    p2 = ROOT / fname
    if p1.exists():
        path = str(p1)
    elif p2.exists():
        path = str(p2)
    else:
        pytest.skip(f"Missing fixture: {fname} (looked in {p1} and {p2})")

    show = bool(pytestconfig.getoption("--show"))
    save_debug = bool(pytestconfig.getoption("--save-debug"))
    sample_fps = float(pytestconfig.getoption("--sample-fps"))

    got_in, got_out, out_path = run_clip(path, sample_fps=sample_fps, show=show, save_debug=save_debug)

    try:
        assert got_in == exp_in
        assert got_out == exp_out
    except AssertionError:
        if out_path:
            print(f"\n[DEBUG VIDEO SAVED] {out_path}\n")
        raise
