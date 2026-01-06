# webcamera_vehicle_counter (OpenVINO + Telegram)

A lightweight service that reads an **RTSP** stream from a web camera, **detects vehicles** (OpenVINO) and **tracks them** (ByteTrack),
detects **IN/OUT** events at the entrance/exit gate, and maintains the **current parking state** (how many spots are free).

On every event it sends a **single Telegram message**:
**video (MP4/H.264) + caption** with the current number of free spots.

---

## Description

The service:
- connects to an RTSP source
- runs a vehicle detector with a configurable FPS
- tracks objects across frames to determine movement direction
- splits the **gate polygon** into several zones and uses the track trajectory to classify the event as **IN** or **OUT**
- updates the current parking state and persists it between restarts
- sends an annotated Telegram clip (zone + detections) while optionally saving a **raw** (clean) clip locally for tests

---

## Key Features

- RTSP input (web camera / IP camera)
- OpenVINO inference (CPU-only)
- ByteTrack tracking (via `supervision`)
- Gate/ROI calibration with `calibrate_zone.py`
- Telegram notifications: **one message = video + caption**
- Persistent state:
  - `parking_state.json` — current free/total
  - `parking_stats.db` — SQLite event history (IN/OUT)
- Test runner with PyTest (fixtures + optional debug rendering)

---

## Requirements

- Python **3.8+** (recommended: 3.10/3.11)
- **ffmpeg** (required to encode MP4/H.264 clips)
- Internet connection (for Telegram)
- CPU only (no GPU required)

---

## How to Run

### 1) Clone the Repository

```bash
git clone <repository_link>
cd webcamera_vehicle_counter
```

### 2) Create a Virtual Environment and Install Dependencies

```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

Download model via 
```bash 
python scripts/download_model.py
```

> **ffmpeg**
> - Linux: `sudo apt-get install -y ffmpeg`
> - Windows: install ffmpeg and either add it to `PATH` or set `FFMPEG_BIN=...\ffmpeg.exe`

### 3) Configure Environment Variables

Create a `.env` file in the project root:

```ini
# RTSP
VIDEO_SOURCE=rtsp://user:pass@ip:554/...

# OpenVINO model
MODEL_XML=./models/person-vehicle-bike-detection-2002.xml

# Telegram
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=123456789

# Parking
PARKING_TOTAL_SPOTS=40
# Used only if parking_state.json does not exist yet
PARKING_INITIAL_FREE_SPOTS=20

# Clip buffer / encoding
PRE_EVENT_SEC=5
POST_EVENT_SEC=5
BUFFER_FPS=10
VIDEO_FPS=10
VIDEO_MAX_WIDTH=960

# Output
CLIPS_DIR=clips
SAVE_RAW_CLIP=1      # save clean clip to CLIPS_DIR (no overlays)
CLIP_ANNOTATE=1      # annotate Telegram clip (zone + detections)

# Debug
SHOW=0               # show debug window (local)
DRAW=1               # enable overlays for Telegram clip

# ffmpeg (optional)
# FFMPEG_BIN=/usr/bin/ffmpeg
# FFMPEG_BIN=C:\ffmpeg\bin\ffmpeg.exe
```

---

## Model

Recommended Open Model Zoo (OMZ) model: **person-vehicle-bike-detection-2002** (vehicle=0, person=1, bike=2).

---

## Gate / ROI Calibration

Run:

```bash
python calibrate_zone.py
```

Click 4 points in this order: **BL, BR, TR, TL**.  
The script will save `zone_polygon.json`.

---

## Run the Service Locally

```bash
python rtsp_service.py
```

---

## Files (State & History)

- `parking_state.json` — current state (free/total), persisted between restarts
- `parking_stats.db` — SQLite event history (IN/OUT), for future daily/weekly/monthly stats

---

## How to Set Up as a Service (systemd / Ubuntu)

Create a systemd unit:

`/etc/systemd/system/webcamera_vehicle_counter.service`

```ini
[Unit]
Description=webcamera_vehicle_counter (RTSP parking counter)
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/webcamera_vehicle_counter
ExecStart=/path/to/webcamera_vehicle_counter/venv/bin/python /path/to/webcamera_vehicle_counter/rtsp_service.py
Environment="PYTHONUNBUFFERED=1"
Restart=always
RestartSec=5s

# Optional limits (tune for your server)
# MemoryLimit=1G
# CPUQuota=70%

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable webcamera_vehicle_counter.service
sudo systemctl start webcamera_vehicle_counter.service
```

Check status and logs:

```bash
sudo systemctl status webcamera_vehicle_counter.service
journalctl -u webcamera_vehicle_counter.service -f
```

---

## Tests

Put test clips into `tests/fixtures/` and run:

```bash
python -m pytest -s --show
# or:
python -m pytest -s --save-debug
```

Options:
- `--sample-fps=10` (default 10)
- `--show` (live window, ESC/q to stop)
- `--save-debug` (writes annotated mp4 into `tests/_out/`)

---

## About

**webcamera_vehicle_counter** — RTSP parking spot counter with Telegram alerts (OpenVINO + ByteTrack).
