"""
storage.py - lightweight persistence for parking state + event history.

Goal: keep *current* free spots count consistent across service restarts,
and collect entry/exit history for later stats.

- State is stored as JSON (atomic write).
- Events are appended into SQLite (built-in, safe for small volumes).

Environment (optional):
- PARKING_STATE_PATH (default: parking_state.json)
- PARKING_DB_PATH (default: parking_stats.db)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


EventType = Literal["IN", "OUT"]


@dataclass(frozen=True)
class ParkingState:
    total_spots: int
    free_spots: int
    updated_ts: float

    @property
    def occupied_spots(self) -> int:
        return max(0, int(self.total_spots) - int(self.free_spots))


class ParkingStore:
    def __init__(
        self,
        state_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self.state_path = Path(state_path or os.getenv("PARKING_STATE_PATH", "parking_state.json"))
        self.db_path = Path(db_path or os.getenv("PARKING_DB_PATH", "parking_stats.db"))
        self._init_db()

    # ---------- public API ----------

    def load_or_init_state(self, total_spots: int, free_spots_default: Optional[int] = None) -> ParkingState:
        """
        Load state from JSON if exists, otherwise create it.
        If total_spots differs from file -> update total_spots (keeps free_spots clamped).
        """
        total_spots = int(total_spots)
        if total_spots <= 0:
            raise ValueError("total_spots must be > 0")

        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                st = ParkingState(
                    total_spots=int(data["total_spots"]),
                    free_spots=int(data["free_spots"]),
                    updated_ts=float(data.get("updated_ts", time.time())),
                )
            except Exception:
                # corrupt state -> fall back
                st = self._new_state(total_spots, free_spots_default)
        else:
            st = self._new_state(total_spots, free_spots_default)

        # reconcile total_spots (e.g., config changed)
        if st.total_spots != total_spots:
            free = self._clamp_free(st.free_spots, total_spots)
            st = ParkingState(total_spots=total_spots, free_spots=free, updated_ts=time.time())
            self.save_state(st)

        return st

    def save_state(self, st: ParkingState) -> None:
        payload = {"total_spots": int(st.total_spots), "free_spots": int(st.free_spots), "updated_ts": float(st.updated_ts)}
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def apply_event(self, st: ParkingState, ev: EventType, ts: Optional[float] = None) -> ParkingState:
        ts = float(ts) if ts is not None else time.time()

        if ev not in ("IN", "OUT"):
            raise ValueError(f"Unsupported event: {ev}")

        # IN => one more car => free decreases
        delta = -1 if ev == "IN" else +1
        new_free = self._clamp_free(int(st.free_spots) + delta, int(st.total_spots))
        new_st = ParkingState(total_spots=int(st.total_spots), free_spots=int(new_free), updated_ts=float(ts))

        self.save_state(new_st)
        self._insert_event(ts=ts, ev=ev, free_spots=new_st.free_spots, total_spots=new_st.total_spots)
        return new_st

    # ---------- internals ----------

    def _new_state(self, total_spots: int, free_spots_default: Optional[int]) -> ParkingState:
        if free_spots_default is None:
            free_spots_default = total_spots
        free = self._clamp_free(int(free_spots_default), int(total_spots))
        st = ParkingState(total_spots=int(total_spots), free_spots=int(free), updated_ts=time.time())
        self.save_state(st)
        return st

    def _clamp_free(self, free_spots: int, total_spots: int) -> int:
        return max(0, min(int(free_spots), int(total_spots)))

    def _init_db(self) -> None:
        # create minimal schema
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    ev TEXT NOT NULL,
                    free_spots INTEGER NOT NULL,
                    total_spots INTEGER NOT NULL
                );
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);")

    def _insert_event(self, ts: float, ev: str, free_spots: int, total_spots: int) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO events(ts, ev, free_spots, total_spots) VALUES(?,?,?,?)",
                (float(ts), str(ev), int(free_spots), int(total_spots)),
            )
