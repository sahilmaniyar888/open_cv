"""Failure detection module for tracking reliability monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class FailureEvent:
    frame_idx: int
    event_type: str  # "fragmentation", "id_switch", "high_uncertainty", "track_loss"
    track_id: int | None = None
    severity: float = 0.0  # 0-1
    details: str = ""


class FailureDetector:
    """
    Monitors tracker output for signs of failure.

    Detects track fragmentation, ID switches, prolonged uncertainty,
    and sudden track loss events.
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.6,
        uncertainty_window: int = 5,
        fragmentation_gap: int = 10,
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_window = uncertainty_window
        self.fragmentation_gap = fragmentation_gap

        self.events: list[FailureEvent] = []
        self.frame_idx = 0

        # per-track history for pattern detection
        self._track_last_seen: dict[int, int] = {}
        self._track_uncertainty_buffer: dict[int, list[float]] = {}
        self._prev_track_ids: set[int] = set()
        self._active_count_history: list[int] = []

    def check(self, tracks: list[dict], frame_idx: int | None = None):
        """
        Analyze current frame tracks for failure indicators.

        Args:
            tracks: list of track dicts from tracker.update()
            frame_idx: optional explicit frame index
        """
        if frame_idx is not None:
            self.frame_idx = frame_idx
        else:
            self.frame_idx += 1

        current_ids = set()
        for t in tracks:
            tid = t["track_id"]
            current_ids.add(tid)

            # check uncertainty
            unc = t.get("uncertainty", 0)
            if tid not in self._track_uncertainty_buffer:
                self._track_uncertainty_buffer[tid] = []
            self._track_uncertainty_buffer[tid].append(unc)

            buf = self._track_uncertainty_buffer[tid]
            if len(buf) >= self.uncertainty_window:
                recent = buf[-self.uncertainty_window:]
                avg_unc = np.mean(recent)
                if avg_unc > self.uncertainty_threshold:
                    self.events.append(FailureEvent(
                        frame_idx=self.frame_idx,
                        event_type="high_uncertainty",
                        track_id=tid,
                        severity=float(avg_unc),
                        details=f"sustained uncertainty {avg_unc:.3f} over {self.uncertainty_window} frames",
                    ))

            # check fragmentation (track reappearing after gap)
            if tid in self._track_last_seen:
                gap = self.frame_idx - self._track_last_seen[tid]
                if gap > self.fragmentation_gap:
                    self.events.append(FailureEvent(
                        frame_idx=self.frame_idx,
                        event_type="fragmentation",
                        track_id=tid,
                        severity=min(gap / 50.0, 1.0),
                        details=f"track reappeared after {gap} frame gap",
                    ))

            self._track_last_seen[tid] = self.frame_idx

        # check for sudden track loss
        lost_ids = self._prev_track_ids - current_ids
        for lid in lost_ids:
            self.events.append(FailureEvent(
                frame_idx=self.frame_idx,
                event_type="track_loss",
                track_id=lid,
                severity=0.5,
                details=f"track {lid} disappeared",
            ))

        # check for sudden count drop
        self._active_count_history.append(len(current_ids))
        if len(self._active_count_history) > 3:
            prev_avg = np.mean(self._active_count_history[-4:-1])
            if prev_avg > 0 and len(current_ids) < prev_avg * 0.5:
                self.events.append(FailureEvent(
                    frame_idx=self.frame_idx,
                    event_type="track_loss",
                    severity=0.8,
                    details=f"track count dropped from ~{prev_avg:.0f} to {len(current_ids)}",
                ))

        self._prev_track_ids = current_ids

    def get_recent_events(self, n: int = 10) -> list[FailureEvent]:
        return self.events[-n:]

    def get_summary(self) -> dict:
        type_counts = {}
        for e in self.events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1

        avg_severity = (
            np.mean([e.severity for e in self.events]) if self.events else 0.0
        )

        return {
            "total_events": len(self.events),
            "event_counts": type_counts,
            "avg_severity": float(avg_severity),
            "frames_analyzed": self.frame_idx,
        }

    def reset(self):
        self.events = []
        self.frame_idx = 0
        self._track_last_seen = {}
        self._track_uncertainty_buffer = {}
        self._prev_track_ids = set()
        self._active_count_history = []
