"""Track data structures for multi-object tracking."""

from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter


class TrackState:
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    REMOVED = 3


def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1,y1,x2,y2] to [cx, cy, s, r] where s=area, r=aspect ratio."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / (h + 1e-6)
    return np.array([cx, cy, s, r]).reshape((4, 1))


def z_to_bbox(z: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, s, r] back to [x1, y1, x2, y2]."""
    cx, cy, s, r = z.flatten()[:4]
    w = np.sqrt(max(s * r, 1.0))
    h = s / (w + 1e-6)
    return np.array([
        cx - w / 2.0, cy - h / 2.0,
        cx + w / 2.0, cy + h / 2.0
    ])


class Track:
    _next_id = 1

    def __init__(self, bbox: np.ndarray, score: float, class_id: int,
                 uncertainty: float = 0.0):
        self.track_id = Track._next_id
        Track._next_id += 1

        self.kf = self._init_kalman(bbox)
        self.score = score
        self.class_id = class_id
        self.uncertainty = uncertainty
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.hit_streak = 1
        self.history: list[np.ndarray] = [bbox.copy()]

        # for uncertainty tracking
        self.score_history: list[float] = [score]
        self.uncertainty_history: list[float] = [uncertainty]

    def _init_kalman(self, bbox: np.ndarray) -> KalmanFilter:
        """Initialize a constant velocity Kalman filter for bounding box tracking."""
        kf = KalmanFilter(dim_x=7, dim_z=4)
        # state: [cx, cy, s, r, vx, vy, vs]
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)

        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)

        kf.R[2:, 2:] *= 10.0
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01

        kf.x[:4] = bbox_to_z(bbox)
        return kf

    def predict(self) -> np.ndarray:
        """Advance state and return predicted bbox."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        predicted_bbox = z_to_bbox(self.kf.x)
        return predicted_bbox

    def update(self, bbox: np.ndarray, score: float, class_id: int,
               uncertainty: float = 0.0):
        """Update track with matched detection."""
        self.kf.update(bbox_to_z(bbox))
        self.score = score
        self.class_id = class_id
        self.uncertainty = uncertainty
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.history.append(bbox.copy())
        self.score_history.append(score)
        self.uncertainty_history.append(uncertainty)

        if self.state == TrackState.TENTATIVE and self.hits >= 3:
            self.state = TrackState.CONFIRMED

    def get_bbox(self) -> np.ndarray:
        """Get current estimated bounding box."""
        return z_to_bbox(self.kf.x)

    def get_covariance(self) -> np.ndarray:
        """Get position covariance (uncertainty from Kalman filter)."""
        return self.kf.P[:4, :4].copy()

    def mark_lost(self):
        self.state = TrackState.LOST

    def mark_removed(self):
        self.state = TrackState.REMOVED

    @classmethod
    def reset_id_counter(cls):
        cls._next_id = 1
