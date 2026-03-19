"""ByteTrack-style multi-object tracker with uncertainty awareness."""

from __future__ import annotations

import numpy as np

from .track import Track, TrackState
from .association import associate_detections, associate_bytetrack


class AdaptTracker:
    """
    Multi-object tracker using ByteTrack association with uncertainty propagation.

    Maintains a set of tracks, associates new detections each frame,
    and manages track lifecycle (creation, update, deletion).
    """

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        uncertainty_threshold: float = 0.7,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.uncertainty_threshold = uncertainty_threshold

        self.tracks: list[Track] = []
        self.frame_count = 0

        # tracking stats
        self.stats = {
            "total_tracks_created": 0,
            "total_tracks_removed": 0,
            "total_matches": 0,
            "total_id_switches": 0,
            "high_uncertainty_frames": 0,
        }

    def update(self, detections: dict) -> list[dict]:
        """
        Update tracker with new detections.

        Args:
            detections: dict with bboxes, scores, class_ids, uncertainties

        Returns:
            list of active track dicts with keys:
                track_id, bbox, score, class_id, uncertainty, state
        """
        self.frame_count += 1

        bboxes = detections["bboxes"]
        scores = detections["scores"]
        class_ids = detections["class_ids"]
        uncertainties = detections.get("uncertainties", np.zeros(len(scores)))

        # split detections by confidence
        high_mask = scores >= self.high_thresh
        low_mask = (scores >= self.low_thresh) & (scores < self.high_thresh)

        high_bboxes = bboxes[high_mask]
        high_scores = scores[high_mask]
        high_classes = class_ids[high_mask]
        high_uncerts = uncertainties[high_mask]

        low_bboxes = bboxes[low_mask]
        low_scores = scores[low_mask]
        low_classes = class_ids[low_mask]
        low_uncerts = uncertainties[low_mask]

        # predict existing tracks
        predicted_bboxes = []
        for track in self.tracks:
            pred = track.predict()
            predicted_bboxes.append(pred)
        predicted_bboxes = np.array(predicted_bboxes) if predicted_bboxes else np.empty((0, 4))

        # ByteTrack two-stage association
        if len(predicted_bboxes) > 0:
            matches_high, unmatched_dets, unmatched_trks, matches_low = (
                associate_bytetrack(
                    high_bboxes, low_bboxes, predicted_bboxes,
                    high_threshold=self.iou_threshold,
                    low_threshold=self.iou_threshold * 0.7,
                )
            )
        else:
            matches_high = []
            matches_low = []
            unmatched_dets = list(range(len(high_bboxes)))
            unmatched_trks = []

        # update matched tracks (high confidence)
        for det_idx, trk_idx in matches_high:
            self.tracks[trk_idx].update(
                high_bboxes[det_idx], high_scores[det_idx],
                high_classes[det_idx], high_uncerts[det_idx]
            )
            self.stats["total_matches"] += 1

        # update matched tracks (low confidence)
        for det_idx, trk_idx in matches_low:
            self.tracks[trk_idx].update(
                low_bboxes[det_idx], low_scores[det_idx],
                low_classes[det_idx], low_uncerts[det_idx]
            )
            self.stats["total_matches"] += 1

        # create new tracks from unmatched high-confidence detections
        for det_idx in unmatched_dets:
            new_track = Track(
                high_bboxes[det_idx], high_scores[det_idx],
                high_classes[det_idx], high_uncerts[det_idx]
            )
            self.tracks.append(new_track)
            self.stats["total_tracks_created"] += 1

        # handle unmatched tracks
        for trk_idx in unmatched_trks:
            track = self.tracks[trk_idx]
            if track.time_since_update > self.max_age:
                track.mark_removed()
                self.stats["total_tracks_removed"] += 1
            elif track.state == TrackState.CONFIRMED:
                track.mark_lost()

        # remove dead tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.REMOVED]

        # check for high uncertainty
        active = [t for t in self.tracks if t.state == TrackState.CONFIRMED]
        if active:
            avg_uncert = np.mean([t.uncertainty for t in active])
            if avg_uncert > self.uncertainty_threshold:
                self.stats["high_uncertainty_frames"] += 1

        return self._get_output()

    def _get_output(self) -> list[dict]:
        """Get list of confirmed/tentative tracks for output."""
        output = []
        for track in self.tracks:
            if track.state == TrackState.REMOVED:
                continue
            if track.state == TrackState.TENTATIVE and track.hits < self.min_hits:
                continue

            bbox = track.get_bbox()
            cov = track.get_covariance()
            kf_uncertainty = float(np.trace(cov[:2, :2]))

            output.append({
                "track_id": track.track_id,
                "bbox": bbox.tolist(),
                "score": float(track.score),
                "class_id": int(track.class_id),
                "uncertainty": float(track.uncertainty),
                "kf_uncertainty": kf_uncertainty,
                "state": track.state,
                "age": track.age,
                "hits": track.hits,
                "time_since_update": track.time_since_update,
            })

        return output

    def get_stats(self) -> dict:
        """Get accumulated tracking statistics."""
        return {
            **self.stats,
            "active_tracks": len([t for t in self.tracks
                                  if t.state == TrackState.CONFIRMED]),
            "tentative_tracks": len([t for t in self.tracks
                                     if t.state == TrackState.TENTATIVE]),
            "lost_tracks": len([t for t in self.tracks
                                if t.state == TrackState.LOST]),
            "frame_count": self.frame_count,
        }

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.frame_count = 0
        Track.reset_id_counter()
        self.stats = {k: 0 for k in self.stats}
