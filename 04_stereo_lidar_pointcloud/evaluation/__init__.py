"""Structured per-run evaluation: LiDAR ray metrics, photometric, cross-method, consensus."""

from evaluation.depth_maps import METHODS, load_metric_depth, load_or_compute_stereo_geometry

__all__ = ["METHODS", "load_metric_depth", "load_or_compute_stereo_geometry"]
