"""Synthetic dataset generation for public nulling-angle prediction demo.

This module intentionally does not use measured self-interference channels,
S-parameter matrices, or lab-owned datasets. The synthetic labels mimic the
structure of PSO-optimized nulling-angle labels for portfolio demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AngleConfig:
    steering_min_deg: float = -40.0
    steering_max_deg: float = 40.0
    null_min_deg: float = -90.0
    null_max_deg: float = 90.0
    min_separation_deg: float = 15.0


def _normalize(value: np.ndarray, low: float, high: float) -> np.ndarray:
    return (value - low) / (high - low)


def _denormalize(value: np.ndarray, low: float, high: float) -> np.ndarray:
    return value * (high - low) + low


def normalize_null_angles(null_angles_deg: np.ndarray, cfg: AngleConfig = AngleConfig()) -> np.ndarray:
    return _normalize(null_angles_deg, cfg.null_min_deg, cfg.null_max_deg)


def denormalize_null_angles(null_angles_norm: np.ndarray, cfg: AngleConfig = AngleConfig()) -> np.ndarray:
    return _denormalize(null_angles_norm, cfg.null_min_deg, cfg.null_max_deg)


def denormalize_steering_angles(steering_norm: np.ndarray, cfg: AngleConfig = AngleConfig()) -> np.ndarray:
    return _denormalize(steering_norm, cfg.steering_min_deg, cfg.steering_max_deg)


def _enforce_min_separation(null_angle: np.ndarray, steering_angle: np.ndarray, cfg: AngleConfig) -> np.ndarray:
    """Move nulling angles away from the steering angle if they are too close."""
    diff = null_angle - steering_angle
    too_close = np.abs(diff) < cfg.min_separation_deg
    direction = np.where(diff >= 0.0, 1.0, -1.0)
    adjusted = steering_angle + direction * cfg.min_separation_deg
    null_angle = np.where(too_close, adjusted, null_angle)
    return np.clip(null_angle, cfg.null_min_deg, cfg.null_max_deg)


def generate_synthetic_dataset(
    n_samples: int = 8000,
    seed: int = 7,
    cfg: AngleConfig = AngleConfig(),
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Generate a synthetic supervised-learning dataset.

    Features contain six engineered terms inspired by the angle-only inputs
    used in nulling-angle prediction:
      [theta_D_norm, theta_U_norm, sin(theta_D), sin(theta_U),
       cos(theta_D - theta_U), normalized angular difference].

    Labels are synthetic PSO-like optimized nulling angles normalized to [0, 1].
    """
    rng = np.random.default_rng(seed)
    theta_d = rng.uniform(cfg.steering_min_deg, cfg.steering_max_deg, n_samples)
    theta_u = rng.uniform(cfg.steering_min_deg, cfg.steering_max_deg, n_samples)

    theta_d_rad = np.deg2rad(theta_d)
    theta_u_rad = np.deg2rad(theta_u)

    # Synthetic PSO-like labels. These nonlinear functions create structured
    # labels while keeping the demo independent from measured SI-channel data.
    null_d = -0.70 * theta_u + 18.0 * np.sin(theta_d_rad - 0.55 * theta_u_rad) + 8.0 * np.cos(2 * theta_d_rad)
    null_u = -0.65 * theta_d + 16.0 * np.sin(theta_u_rad + 0.45 * theta_d_rad) - 7.0 * np.cos(2 * theta_u_rad)

    # Add small synthetic optimization noise.
    null_d += rng.normal(0.0, 2.0, n_samples)
    null_u += rng.normal(0.0, 2.0, n_samples)

    null_d = np.clip(null_d, cfg.null_min_deg, cfg.null_max_deg)
    null_u = np.clip(null_u, cfg.null_min_deg, cfg.null_max_deg)
    null_d = _enforce_min_separation(null_d, theta_d, cfg)
    null_u = _enforce_min_separation(null_u, theta_u, cfg)

    theta_d_norm = _normalize(theta_d, cfg.steering_min_deg, cfg.steering_max_deg)
    theta_u_norm = _normalize(theta_u, cfg.steering_min_deg, cfg.steering_max_deg)
    angular_diff_norm = _normalize(theta_d - theta_u, -80.0, 80.0)

    features = np.column_stack(
        [
            theta_d_norm,
            theta_u_norm,
            np.sin(theta_d_rad),
            np.sin(theta_u_rad),
            np.cos(theta_d_rad - theta_u_rad),
            angular_diff_norm,
        ]
    ).astype(np.float32)

    labels = normalize_null_angles(np.column_stack([null_d, null_u]), cfg).astype(np.float32)

    meta = {
        "theta_d_deg": theta_d.astype(np.float32),
        "theta_u_deg": theta_u.astype(np.float32),
        "null_d_deg": null_d.astype(np.float32),
        "null_u_deg": null_u.astype(np.float32),
    }
    return features, labels, meta


def synthetic_mutual_coupling_score(
    theta_d_deg: np.ndarray,
    theta_u_deg: np.ndarray,
    pred_null_deg: np.ndarray,
    target_null_deg: np.ndarray,
    cfg: AngleConfig = AngleConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic baseline and predicted MC scores for visualization.

    More negative values represent better isolation. This is not a physical
    measured-channel model; it is only used to produce a portfolio demo plot.
    """
    spatial_hotspot = 4.0 * np.exp(-((theta_d_deg - 25.0) ** 2 + (theta_u_deg - 10.0) ** 2) / 450.0)
    diagonal_coupling = 3.0 * np.exp(-((theta_d_deg - theta_u_deg) ** 2) / 280.0)
    baseline_mc_db = -63.0 + spatial_hotspot + diagonal_coupling

    prediction_error = np.mean(np.abs(pred_null_deg - target_null_deg), axis=1)
    improvement_db = 10.0 * np.exp(-prediction_error / 10.0)

    sep_d = np.abs(pred_null_deg[:, 0] - theta_d_deg)
    sep_u = np.abs(pred_null_deg[:, 1] - theta_u_deg)
    separation_penalty = np.maximum(0.0, cfg.min_separation_deg - sep_d) + np.maximum(0.0, cfg.min_separation_deg - sep_u)

    predicted_mc_db = baseline_mc_db - improvement_db + 0.20 * separation_penalty
    return baseline_mc_db.astype(np.float32), predicted_mc_db.astype(np.float32), improvement_db.astype(np.float32)
