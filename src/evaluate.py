"""Evaluate a trained nulling-angle predictor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from models import build_model
from synthetic_data import denormalize_null_angles, synthetic_mutual_coupling_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate supervised nulling-angle prediction demo.")
    parser.add_argument("--checkpoint", type=str, default="results/dnn_model.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train a model first.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_test = ckpt["x_test"]
    y_test = ckpt["y_test"]
    idx = ckpt["test_indices"]
    meta = ckpt["meta"]

    with torch.no_grad():
        pred = model(torch.tensor(x_test)).numpy()

    pred_deg = denormalize_null_angles(pred)
    target_deg = denormalize_null_angles(y_test)
    mae_deg = np.mean(np.abs(pred_deg - target_deg), axis=0)

    theta_d = meta["theta_d_deg"][idx]
    theta_u = meta["theta_u_deg"][idx]
    baseline_mc, predicted_mc, improvement = synthetic_mutual_coupling_score(theta_d, theta_u, pred_deg, target_deg)

    print("Evaluation summary")
    print("------------------")
    print(f"Model: {ckpt['model_name']}")
    print(f"Test samples: {len(x_test)}")
    print(f"MAE null D angle: {mae_deg[0]:.2f} deg")
    print(f"MAE null U angle: {mae_deg[1]:.2f} deg")
    print(f"Average synthetic baseline MC:  {baseline_mc.mean():.2f} dB")
    print(f"Average synthetic predicted MC: {predicted_mc.mean():.2f} dB")
    print(f"Average synthetic improvement:  {(baseline_mc - predicted_mc).mean():.2f} dB")


if __name__ == "__main__":
    main()
