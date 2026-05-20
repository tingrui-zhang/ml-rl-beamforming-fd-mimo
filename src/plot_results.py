"""Generate plots for the supervised nulling-angle prediction demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models import build_model
from synthetic_data import denormalize_null_angles, synthetic_mutual_coupling_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot supervised nulling-angle prediction demo results.")
    parser.add_argument("--checkpoint", type=str, default="results/dnn_model.pt")
    parser.add_argument("--out-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    history = ckpt["history"]
    x_test = ckpt["x_test"]
    y_test = ckpt["y_test"]
    idx = ckpt["test_indices"]
    meta = ckpt["meta"]

    with torch.no_grad():
        pred = model(torch.tensor(x_test)).numpy()

    pred_deg = denormalize_null_angles(pred)
    target_deg = denormalize_null_angles(y_test)
    theta_d = meta["theta_d_deg"][idx]
    theta_u = meta["theta_u_deg"][idx]
    baseline_mc, predicted_mc, improvement = synthetic_mutual_coupling_score(theta_d, theta_u, pred_deg, target_deg)
    mc_gain = baseline_mc - predicted_mc

    # Training curve
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(history[:, 0], history[:, 1], label="Train")
    plt.plot(history[:, 0], history[:, 2], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png", dpi=200)
    plt.close()

    # Predicted vs target nulling angles
    plt.figure(figsize=(5.6, 5.2))
    plt.scatter(target_deg[:, 0], pred_deg[:, 0], s=8, alpha=0.55, label="DL null angle")
    plt.scatter(target_deg[:, 1], pred_deg[:, 1], s=8, alpha=0.55, label="UL null angle")
    lims = [-90, 90]
    plt.plot(lims, lims, linestyle="--", linewidth=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Target nulling angle (deg)")
    plt.ylabel("Predicted nulling angle (deg)")
    plt.title("Predicted vs Target Nulling Angles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_scatter.png", dpi=200)
    plt.close()

    # Synthetic MC improvement map
    plt.figure(figsize=(6.4, 5.2))
    scatter = plt.scatter(theta_d, theta_u, c=mc_gain, s=12, alpha=0.8, cmap="coolwarm")
    plt.xlabel("DL steering angle (deg)")
    plt.ylabel("UL steering angle (deg)")
    plt.title("Synthetic Demo: MC Improvement over Baseline")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Synthetic MC improvement (dB)")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "mc_improvement_map.png", dpi=200)
    plt.close()

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
