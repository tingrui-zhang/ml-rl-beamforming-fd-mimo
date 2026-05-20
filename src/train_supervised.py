"""Train a supervised nulling-angle predictor on synthetic demo data."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from synthetic_data import generate_synthetic_dataset
from models import build_model
from losses import NullingAngleLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised nulling-angle prediction demo.")
    parser.add_argument("--model", choices=["dnn", "cnn"], default="dnn")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--n-samples", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, meta = generate_synthetic_dataset(n_samples=args.n_samples, seed=args.seed)
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x, y, np.arange(len(x)), test_size=0.2, random_state=args.seed
    )

    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model)
    criterion = NullingAngleLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    history = []
    start_time = time.time()
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss, main_loss, constraint_loss = criterion(pred, yb, xb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                loss, _, _ = criterion(pred, yb, xb)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        history.append((epoch, train_loss, val_loss))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | train loss={train_loss:.4f} | val loss={val_loss:.4f}")

    elapsed = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_name": args.model,
        "model_state_dict": model.state_dict(),
        "history": np.array(history, dtype=np.float32),
        "x_test": x_test,
        "y_test": y_test,
        "test_indices": idx_test,
        "meta": meta,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "elapsed_seconds": elapsed,
    }
    ckpt_path = out_dir / f"{args.model}_model.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Training time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
