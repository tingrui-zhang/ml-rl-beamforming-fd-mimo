# ML/RL-Based Beamforming Demo for Full-Duplex mMIMO

This repository provides a **simplified public demo** inspired by my research on machine-learning-based nulling-control beamforming for full-duplex massive MIMO systems.

The current version focuses on a supervised-learning demo: a neural network predicts nulling-control angles from downlink/uplink steering-angle conditions. Synthetic labels are used to mimic PSO-optimized nulling-angle labels.

> **Note:** This repository is for portfolio and educational purposes. It does **not** include proprietary measured self-interference channel data, S-parameter measurements, unpublished research code, or lab-owned datasets.

## Background

Full-duplex massive MIMO can improve spectral efficiency by allowing simultaneous transmission and reception. However, strong self-interference between co-located transmit and receive arrays can severely degrade receiver performance.

Nulling-control beamforming improves Tx--Rx isolation by placing radiation nulls toward dominant coupling directions while maintaining the desired beam direction. In my research, supervised learning models are trained using PSO-optimized nulling-angle labels to enable fast beam-control inference.

## Demo Task

**Input features**

- Downlink steering angle
- Uplink steering angle
- Simple engineered angular features

**Output labels**

- Downlink nulling angle
- Uplink nulling angle

**Model options**

- Fully connected DNN nulling-angle predictor
- Lightweight 1D CNN nulling-angle predictor

## Repository Structure

```text
ml-rl-beamforming-fd-mimo/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── synthetic_data.py
│   ├── models.py
│   ├── losses.py
│   ├── train_supervised.py
│   ├── evaluate.py
│   └── plot_results.py
└── results/
```

## Quick Start

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Train a DNN model using synthetic data:

```bash
python src/train_supervised.py --model dnn --epochs 80 --n-samples 8000
```

Evaluate the trained model:

```bash
python src/evaluate.py --checkpoint results/dnn_model.pt
```

Generate result plots:

```bash
python src/plot_results.py --checkpoint results/dnn_model.pt
```

## Example Outputs

The scripts generate outputs such as:

- `results/training_loss.png`
- `results/prediction_scatter.png`
- `results/mc_improvement_map.png`

The mutual-coupling values in this demo are **synthetic scoring metrics** designed for visualization only. They are not the measured-channel results reported in the paper.

## Related Publication

T. Zhang, Y. Gong, and T. Le-Ngoc, “Supervised Learning for Nulling Angle Prediction with Measured Self-Interference Channels for Full-Duplex Isolation Enhancement,” *Proc. IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN)*, 2025.

## Future Work

Planned extensions:

- Add a simplified reinforcement-learning environment for nulling-control beamforming.
- Add MIMO beam-pattern and LCMV-based null-steering visualization examples.
- Add a notebook walkthrough for the supervised-learning pipeline.
