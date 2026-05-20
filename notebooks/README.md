# Notebooks

A cleaned notebook walkthrough can be added here later.

For the current version, use the scripts in `src/`:

```bash
python src/train_supervised.py --model dnn --epochs 80 --n-samples 8000
python src/evaluate.py --checkpoint results/dnn_model.pt
python src/plot_results.py --checkpoint results/dnn_model.pt
```
