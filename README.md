# MS-DGCNN++

**MS-DGCNN++: Multi-Scale Dynamic Graph Convolution with Scale-Dependent Normalization for Robust LiDAR Tree Species Classification**

| Resource | Link |
|----------|------|
| Unified 3D library (large benchmarks & preprocessing) | [github.com/said-ohamouddou/LIDARLearn](https://github.com/said-ohamouddou/LIDARLearn) |

---

## What this repository is for

**MS-DGCNN++** is a hierarchical multi-scale dynamic graph CNN with *scale-dependent edge encoding*: raw displacement features at the local scale (small neighborhoods, low SNR for directions) and hybrid raw-plus-normalized directional features at the intermediate scale (larger neighborhoods, high SNR). A noise-sensitivity argument shows that the mean squared error of normalized directions decays as **O(1/s²)** with neighbor spacing *s*, while raw displacements do not, which motivates the asymmetric encoding across scales.

This repository contains the **ablation and robustness experiments** on **STPCTLS** (5-fold cross-validation): per-scale encoding, upper-canopy density dropout, Gaussian noise, outliers, component ablations and *k*-scale sensitivity, together with the scripts that produce the paper's tables and figures.

The **large-scale state-of-the-art comparisons** on STPCTLS and HeliALS were run with **[LIDARLearn](https://github.com/said-ohamouddou/LIDARLearn)**, which integrates MS-DGCNN++ with shared preprocessing, configs and training scripts. Start there for dataset preprocessing and for replicating the benchmark tables.

## Experiments ↔ scripts

| Experiment | Paper | Entry point |
|------------|-------|-------------|
| Per-scale edge encoding (RO / HE / DA / RA variants) | Table 6 | `core_experiments/experiment1_ablation.py` |
| Upper-canopy density dropout (+ matched-training supplement) | Density figure / table | `core_experiments/experiment2_density_dropout.py` |
| Gaussian noise sweep (0–1000 mm) | Noise figure / table | `core_experiments/experiment3_noise_sweep.py` |
| Component ablations at 100 / 50 / 5 % retention | Component table | `core_experiments/run_ablations.py` |
| Outlier robustness (0–100 % outliers) | Table 5 | `core_experiments/run_ablations.py --outliers` |
| *k*-scale sensitivity at 100 / 5 % retention | Table 7 | `core_experiments/run_kscale_ablation.py` |

The robustness studies compare MS-DGCNN++ against the baselines defined in `models/baselines.py`: DGCNN (*k*₁=5, *k*=30), MS-DGCNN and Point-M2AE.

## Protocol

- **Train clean, test corrupted.** Every model is trained once on clean data; density dropout, noise and outliers are applied only to the validation clouds of each fold.
- **One training recipe.** All scripts share the same settings (`PAPER_TRAINING` in `utils/experiment_utils.py`): 5 folds, 100 epochs, batch size 16, lr 5e-4, weight decay 1e-4, 1024 points, seed 42. Scripts refuse to run with different values, so that a model configuration always maps to the same checkpoint in every table and figure.
- **Shared checkpoint cache.** Trained folds and evaluations are stored in `results/paper_shared_v2/` under a hash of the model configuration, code, data and splits. Later scripts reuse checkpoints from earlier ones instead of retraining, and an interrupted run resumes where it stopped.
- **New corruption levels** can be evaluated from the saved checkpoints without retraining, e.g.

  ```bash
  python core_experiments/experiment2_density_dropout.py --eval_only --retention_rates 1.0 0.03
  python core_experiments/experiment3_noise_sweep.py --eval_only --noise_levels 0 25 250
  ```

## Repository layout

| Path | Purpose |
|------|---------|
| `models/` | `msdgcnn2.py` (MS-DGCNN++), `msdgcnn.py` (MS-DGCNN), `baselines.py`, `pointm2ae/` (Point-M2AE with pretrained weights) |
| `core_experiments/` | The six experiment scripts listed above |
| `utils/` | CV splits, dataloaders and corruptions, training loop and cache (`experiment_utils.py`), LaTeX tables, plot style |
| `data/STPCTLC/` | STPCTLS HDF5 file and CV split JSON |
| `results/paper_v2/` | Per-experiment outputs: configs, metrics JSON, LaTeX tables, figures |
| `results/paper_shared_v2/` | Shared checkpoint and evaluation cache |
| `logs/` | Logs written by `run_all.sh` |
| `run_all.sh`, `watch_progress.sh` | Run every experiment in order; live progress view |

The directory name `STPCTLC` follows this codebase; the dataset is called **STPCTLS** in the paper.

## Requirements

- Python 3.10+
- NVIDIA GPU recommended (falls back to CPU if CUDA is unavailable)

Install **PyTorch** for your platform from [pytorch.org](https://pytorch.org), then:

```bash
pip install -r requirements.txt
```

MS-DGCNN++ uses pure PyTorch *k*-NN graph features. The **Point-M2AE** baseline (`models/pointm2ae/`) is adapted for these experiments and may need its CUDA extensions (`pointnet2_ops`, `knn_cuda`, …); for the official release see [github.com/zrrskywalker/point-m2ae](https://github.com/zrrskywalker/point-m2ae).

The Point-M2AE ShapeNet-pretrained weights (175 MB) are not included in this repository. Download them from the official Point-M2AE release and save them as `models/pointm2ae/pretrained_m2ae.pth`; they are only needed for the Point-M2AE baseline.

## Data (STPCTLS)

The repository includes the STPCTLS point clouds (one folder per species), the HDF5 file used by the loaders and the CV splits:

```
data/STPCTLC/point_cloud_data.h5
data/STPCTLC/cv_splits_k5_seed42.json
```

Coordinates are expected in meters (`--coordinate_unit m`, the default). Stratified 5-fold splits are created on first run and reused afterwards (`data/STPCTLC/cv_splits_k5_seed42.json`). Relative paths (`--data_path`, `--output_dir`, `--cache_dir`) are resolved from the repository root.

## Running

### All experiments

```bash
bash run_all.sh
```

Runs the six steps in order, showing one line per epoch in the terminal and writing full logs to `logs/<step>.log` and `logs/run_all.log`. Re-running resumes: finished folds and evaluations are reused. Edit `run_all.sh` to point it at your own conda environment.

To follow a running job from another terminal:

```bash
bash watch_progress.sh
```

Do not edit `models/`, `utils/` or `core_experiments/` while a run is in progress: the code is part of the checkpoint key, so a change makes later steps retrain instead of reuse.

### Single experiment

```bash
python core_experiments/experiment1_ablation.py
python core_experiments/experiment2_density_dropout.py
python core_experiments/experiment3_noise_sweep.py
python core_experiments/run_ablations.py
python core_experiments/run_ablations.py --outliers
python core_experiments/run_kscale_ablation.py
```

Use `--help` on any script for its options (corruption levels, `--eval_only`, `--device`, `--output_dir`, …).

## Outputs

Each run writes to a timestamped folder under `results/paper_v2/` (`config.json`, per-fold evaluations, aggregated results JSON, `tables/` with LaTeX tables and `figures/` with PDF figures). Trained checkpoints live in `results/paper_shared_v2/`.
