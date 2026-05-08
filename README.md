# MS-DGCNN++

**MS-DGCNN++: Multi-Scale Dynamic Graph Convolution with Scale-Dependent Normalization for Robust LiDAR Tree Species Classification**

| Resource | Link |
|----------|------|
| Preprint | [arXiv:2507.12602](https://arxiv.org/abs/2507.12602) · [PDF](https://arxiv.org/pdf/2507.12602.pdf) |
| Unified 3D library (large benchmarks & preprocessing) | [github.com/said-ohamouddou/LIDARLearn](https://github.com/said-ohamouddou/LIDARLearn) |

---

## What this repository is for

**MS-DGCNN++** is a hierarchical multi-scale dynamic graph CNN with *scale-dependent edge encoding*: raw displacement features at the local scale (small neighborhoods, low SNR for directions) and hybrid raw-plus-normalized directional features at an intermediate scale (larger neighborhoods, high SNR). A theoretical noise-sensitivity argument motivates why normalized directions have mean squared error decaying as **O(1/s²)** with neighbor spacing *s*, while raw displacements do not—supporting asymmetric encoding across scales.

This repo holds the **structured ablation suite**, **robustness evaluations**, training utilities, and tables/figures pipelines used in **Section “Ablation studies and robustness”** of `sn-article.tex`.

The **large-scale state-of-the-art comparisons** on **STPCTLS** (seven species, terrestrial laser scanning) and **HeliALS** (nine species, airborne laser scanning, geometry-only), including **56 models** on STPCTLS as reported in the paper, were run using our open-source library **[LIDARLearn](https://github.com/said-ohamouddou/LIDARLearn)**, which integrates **MS-DGCNN++** with shared preprocessing, configs, and training scripts. For **dataset preprocessing**, multispectral / ALS workflows, and replication of those benchmark tables, start from **LIDARLearn** (`preprocessing/`, `datasets/`, `scripts/`, `docs/`).

## Paper experiments ↔ scripts

| § Experiment | Topic | Entry point |
|--------------|--------|-------------|
| Per-scale encoding | Raw vs hybrid vs asymmetric normalization | `core_experiments/experiment1_ablation.py` |
| Density dropout | Canopy thinning / retention | `core_experiments/experiment2_density_dropout.py` |
| Noise sweep | SNR crossover and degradation | `core_experiments/experiment3_noise_sweep.py` |
| Max-pooling provenance | Which neighbors win max pooling | `core_experiments/experiment4_maxpool_provenance.py` |
| Isotropy / effective rank | Feature-space geometry | `core_experiments/experiment5_isotropy.py` |
| Component / fusion / *k*-scale | Broader ablations | `core_experiments/run_ablations.py`, `run_kscale_ablation.py` |

Batch runners: [`run_core_experiments.sh`](run_core_experiments.sh), [`run_robustness.sh`](run_robustness.sh) (noise, outliers, dropout, *n*-points, few-shot—after training clean checkpoints).

## Repository layout

| Path | Purpose |
|------|---------|
| `models/` | `msdgcnn2.py` (MS-DGCNN++), `msdgcnn.py`, optional `pointm2ae/` baseline |
| `core_experiments/` | Experiments 1–5 and aggregated ablation drivers |
| `robustness_eval/` | Checkpoint training + perturbation protocols |
| `utils/` | CV splits, dataloaders, training loop, plots, LaTeX helpers |
| `data/STPCTLC/` | Local layout for STPCTLS HDF5 and CV split JSON |
| `results/` | Example outputs (JSON, TeX, figures) |

The directory name `STPCTLC` follows this codebase; the dataset is referred to as **STPCTLS** in `sn-article.tex`.

## Requirements

- Python 3.10+ recommended  
- NVIDIA GPU recommended (training falls back to CPU if CUDA is unavailable)

```bash
pip install -r requirements.txt
```

Install **PyTorch** for your stack from [pytorch.org](https://pytorch.org) (CUDA wheel vs CPU).

### Point-M2AE baseline

The **`models/pointm2ae/`** copy here is adapted for these experiments. For the official Point-M2AE release (paper code, weights, full dependencies), see **[github.com/zrrskywalker/point-m2ae](https://github.com/zrrskywalker/point-m2ae)**.

MS-DGCNN++ core paths use pure PyTorch *k*-NN graph features. Enabling the Point-M2AE baseline in this repo may still require CUDA extensions (`pointnet2_ops`, `knn_cuda`, etc.).

## Data (STPCTLS)

Place the HDF5 bundle expected by the loaders as:

`data/STPCTLC/point_cloud_data.h5`

On first run, stratified CV splits are created or reused, e.g. `data/STPCTLC/cv_splits_k5_seed42.json`.

**Path tip:** `core_experiments/*.py` joins relative `--data_path` with the `core_experiments/` directory. From the repo root use an absolute path:

```bash
python core_experiments/experiment1_ablation.py --data_path "$(pwd)/data/STPCTLC"
```

## Running experiments

From the repository root:

### Core experiments

```bash
chmod +x run_core_experiments.sh
./run_core_experiments.sh           # experiments 1–5
./run_core_experiments.sh 1       # single experiment
./run_core_experiments.sh ablations
./run_core_experiments.sh kscale
```

### Robustness

```bash
chmod +x run_robustness.sh
./run_robustness.sh train
./run_robustness.sh dropout
./run_robustness.sh noise
./run_robustness.sh outlier
./run_robustness.sh npoints
./run_robustness.sh few_shot
./run_robustness.sh all
```

Use `--help` on individual Python scripts for epochs, batch size, folds, and output dirs.

## Outputs

Artifacts land under `results/` or `--output_dir`: configs, metrics JSON, LaTeX tables, and figures aligned with the paper’s supplementary material workflow.

## Citation

Please cite the arXiv preprint and the original **STPCTLS** / **HeliALS** dataset publications referenced in `sn-article.tex`. When you use the unified benchmarking stack, cite **[LIDARLearn](https://github.com/said-ohamouddou/LIDARLearn)** as indicated there.

```bibtex
@misc{ohamouddou2025msdgcnnpp,
  title         = {{MS-DGCNN++}: Multi-Scale Dynamic Graph Convolution with Scale-Dependent Normalization for Robust {LiDAR} Tree Species Classification},
  author        = {Ohamouddou, Said and El Afia, Hanaa and Boulaich, Mohamed Hamza and El Afia, Abdellatif and Chiheb, Raddouane},
  year          = {2025},
  eprint        = {2507.12602},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2507.12602}
}
```
