#!/bin/bash
# Run core experiments (theoretical framework validation)
# Usage: ./run_experiments.sh [experiment_number]
# Example: ./run_experiments.sh 1  # Run only experiment 1
# Example: ./run_experiments.sh    # Run all experiments

cd "$(dirname "$0")"

if [ -n "$1" ]; then
    case $1 in
        1) python core_experiments/experiment1_ablation.py "${@:2}" ;;
        2) python core_experiments/experiment2_density_dropout.py "${@:2}" ;;
        3) python core_experiments/experiment3_noise_sweep.py "${@:2}" ;;
        4) python core_experiments/experiment4_maxpool_provenance.py "${@:2}" ;;
        5) python core_experiments/experiment5_isotropy.py "${@:2}" ;;
        ablations) python core_experiments/run_ablations.py "${@:2}" ;;
        kscale) python core_experiments/run_kscale_ablation.py "${@:2}" ;;
        *) echo "Unknown experiment: $1"; echo "Options: 1, 2, 3, 4, 5, ablations, kscale"; exit 1 ;;
    esac
else
    echo "=== Running All Core Experiments ==="
    echo "[1/5] Experiment 1: Per-Scale Encoding Ablation"
    python core_experiments/experiment1_ablation.py
    
    echo "[2/5] Experiment 2: Density Dropout"
    python core_experiments/experiment2_density_dropout.py
    
    echo "[3/5] Experiment 3: Noise Sweep"
    python core_experiments/experiment3_noise_sweep.py
    
    echo "[4/5] Experiment 4: MaxPool Provenance"
    python core_experiments/experiment4_maxpool_provenance.py
    
    echo "[5/5] Experiment 5: Isotropy Analysis"
    python core_experiments/experiment5_isotropy.py
    
    echo "=== All Core Experiments Complete ==="
fi
