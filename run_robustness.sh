#!/bin/bash
# Run robustness evaluations
# Usage: ./run_robustness.sh [eval_type]
# Example: ./run_robustness.sh train     # Train clean checkpoints first (required)
# Example: ./run_robustness.sh dropout   # Run dropout eval
# Example: ./run_robustness.sh all       # Run all robustness evals (after training)

cd "$(dirname "$0")"

if [ -n "$1" ]; then
    case $1 in
        train) python robustness_eval/train_clean_checkpoints.py "${@:2}" ;;
        dropout) python robustness_eval/run_dropout_robustness_eval.py "${@:2}" ;;
        noise) python robustness_eval/run_noise_robustness_eval.py "${@:2}" ;;
        outlier) python robustness_eval/run_outlier_robustness_eval.py "${@:2}" ;;
        npoints) python robustness_eval/run_npoints_val_only_eval.py "${@:2}" ;;
        few_shot) python robustness_eval/run_few_shot_eval.py "${@:2}" ;;
        all)
            echo "=== Running All Robustness Evaluations ==="
            echo "[1/5] Training Clean Checkpoints (required for other evals)"
            python robustness_eval/train_clean_checkpoints.py "${@:2}"
            
            echo "[2/5] Dropout Robustness"
            python robustness_eval/run_dropout_robustness_eval.py "${@:2}"
            
            echo "[3/5] Noise Robustness"
            python robustness_eval/run_noise_robustness_eval.py "${@:2}"
            
            echo "[4/5] Outlier Robustness"
            python robustness_eval/run_outlier_robustness_eval.py "${@:2}"
            
            echo "[5/5] N-Points Sensitivity"
            python robustness_eval/run_npoints_val_only_eval.py "${@:2}"
            
            echo "=== All Robustness Evaluations Complete ==="
            ;;
        *) echo "Unknown eval: $1"; echo "Options: train, dropout, noise, outlier, npoints, few_shot, all"; exit 1 ;;
    esac
else
    echo "Usage: ./run_robustness.sh [command]"
    echo ""
    echo "Commands:"
    echo "  train     - Train clean checkpoints (REQUIRED first)"
    echo "  dropout   - Run dropout robustness eval"
    echo "  noise     - Run noise robustness eval"
    echo "  outlier   - Run outlier robustness eval"
    echo "  npoints   - Run n-points sensitivity eval"
    echo "  few_shot  - Run few-shot learning eval"
    echo "  all       - Run train + all robustness evals"
    echo ""
    echo "Example: ./run_robustness.sh train && ./run_robustness.sh dropout"
fi
