#!/bin/bash
#SBATCH --job-name=credit_report
#SBATCH --partition=aoraki
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/supp_report_%j.out
#SBATCH --error=logs/supp_report_%j.err
set -euo pipefail
cd "$HOME/credit_project"
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1
"$PWD/.conda/bin/python" supplementary/check.py "$@"
"$PWD/.conda/bin/python" supplementary/analyze.py "$@"
