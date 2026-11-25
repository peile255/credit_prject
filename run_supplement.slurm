#!/bin/bash
#SBATCH --job-name=credit_sup
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/supp_%A_%a.out
#SBATCH --error=logs/supp_%A_%a.err
#SBATCH --array=1-6%3
set -euo pipefail
cd "$HOME/credit_project"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY="$PWD/.conda/bin/python"
"$PY" -u supplementary/run.py "s${SLURM_ARRAY_TASK_ID}" "$@"
