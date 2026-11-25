#!/bin/bash
#SBATCH --job-name=credit_smoke
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -e

cd "$HOME/credit_project"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY="$HOME/credit_project/.conda/bin/python"

echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Python: $PY"
echo "=================================="

"$PY" --version

"$PY" -m unittest discover -s tests

"$PY" -u run_all.py --smoke

"$PY" -u analyze.py --results results_smoke

echo "=================================="
echo "Finished: $(date)"
echo "=================================="
