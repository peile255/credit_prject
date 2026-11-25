#!/bin/bash
#SBATCH --job-name=credit_full
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err

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
echo "Trustworthy Credit Default"
echo "Full Experiment: RQ1 RQ2 RQ3"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "Python: $PY"
echo "=================================="

"$PY" --version

"$PY" -m unittest discover -s tests

"$PY" -u run_all.py

"$PY" -u analyze.py --results results

echo "=================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "Finished: $(date)"
echo "=================================="
