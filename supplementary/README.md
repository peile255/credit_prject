# Credit default supplementary experiments (S1–S6)

Install by extracting this ZIP into the existing `~/credit_project`, not a new standalone project. Requires the original `data/credit.csv`, `src/`, `run_all.py`, `config.yaml`, and completed `results/` from Aoraki job 6565710. Uses the existing `.conda/bin/python`; no extra dependencies. Original files are read-only to these scripts. Output roots are `results_supplementary_smoke/` and `results_supplementary/`.

## Stages
- S1: seeds 42–44; 20 randomized candidates per base model and weighting setting, group-aware 5-fold AP optimization on training data only. For strict stacking OOF, tuning repeats within each OOF training fold (nested CV). 200 trees/iterations remain fixed. Voting and stacking use the tuned base learners; meta C=1. Compare on identical original partitions. 42 final evaluations. This is a sensitivity analysis, not a replacement for the ten-split baseline.
- S2: all ten seeds; 2x2 base/meta weighting x raw/Platt/isotonic, 120 evaluations. Meta training uses grouped OOF predictions. Calibration and threshold selection use the original calibration partition.
- S3: first ten saved development bootstrap ID arrays; three model variants (weighted stacking raw/Platt/isotonic), nine matched budgets (16/32/64 background rows x 4/8/16 forward/reverse permutation cycles), 40 fixed original local cases, 10 per stratum. Original 16 background rows retained, additional rows nested and selected from training. Fixed permutation seed for comparisons. 1,080 instance/config rows. Per-bootstrap attribution arrays and times saved. This is interventional Monte Carlo Shapley, not TreeSHAP.
- S4: retrains original reference and 30 development +10 held-out bootstrap models using original saved IDs. Verifies reference probabilities against stored predictions before full candidate evaluation (max absolute tolerance 1e-5). Original 100 random high-risk cases plus four threshold-distance strata, up to 100 each; overlapping cohorts reported separately. Credit-limit-only changes within training 1st/99th percentiles, rounded to 1,000, radius ±10/20/30/50%, 25 grid points per radius. Nearest, Robust, High gate, Moderate-or-High gate, and separate risk-reduction scenario. Held-out models never used for selection. Coverage for RiskReduction means risk reduction, NOT class flip. Report unconditional metrics with abstention zero and conditional held-out metrics. No causal/actionable recourse claims. Original cohort has 2,000 rows (100×4×5); stratified cohorts may have fewer than 100 each. Gate levels calculated using 30 development models, not the held-out ten. Candidate ranking for both gate methods is unchanged from Robust; gate only changes abstention. RiskReduction picks largest reference probability reduction; it is not a threshold-crossing algorithm.
- S5: reuses reference test predictions; paired predictor-group bootstrap (2,000 draws), 26 existing contrasts plus six ECE contrasts; raw metric difference as effect size, percentile CI, approximate centered bootstrap two-sided p-values, Holm within RQ. Conditional on fitted models and fixed reference split. ECE and sparse-event inference are approximate. CIs are pointwise, not simultaneous. No claim that ten overlapping splits are independent.
- S6: original 80 local cases, Platt model top-5, six pairwise strata comparisons × four metrics; Mann–Whitney U (distribution comparison), Cliff's delta, mean-difference bootstrap CI, Holm across 24 contrasts. Conditional on shared fitted bootstrap models. Original local predictor rows checked to have no duplicates. Exploratory, not causal.
- Report: checks expected sizes and ranges, creates 7 CSV summaries and six figures in PNG/PDF. S4 adds paired cluster-bootstrap contrasts for coverage and unconditional held-out threshold validity, Holm across all reported cohort/radius/method/metric contrasts; shared cohorts are not independent. Sparse results have low power and bootstrap may be degenerate; interpret as exploratory. No test-set tuning.

## Run
From existing project on Aoraki:
```
cd ~/credit_project
mkdir -p logs
unzip -n ~/supplementary.zip
bash -n run_supplement.slurm
bash -n run_supplement_report.slurm
SMOKE=$(sbatch --parsable run_supplement.slurm --smoke)
SMOKE=${SMOKE%%;*}
echo "$SMOKE" > supplementary_smoke_job_id.txt
sbatch --dependency=afterok:$SMOKE run_supplement_report.slurm --smoke
```
After all six smoke tasks and report complete successfully:
```
FULL=$(sbatch --parsable run_supplement.slurm)
FULL=${FULL%%;*}
echo "$FULL" > supplementary_full_job_id.txt
REPORT=$(sbatch --parsable --dependency=afterok:$FULL run_supplement_report.slurm)
echo "${REPORT%%;*}" > supplementary_report_job_id.txt
```
Array runs up to three stages simultaneously, each 4 CPUs, 16 GB, 24 h limit. S1 nested tuning is intentionally much heavier than original experiment. These resource settings are requests, not measured full-stage needs. Use sacct and logs to monitor. On failure, rerun only failed tasks e.g. `sbatch --array=1 run_supplement.slurm` (no `--force`); then manually submit report once all stages complete. Do not run duplicate copies of the same stage concurrently. Completed stages skip; S1/S2 resume seed/weight or seed chunks, S3/S4 resume per-bootstrap arrays, S5/S6 restart their statistical stage if interrupted. Fingerprint guards changed source/config/input/version; archive old stage output before changing settings or use `--force` to recompute that stage. S4 still retrains the reference at each resume; S3/S4 saved checkpoint arrays rely on deterministic fixed selections. If modifying smoke or full design, remove/rename old stage results first. Original results never overwritten.

## Validation
Local smoke execution against the uploaded completed archive: all six stages completed, 6 output integration checks passed, 7 report CSVs and 12 figure files produced; completed-stage skip verified. Full-scale supplementary run has NOT been executed on Aoraki. Smoke uses reduced training subsets/trees/folds/trials/bootstraps; S5/S6 reuse original reference results with fewer statistical resamples. No scientific conclusion should be drawn from smoke outputs. Aoraki environment may differ from local test environment; check logs and original-reference reproduction guard.
