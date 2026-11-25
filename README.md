# Trustworthy Credit Default Prediction

Single-dataset reproducible empirical study / 单数据集信用违约预测实验项目。

**Research title:** Trustworthy Credit Default Prediction via Calibrated Ensemble Learning and Stability-Aware Explanations

This is an experimental implementation, not a completed paper. The supplied data were decoded from the uploaded original Excel workbook. No full-study results are claimed. `results_smoke/` contains an explicitly labelled small-scale integration test, not publication results.

## Quick start / 运行步骤

Recommended Python 3.11 or 3.12; tested in the authoring environment on Python 3.13 with package versions recorded in `results_smoke/audit.json`. Use an isolated environment. Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python run_all.py --smoke
python analyze.py --results results_smoke
python run_all.py
python analyze.py --results results
```

On Linux/macOS activate with `source .venv/bin/activate`. For reproducibility keep the audit package versions and config with results. Limit BLAS threads if needed (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` in Bash). The full experiment is CPU-intensive; first measure the smoke and pilot runs. `config.yaml` is the prespecified full configuration. Do not reduce it after inspecting test results. Use `--output results_new` to avoid overwriting a prior run, and `--config path.yaml` to specify a separate configuration. `analyze.py` requires a successful completion marker.

`data/credit.csv` is ready to use, so no download or Excel reader is required for experiments. To independently regenerate it from the included source archive run `python prepare_data.py` after installing xlrd. The original data have 30,000 rows, 23 predictors, 6,636 defaults and no empty numeric cells; 56 repeated predictor rows are grouped during splitting. IDs are excluded from predictors.

## Files

- `data/source.zip`: original user-uploaded archive; `credit.csv`: extracted numeric values, with target renamed `default`.
- `docs/EXPERIMENT_DESIGN.md`: detailed RQs, controlled comparisons, interpretation and limitations.
- `src/models.py`: model fit, weighting, grouped OOF stacking, calibration.
- `src/explain.py`: interventional permutation-based Monte Carlo Shapley attributions with additive check, uncertainty aggregation.
- `src/evaluation.py`: metrics and paired bootstrap effects, stability.
- `run_all.py`: data audit, splits, RQ1–RQ3, robustness and manifests.
- `analyze.py`: descriptive tables, paired RQ3 tests with Holm correction, risk bands and eight figures (PNG/PDF).
- `tests/test_core.py`: dataset, split, metric, attribution unit tests.
- `results_smoke`: reduced-scale outputs, generated on a 1,500-row random subset. Never use these for paper conclusions.

## Outputs

`results/audit.json`, `split_manifest.csv`, `reference_predictions.csv`, `paired_effects.csv`, `rq1_metrics.csv`, `rq2_calibration.csv`, `rq2_reliability_bins.csv`, `rq2_missing_robustness.csv`, `rq2_bootstrap_risk.csv`, `rq3_stability.csv`, `rq3_attributions.csv`, `rq3_global.csv`, `rq3_stability_validation.csv`, `rq3_shap_budget_sensitivity.csv`, `rq3_counterfactual.csv`, `shap_values.npz`, `shap_background.npy`, `bootstrap_ids.npy`, `explanation_manifest.csv`, `runtime.csv`, `shap_runtime.csv`, `complete.json`, and generated `tables/` and `figures/`.

The default configuration uses fixed, declared hyperparameters rather than post-hoc tuning. No claims of best possible predictive accuracy. All conclusions are conditional on the single historical benchmark. Counterfactual scenarios change only credit limit under a hypothetical lender-controlled assumption; they are not borrower advice, proof of causal impact, or a deployable credit decision system. Demographic features are retained to reproduce the benchmark, not endorsed for credit decisions. Fairness and temporal external validation are outside the primary scope.

## Data citation

Yeh, I.-C. (2016). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H . Original publication: Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473–2480. Cite the dataset and check its current terms before redistribution.
