# Insurance Risk Modeling

End-to-end loss-cost regression and claim-status classification on a ~40k-policy auto insurance dataset. Tweedie-loss gradient boosting for the regression side; tuned LightGBM with isotonic calibration for the classification side. Final deliverables include calibrated risk scores that can feed directly into a premium formula.

---

## Executive summary

Auto insurers face a fundamental pricing tradeoff: charge a high-risk policyholder too little and the book bleeds on claims; charge a careful driver too much and they walk to a competitor. I built two models on 39,928 policies to drive that pricing decision: a Tweedie-loss regression for expected loss cost (best 5-fold CV MSE 257,151) and an isotonic-calibrated LightGBM classifier for claim status (validation ROC-AUC 0.7865, Brier 0.086, top-decile lift 3.2x). The top decile of predicted claim scores captures roughly 32% of all actual claims, giving the underwriting team a clean lever for tiered review. The single largest quality improvement in the pipeline was probability calibration (47% Brier reduction), which is what makes the classifier's output usable as a probability rather than only as a rank.

## Problem

Insurance pricing hinges on two predictive questions about every policy:

1. **Regression.** What loss is this policy likely to generate over the year? Specifically `LC` (loss cost per exposure unit) and `HALC` (a historically-adjusted variant used by the actuarial team).
2. **Classification.** Will this policy file at least one claim (`CS`)? This drives reserve setting and risk segmentation.

Both targets are heavily zero-inflated (about 89% of policies claim nothing in a year) and `CS` is moderately imbalanced (11.15% positive class). Those two facts drive every modeling choice below.

## Data overview

|  | Train | Test |
|---|---|---|
| Rows | 39,928 | 13,310 |
| Features | 25 | 25 |
| Targets | LC, HALC, CS | (held-out, no labels) |
| Missing values | 0 | 0 |
| LC zero rate | 88.9% | n/a |
| CS positive rate | 11.15% | n/a |

Features are anonymized `X.*` columns covering driver attributes (age, driving experience), vehicle attributes (year, horsepower, cylinder, market value, weight), policy attributes (premium, tenure, distribution channel, payment frequency), and historical behaviour (prior cancellations, renewal gap). Four claim-derived columns (`X.15`-`X.18`) are excluded from the feature set to prevent leakage; the held-out test file ships with those columns already stripped.

## Modeling approaches

### Regression (LC, HALC)

I treated the zero-inflated loss distribution as a Tweedie process. Each model below was evaluated with the same 5-fold CV, MSE metric. The progression below is the order I tried things in; each addition was motivated by a specific weakness of the model before it.

| Model | LC CV-MSE | HALC CV-MSE | Why I tried it |
|---|---|---|---|
| Tweedie GLM (best p=1.1) | 260,385 | 1,076,974 | Linear baseline; the standard actuarial starting point for Tweedie data |
| XGBoost baseline (Tweedie) | 259,418 | 1,071,200 | Adds non-linearity and feature interactions over the GLM |
| LightGBM baseline (Tweedie) | 261,493 | 1,080,561 | Cross-check vs XGBoost; different tree growth strategy |
| Two-part frequency x severity | 264,453 | 1,087,922 | Tests whether decoupling P(claim) and E[claim\|claim>0] helps |
| LightGBM tuned (RandomizedSearchCV) | 258,598 | 1,069,206 | Tighten the boosting model after the cross-check |
| **XGBoost tuned (RandomizedSearchCV)** | **257,151** | **1,065,497** | **Selected: best on both targets** |

**Selected model.** Tuned XGBoost with Tweedie objective at `tweedie_variance_power=1.1`. The tuned `p` is itself informative: the data behaves closer to a compound Poisson process (frequency-dominated) than to a heavy-tailed gamma (severity-dominated). The improvement over the GLM baseline is about 1.2% on LC and 1.1% on HALC, which is small but consistent and shows up across both targets.

The two-part model is the worst in the comparison. Decoupling the two stages discards information: the same features predict both whether a claim occurs and how big it is, and the single-model Tweedie boosters pool those signals through one loss surface. I kept the two-part result in the comparison as a documented null rather than removing it.

### Classification (CS)

For classification I worked up from a regularised logistic baseline through tree ensembles, randomized hyperparameter search, calibration, and a stacking ensemble. All metrics below are on a stratified 80/20 validation holdout.

| Model | ROC-AUC | PR-AUC | Brier | Why I tried it |
|---|---|---|---|---|
| Logistic, unweighted | 0.7232 | 0.237 | 0.093 | Linear baseline (standard in insurance pricing) |
| Logistic, balanced | 0.7239 | 0.233 | 0.215 | Test whether class weighting helps |
| Logistic + SMOTE | 0.7190 | 0.228 | 0.215 | Test resampling vs reweighting |
| Random Forest (balanced) | 0.7691 | 0.299 | 0.147 | Non-boosting tree baseline |
| XGBoost (scale_pos_weight) | 0.7799 | 0.333 | 0.147 | Boosted trees with native imbalance handling |
| LightGBM (class_weight balanced) | 0.7880 | 0.348 | 0.162 | Cross-check vs XGBoost |
| XGBoost tuned | 0.7840 | 0.328 | 0.178 | Randomized hyperparameter search |
| LightGBM tuned | 0.7869 | 0.350 | 0.165 | Randomized hyperparameter search |
| Stacking ensemble | 0.7886 | 0.344 | 0.191 | Combine model families through a meta-learner |
| **LightGBM tuned + isotonic calibration** | **0.7865** | **0.344** | **0.086** | **Selected: calibrated probabilities for the pricing formula** |

**Selected model and rationale.** Tuned LightGBM with isotonic calibration. Three reasons over the stacking ensemble:

1. **It satisfies my pre-registered decision rule.** Before training the stack I committed to adopting it only if it beat the best single model by at least 0.001 ROC-AUC. The stack beat tuned LightGBM by 0.0006. That is below the threshold and inside the bootstrap CI for sampling noise on this validation set.
2. **It produces calibrated probabilities.** The downstream use is plugging the score into a premium formula, which requires a probability that matches the true conditional claim rate. Isotonic calibration drops Brier score from 0.162 to 0.086, a 47% reduction; the stack's Brier is 0.191. The stack ranks slightly better but it is unusable for pricing.
3. **It is operationally simpler.** A single booster plus a calibrator is far cheaper to deploy, monitor, and refresh than a four-model stack with a meta-learner.

## Key findings

1. **Tenure is the strongest single signal.** `policy_duration` is the top SHAP feature for both regression and classification. The empirical claim rate falls steadily from roughly 21% in the shortest-tenure decile to under 5% in the longest. Long-tenured customers self-select into lower risk.
2. **Cancellations and premium history are the next most important features.** `X.12` (prior cancellations) and `X.14` (net premium) rank second and third on SHAP for both tasks.
3. **The classification model concentrates risk usefully.** The top decile of predicted CS scores captures 32% of actual claims at 3.2x lift over the base rate. The top two deciles cover 52%; the top four cover 79%.
4. **The same features drive frequency and severity.** Both the regression and the classification models rank the same features at the top, which justifies treating loss cost as a single Tweedie process rather than splitting into a frequency model and a severity model.
5. **Calibration is the most valuable single step in the pipeline.** A 47% Brier improvement from isotonic calibration is larger than any gain from algorithm choice or hyperparameter tuning.

## Business recommendations

1. **Use calibrated probabilities directly in pricing, not just for ranking.** The isotonic-calibrated LightGBM score approximates the true conditional claim probability and can be multiplied through a premium formula. Stacked or raw boosted probabilities cannot.
2. **Treat the top decile as the underwriting review tier.** Capturing 32% of claims while reviewing 10% of policies is a roughly 3x efficiency gain over uniform sampling.
3. **Replace the default 0.5 cutoff with a cost-sensitive threshold.** At an assumed 10:1 false-negative-to-false-positive cost ratio, the optimal threshold is well below 0.5 and recovers about 82% of true claimants at the cost of flagging 38% of the book.
4. **Monitor AUC by tenure cohort.** The model is somewhat weaker on the longest-tenure quartile (AUC 0.70 vs 0.78 overall). Either a stratified model or a tenure-aware loss reweighting is worth piloting if the gap persists.
5. **Attach SHAP reason codes to adverse pricing decisions.** Regulators in several jurisdictions require pricing decisions to be explainable per applicant; LightGBM with SHAP gives this for free.

## Project next steps

1. **Joint multi-target modeling.** `LC`, `HALC`, and `CS` are mathematically linked. A multi-task model that learns them jointly should be more efficient than three separate fits.
2. **Cohort-specific models for long-tenure renewals.** Stability analysis flagged a real generalisation gap on the longest-tenure cohort.
3. **Monotone constraints.** Regulators often require monotone constraints on certain features (premium should not decrease in risk). Both LightGBM and XGBoost support this natively.
4. **Calibration refresh schedule.** Isotonic curves drift as the portfolio mix changes even when the underlying model is unchanged; production deployment needs periodic re-calibration.

## Tech stack

- Python 3.10+
- pandas, numpy, scipy
- scikit-learn (GLM, logistic, calibration, stacking, cross-validation)
- XGBoost, LightGBM (Tweedie regression and binary classification)
- SHAP (model interpretation)
- imbalanced-learn (SMOTE comparison)
- matplotlib, seaborn (visualisation)

## How to run

```bash
# Clone
git clone https://github.com/oliviacai210/insurance-risk-modeling.git
cd insurance-risk-modeling

# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Open the notebook
jupyter notebook notebooks/insurance_risk_modeling.ipynb
```

The notebook is end-to-end runnable from the `data/` CSVs that ship with the repo. The full notebook takes a few minutes on a standard laptop with `n_jobs=-1`. Search budgets in the notebook are intentionally moderate (`n_iter=15` for XGBoost, `n_iter=6` with `cv=3` for LightGBM); the original course version used `n_iter=40`. The difference in best CV-MSE is well under 1% in side experiments.

## Repository layout

```
insurance-risk-modeling/
  README.md
  requirements.txt
  LICENSE
  .gitignore
  notebooks/
    insurance_risk_modeling.ipynb      # End-to-end combined regression + classification
  data/
    train_clean_final_fixed.csv        # 39,928 policies, 32 columns
    test_clean_final_fixed.csv         # 13,310 policies, 25 columns (held out)
  reports/
    final_report.docx                  # Original group write-up
    poster.pptx                        # Original group poster
  predictions/
    LC_HALC_predictions.csv            # Generated by the notebook
    CS_predictions.csv                 # Generated by the notebook
```

## License

MIT. See [LICENSE](LICENSE).
