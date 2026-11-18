# External Validation of a Predictive Model for Functional Limitations in Older Adults Living with Diabetes

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/badge/managed%20with%20uv-7c3aed.svg)](https://docs.astral.sh/uv/)
[![License: TBA](https://img.shields.io/badge/license-to%20be%20added-lightgrey.svg)](#citation)

---

## Summary

Functional limitation is common among older adults with diabetes, yet prediction models
often remain confined to development datasets. This repository contains the full analysis
pipeline of a study that developed a parsimonious machine-learning model on the **2019
Brazilian National Health Survey (PNS 2019)** — adults aged 60 years or older with
self-reported diabetes, eight harmonized predictors — and evaluated whether it retained
predictive information in two independent datasets: the **PNS 2013** (same self-reported
functional-limitation outcome, historical benchmark under equivalent outcome
ascertainment) and a clinically distinct cohort from **Hospital Pedro Ernesto**
(outcome: Short Physical Performance Battery (SPPB) < 10).

Candidate algorithms were tuned and compared by nested five-fold cross-validation within
the training subset. **CatBoost**, which had the highest outer-fold mean area under the
receiver operating characteristic curve (AUROC), was selected before evaluation in either
target dataset and refitted on all eligible PNS 2019 participants. Held-out AUROC values
ranged from 0.654 to 0.670, with CatBoost numerically highest (0.670). In PNS 2013,
CatBoost achieved an AUROC of 0.660 and a Brier score of 0.214 (n = 1,568). In Pedro
Ernesto, the AUROC was 0.703 and the unrecalibrated Brier score was 0.307 (n = 122).
Repeated cross-validated recalibration reduced systematic calibration error, whereas
exploratory decision-curve analysis did not demonstrate clear incremental net benefit
over an assess-all/treat-all equivalent strategy.

The selected model retained modest discriminatory information across an independent
historical survey benchmark and a clinically distinct cohort, but calibration and
demonstrated clinical utility transported less well. The implementation is a research
prototype for prospective evaluation, not a deployable clinical decision-support system.

---

## Repository contents

| File | Purpose |
|---|---|
| `notebook.py` | Complete analysis pipeline: cohort processing, model selection, final-model refit, and the two-stage external validation (all tables and figures of the paper) |
| `requirements.txt` | Pinned Python dependencies (versions match the environment in which the study was reproduced) |

---

## What `notebook.py` does

`notebook.py` is a self-contained script organised in nine sections:

1. **Libraries and plotting style** — imports and figure constants (300 dpi).
2. **Evaluation and calibration methods** — metrics (AUROC, Brier, scaled Brier,
   calibration-in-the-large, calibration slope, O/E ratio, sensitivity/specificity/PPV/NPV
   at three thresholds), percentile bootstrap CIs (2,000 resamples), ROC / calibration /
   decision-curve plots with bootstrap bands, and forest plots.
3. **Bootstrap helpers** — single-model metric CIs and paired AUROC differences.
4. **`ModelFactory`** — model selection on PNS 2019: a single stratified 80/20
   train/test split (seed 42) shared by all scenarios; nested cross-validation on the
   training subset (outer 5-fold × inner 3-fold, seed 108) for every combination of
   scenario × algorithm; hyperparameter grids for **logistic regression, random forest,
   XGBoost, LightGBM and CatBoost**; factorial scenarios with and without SMOTE-NC
   rebalancing (applied inside the CV folds) and with reduced predictor subsets.
   Results are ranked by outer-fold mean AUROC and written to
   `model_factory_v6_results.md`.
5. **Data-processing helpers** — binary encoding of categoricals, per-capita income
   banded in minimum wages, preprocessor factory.
6. **Dataset import and processing** — reads `pns_2019.csv`, `pns_2013.csv` and
   `hospital.csv` from the repository root and builds the three analysis cohorts:
   - **PNS 2019 (development):** age ≥ 60, diagnosed diabetes, free of foot ulcer or
     amputation; outcome = self-reported functional limitation (item `Q058`).
   - **PNS 2013 (external historical benchmark):** same eligibility and same
     self-reported outcome; variable harmonisation across survey editions.
   - **Hospital Pedro Ernesto (external clinical cohort):** outcome = SPPB < 10.
7. **Model selection, final refit and saving** — the winning scenario + algorithm
   (CatBoost, per the paper) is rebuilt, refitted on 100 % of eligible PNS 2019
   participants, and saved to `modelos_salvos/` (joblib pipeline, preprocessor, model,
   input-column order, native JSON when supported).
8. **Optional TabPFN benchmark** — skipped unless `TABPFN_TOKEN` is set and
   `tabpfn-client` is installed.
9. **External validation (two stages)** — via `run_evaluation()`:
   - **Stage 1:** the frozen model on each external cohort — discrimination, overall
     performance, calibration, threshold-based metrics, decision-curve analysis, 95 %
     CIs from 2,000 bootstrap resamples → `output/stage1_original_model/`;
   - **Stage 2:** intercept-only and Platt recalibration estimated out-of-fold with
     repeated cross-validation, reported as paired differences versus the original
     model → `output/stage2_calibration/`.

Run it from the repository root so that the relative data paths resolve.

---

## Requirements

**Python 3.12** and **[uv](https://docs.astral.sh/uv/)**. Install with:

```bash
uv venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Pinned versions (the environment in which the study was reproduced):

| Package | Version |
|---|---|
| pandas | 2.3.3 |
| numpy | 2.0.2 |
| polars | 1.34.0 |
| joblib | 1.5.3 |
| scikit-learn | 1.6.1 |
| imbalanced-learn | 0.14.1 |
| xgboost | 3.1.3 |
| lightgbm | 4.6.0 |
| catboost | 1.2.8 |
| statsmodels | 0.14.6 |
| scipy | 1.17.1 |
| scikits.bootstrap | 1.1.1 |
| dcurves | 1.1.7 |
| betacal | 1.1.0 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
| tabpfn-client | 0.3.2 *(optional — TabPFN benchmark only)* |

---

## Usage

1. Obtain the PNS 2013 and PNS 2019 microdata (see below) and place them next to
   `notebook.py` as `pns_2013.csv` and `pns_2019.csv`.
2. *(Optional)* Ask the authors for the Hospital Pedro Ernesto file (`hospital.csv`).
   Without it, only the PNS-based external validation will run.
3. Install the environment (see [Requirements](#requirements)).
4. Run the pipeline from the repository root:

   ```bash
   uv run notebook.py
   ```

5. Outputs:
   - `model_factory_v6_results.md` — nested-CV ranking and scenario × algorithm summary;
   - `modelos_salvos/` — final pipeline, preprocessor and model artifacts;
   - `output/stage1_original_model/` — metrics CSVs, ROC, calibration and DCA figures
     with bootstrap bands, forest plots;
   - `output/stage2_calibration/` — recalibration metrics, paired differences versus
     the original model, final calibrator parameters.

---

## Data availability

**No dataset is redistributed by this repository.**

### PNS 2013 and PNS 2019 (public)

The Pesquisa Nacional de Saúde is a national household health survey conducted by the
**Brazilian Institute of Geography and Statistics (IBGE)** together with the Brazilian
Ministry of Health and **Fiocruz (Oswaldo Cruz Foundation)**. Both editions are open data.
Get the microdata and the official variable dictionaries from:

| Source | Link |
|---|---|
| IBGE — PNS page (2013 & 2019 microdata, documentation) | <https://www.ibge.gov.br/estatisticas/sociais/saude/9160-pesquisa-nacional-de-saude.html> |
| IBGE — FTP mirror | <http://ftp.ibge.gov.br/PNS/> |
| Fiocruz — PNS portal (databases, dictionaries, questionnaires) | <https://www.pns.icict.fiocruz.br/bases-de-dados> |

Suggested citation of the source:

> IBGE. Pesquisa Nacional de Saúde 2019: informações sobre os domicílios, acesso e
> utilização dos serviços de saúde. Rio de Janeiro: IBGE; 2020.
> IBGE. Pesquisa Nacional de Saúde 2013. Rio de Janeiro: IBGE; 2014.

All processing steps applied on top of the raw microdata (cohort selection, variable
harmonisation, and binarisation of the outcome) are implemented in `notebook.py` and are
fully reproducible once the microdata files are placed next to it.

### Hospital Pedro Ernesto cohort — available on request

The clinical cohort is **not distributed** by this repository. These are individual-level
clinical data collected under an approved research protocol at Hospital Pedro Ernesto
(Universidade do Estado do Rio de Janeiro). They are **available from the corresponding
author upon reasonable request**, subject to ethics-committee approval and a data-use
agreement. No derived or aggregated version of this cohort is published here.

---


