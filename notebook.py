# -*- coding: utf-8 -*-
"""
Sections
--------
1. Libraries and plotting style
2. Evaluation and calibration methods (two-stage external validation)
3. Bootstrap metrics (percentile CIs for a single model)
4. ModelFactory (nested CV + factorial comparison)
5. Data-processing helpers
6. Dataset import and processing (raw microdata)
7. Model selection, final model refit and saving
8. Optional TabPFN benchmark
9. External validation (two stages)

Python 3.12.
"""

import os
import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import joblib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import norm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

try:
    from tabpfn_client import TabPFNClassifier, set_access_token
    _HAS_TABPFN = True
except ImportError:  
    TabPFNClassifier = None
    set_access_token = None
    _HAS_TABPFN = False


# ────────────────────────────────────────────────────────────────────────────
# Plotting style 
# ────────────────────────────────────────────────────────────────────────────
DPI = 300
FIG_SIZE = (6, 5)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})

# ────────────────────────────────────────────────────────────────────────────
# Evaluation constants 
# ────────────────────────────────────────────────────────────────────────────
EPS = 1e-6  # clipping value that keeps logit() finite
FMT = lambda v: f'{v:.4f}'
CSV_FMT = '%.4f'

ORIGINAL = 'Original model'
METHODS = {
    'intercept': 'Intercept-only',
    'platt':     'Platt (intercept + slope)',
}
COLORS = {
    ORIGINAL:                    'firebrick',
    'Intercept-only':            'purple',
    'Platt (intercept + slope)': 'dodgerblue',
}
# Internal metric keys: auroc (area under the ROC curve), brier (Brier score),
# scaled_brier (1 - Brier / null Brier), citl (calibration-in-the-large),
# slope (calibration slope), oe_ratio (observed-to-expected ratio).
BASIC_METRICS = ['auroc', 'brier', 'scaled_brier', 'citl', 'slope', 'oe_ratio']
# ppv = positive predictive value; npv = negative predictive value
THRESHOLD_METRICS = ['sensitivity', 'specificity', 'ppv', 'npv', 'accuracy']
LABELS = {
    'auroc': 'AUROC', 'brier': 'Brier score', 'scaled_brier': 'Scaled Brier score',
    'citl': 'Calibration intercept (CITL)', 'slope': 'Calibration slope',
    'oe_ratio': 'O/E ratio', 'sensitivity': 'Sensitivity',
    'specificity': 'Specificity', 'ppv': 'PPV', 'npv': 'NPV', 'accuracy': 'Accuracy',
}
# Reference lines in the forest plots (value expected for a useless / perfectly calibrated model)
REFERENCE_LINES = {'auroc': 0.5, 'scaled_brier': 0.0, 'citl': 0.0, 'slope': 1.0, 'oe_ratio': 1.0}
THRESHOLD_LABELS = {'default': 'Default threshold', 'youden': 'Youden threshold',
                    'sens80': 'Sensitivity ≥ 80% threshold'}


def _slug(text):
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')


def _save_figure(fig, path, show):
    fig.tight_layout()
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────
def _glm_intercept(y, lp):
    """Binomial GLM (generalized linear model) with only an intercept and the logit of the
    predictions (lp = linear predictor) as offset (statsmodels)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()


def _estimate_intercept(y, lp):
    """Intercept with the slope fixed at 1 (CITL, calibration-in-the-large), via statsmodels."""
    return float(_glm_intercept(y, lp).params[0])


def _calibration(y, p):
    """Returns (CITL, calibration slope) of predicted probabilities p for the binary outcome y."""
    lp = logit(np.clip(p, EPS, 1 - EPS))
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y).coef_[0, 0]
    return _estimate_intercept(y, lp), slope


def _metrics(y, p, thresholds):
    """Discrimination, calibration and threshold-based metrics for one set of predictions."""
    prev = y.mean()
    brier = np.mean((p - y) ** 2)
    citl, slope = _calibration(y, p)
    out = {
        'auroc': roc_auc_score(y, p),
        'brier': brier,
        'scaled_brier': 1 - brier / (prev * (1 - prev)),  # 1 - Brier / null Brier score
        'citl': citl,
        'slope': slope,
        'oe_ratio': prev / np.mean(p),  # observed-to-expected ratio
    }
    div = lambda a, b: a / b if b > 0 else np.nan
    pos = y == 1
    for name, t in thresholds.items():
        pred = p >= t
        tp, fp = np.sum(pred & pos), np.sum(pred & ~pos)    # true / false positives
        fn, tn = np.sum(~pred & pos), np.sum(~pred & ~pos)  # false / true negatives
        out[f'sensitivity@{name}'] = div(tp, tp + fn)
        out[f'specificity@{name}'] = div(tn, tn + fp)
        out[f'ppv@{name}'] = div(tp, tp + fp)  # positive predictive value
        out[f'npv@{name}'] = div(tn, tn + fn)  # negative predictive value
        out[f'accuracy@{name}'] = (tp + tn) / len(y)
    return out


def _optimal_thresholds(y, p, target_sens=0.8):
    """Returns (Youden threshold, Sens80 threshold = first threshold with sensitivity >= target_sens)."""
    fpr, tpr, thr = roc_curve(y, p)  # false / true positive rates
    return thr[np.argmax(tpr - fpr)], thr[np.argmax(tpr >= target_sens)]


def _mean_metrics(y, P, thresholds):
    """Mean of the metrics over the repetitions (rows of P)."""
    return pd.DataFrame([_metrics(y, P[r], thresholds) for r in range(P.shape[0])]).mean()


def _bootstrap_metrics(y, P_dict, thresholds, n_boot, alpha, seed, ref=None, rederive=False):
    """
    Percentile CI (confidence interval) for the metrics of each version in P_dict
    ({name: matrix (repetitions x n) of probabilities}). At every bootstrap resample of
    patients, one cross-validation repetition is drawn (if there is more than one).
    If `ref` is given, the PAIRED difference (version - ref) is also computed.
    If `rederive`, the Youden and Sens80 thresholds are re-estimated in every resample.
    """
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)
    names = list(P_dict)
    name0 = names[0]
    R = max(P.shape[0] for P in P_dict.values())

    point = {name: _mean_metrics(y, P, thresholds) for name, P in P_dict.items()}
    if rederive:
        point[name0]['youden_threshold_resampled'] = thresholds['youden']
        point[name0]['sens80_threshold_resampled'] = thresholds['sens80']

    boots = {name: [] for name in names}
    deltas = {name: [] for name in names if ref and name != ref}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        r = rng.integers(0, R)
        mets = {}
        for name, P in P_dict.items():
            pb = P[r % P.shape[0]][idx]
            m = _metrics(yb, pb, thresholds)
            if rederive and name == name0:
                m['youden_threshold_resampled'], m['sens80_threshold_resampled'] = _optimal_thresholds(yb, pb)
            mets[name] = m
            boots[name].append(m)
        for name in deltas:
            deltas[name].append({k: mets[name][k] - mets[ref][k] for k in BASIC_METRICS + [
                c for c in mets[name] if '@' in c]})

    def table(est, items):
        df = pd.DataFrame(items).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({'estimate': est, 'ci_low': df.quantile(alpha / 2),
                             'ci_high': df.quantile(1 - alpha / 2)})  # ci = confidence interval

    tables = {name: table(point[name], boots[name]) for name in names}
    delta_tables = {name: table((point[name] - point[ref]).dropna(), deltas[name]) for name in deltas}
    return tables, delta_tables


def _readable_table(tables, metrics, decimals=4):
    """DataFrame of strings 'estimate (CI low; CI high)' for printing."""
    rows = {}
    for name, t in tables.items():
        rows[name] = {
            m: f"{t.loc[m,'estimate']:.{decimals}f} ({t.loc[m,'ci_low']:.{decimals}f}; {t.loc[m,'ci_high']:.{decimals}f})"
            for m in metrics if m in t.index}
    return pd.DataFrame(rows).T


def _long_table(tables, dataset_name, extra_stage=None):
    out = []
    for name, t in tables.items():
        d = t.reset_index().rename(columns={'index': 'metric'})
        d.insert(0, 'version', name)
        d.insert(0, 'dataset', dataset_name)
        out.append(d)
    return pd.concat(out, ignore_index=True)


# ────────────────────────────────────────────────────────────────────────────
# Calibrators (fitted on the model's predicted probability)
# ────────────────────────────────────────────────────────────────────────────
def _fit_calibrator(method, p, y):
    """Fits the calibrator and returns."""
    p = np.clip(p, EPS, 1 - EPS)
    if method == 'intercept':   # statsmodels (GLM, generalized linear model, with offset)
        a = _estimate_intercept(y, logit(p))
        return (lambda q: expit(a + logit(np.clip(q, EPS, 1 - EPS)))), {'intercept': a}
    if method == 'platt':       # scikit-learn (logistic regression on the logit)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
        f = lambda q: lr.predict_proba(logit(np.clip(q, EPS, 1 - EPS)).reshape(-1, 1))[:, 1]
        return f, {'intercept': lr.intercept_[0], 'slope': lr.coef_[0, 0]}
    raise ValueError(f"Unknown calibration method: {method}")


def _oof_recalibrated(y, p, method, n_splits, n_repeats, seed):
    """
    OOF (out-of-fold) recalibrated probabilities (matrix n_repeats x n).
    In each repetition, every patient is recalibrated by a calibrator fitted
    without that patient (stratified cross-validation).
    """
    n = len(y)
    P = np.zeros((n_repeats, n))
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for tr, te in skf.split(p.reshape(-1, 1), y):
            calibrator, _ = _fit_calibrator(method, p[tr], y[tr])
            P[r, te] = calibrator(p[te])
    return P


def _wald_ci(y, p, method, alpha):
    """Wald CI (confidence interval, from statsmodels) of the calibrator parameters fitted on the whole dataset."""
    lp = logit(np.clip(p, EPS, 1 - EPS))
    try:
        if method == 'intercept':
            ci = _glm_intercept(y, lp).conf_int(alpha=alpha)
            return {'intercept': (ci[0, 0], ci[0, 1])}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        ci = fit.conf_int(alpha=alpha)
        return {'intercept': (ci[0, 0], ci[0, 1]), 'slope': (ci[1, 0], ci[1, 1])}
    except Exception:
        return {}


def _final_parameters(y, p, method, n_boot, alpha, seed):
    """
    Final calibrator (whole dataset): estimate, percentile bootstrap CI
    (ci_low/ci_high, CI = confidence interval) and statsmodels Wald CI (wald_low/wald_high).
    """
    _, par = _fit_calibrator(method, p, y)
    rng = np.random.default_rng(seed)
    n, boots = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(_fit_calibrator(method, p[idx], y[idx])[1])
    df = pd.DataFrame(boots)
    tab = pd.DataFrame({'estimate': pd.Series(par), 'ci_low': df.quantile(alpha / 2),
                        'ci_high': df.quantile(1 - alpha / 2)})
    w = _wald_ci(y, p, method, alpha)
    tab['wald_low'] = pd.Series({k: v[0] for k, v in w.items()}, dtype=float)
    tab['wald_high'] = pd.Series({k: v[1] for k, v in w.items()}, dtype=float)
    return tab


# ────────────────────────────────────────────────────────────────────────────
# Curves with CIs (confidence intervals): ROC, calibration, DCA
# ROC = receiver operating characteristic; DCA = decision curve analysis
# ────────────────────────────────────────────────────────────────────────────
def _roc_bootstrap(y, p, n_boot, alpha, seed):
    grid = np.linspace(0, 1, 101)
    fpr, tpr, _ = roc_curve(y, p)  # false / true positive rates
    point = np.interp(grid, fpr, tpr)
    rng, n, bs = np.random.default_rng(seed), len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        f, t, _ = roc_curve(y[idx], p[idx])
        bs.append(np.interp(grid, f, t))
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2], axis=0)
    return grid, point, lo, hi


def _plot_roc(grid, point, lo, hi, auc_text, title, color, path, show):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(grid, point, color=color, lw=1.8, label=auc_text)
    ax.fill_between(grid, lo, hi, color=color, alpha=0.2, label='Bootstrap 95% band')
    ax.plot([0, 1], [0, 1], color='black', ls='--', lw=1, label='Chance')
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title(title)
    ax.legend(loc='lower right')
    _save_figure(fig, path, show)


def _wilson(k, m, z):
    """Wilson score interval for a proportion k/m (z = normal quantile)."""
    ph = k / m
    den = 1 + z ** 2 / m
    c = (ph + z ** 2 / (2 * m)) / den
    h = z * np.sqrt(ph * (1 - ph) / m + z ** 2 / (4 * m ** 2)) / den
    return c - h, c + h


def _plot_calibration(y, p, title, color, path, n_bins, alpha, show, annotation=None):
    z = norm.ppf(1 - alpha / 2)
    xs, ys, err_low, err_high = [], [], [], []
    for g in np.array_split(np.argsort(p), n_bins):
        m, k = len(g), y[g].sum()
        lo, hi = _wilson(k, m, z)
        xs.append(p[g].mean())
        ys.append(k / m)
        err_low.append(k / m - lo)
        err_high.append(hi - k / m)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], color='black', ls='--', lw=1, label='Perfect calibration')
    ax.errorbar(xs, ys, yerr=[err_low, err_high], fmt='o-', color=color, capsize=3, ms=5, lw=1.3,
                label=f'Model ({n_bins} quantile bins)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel(f'Observed proportion ({int((1 - alpha) * 100)}% Wilson CI)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    if annotation:
        ax.text(0.97, 0.03, annotation, transform=ax.transAxes, ha='right', va='bottom', fontsize=7,
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))
    _save_figure(fig, path, show)


def _dca_bootstrap(y, P, th, n_boot, alpha, seed):
    """Net benefit (NB) of the model (P: repetitions x n) and of 'treat all', with bootstrap CI
    (confidence interval)."""
    w = th / (1 - th)  # odds of the threshold probability

    def nb_model(yy, pp):
        pred = pp[:, None] >= th[None, :]
        pos = (yy == 1)[:, None]
        return (pred & pos).sum(0) / len(yy) - (pred & ~pos).sum(0) / len(yy) * w

    def nb_treat_all(yy):
        prev = yy.mean()
        return prev - (1 - prev) * w

    R, n = P.shape[0], len(y)
    point_model = np.mean([nb_model(y, P[r]) for r in range(R)], axis=0)
    point_all = nb_treat_all(y)
    rng, b_model, b_all = np.random.default_rng(seed), [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = rng.integers(0, R)
        b_model.append(nb_model(y[idx], P[r][idx]))
        b_all.append(nb_treat_all(y[idx]))
    q = [alpha / 2, 1 - alpha / 2]
    lo_m, hi_m = np.quantile(b_model, q, axis=0)
    lo_t, hi_t = np.quantile(b_all, q, axis=0)
    return pd.DataFrame({'threshold': th, 'nb_model': point_model, 'nb_model_ci_low': lo_m,
                         'nb_model_ci_high': hi_m, 'nb_treat_all': point_all,
                         'nb_treat_all_ci_low': lo_t, 'nb_treat_all_ci_high': hi_t})


def _plot_dca(d, title, color, path, show):
    """Decision curve analysis (DCA) plot: net benefit (NB) against threshold probability."""
    th = d['threshold'].values
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(th, d['nb_model'], color=color, lw=2, label='Model')
    ax.fill_between(th, d['nb_model_ci_low'], d['nb_model_ci_high'], color=color, alpha=0.2)
    ax.plot(th, d['nb_treat_all'], color='teal', lw=1.5, ls='--', label='Treat all')
    ax.fill_between(th, d['nb_treat_all_ci_low'], d['nb_treat_all_ci_high'], color='teal', alpha=0.15)
    ax.axhline(0, color='black', lw=0.9, ls=':', label='Treat none')
    ax.set_ylim(-0.05, max(d['nb_treat_all'].max(), d['nb_model'].max()) * 1.1)
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title(title)
    ax.legend(loc='upper right')
    _save_figure(fig, path, show)


# ────────────────────────────────────────────────────────────────────────────
# "Forest" plots (estimate + CI, confidence interval) 
# ────────────────────────────────────────────────────────────────────────────
def _forest_ax(ax, rows, ref=None, colors=None):
    """rows: list of dicts with group, series, est (estimate), lo (CI lower limit), hi (CI upper limit)."""
    groups = list(dict.fromkeys(r['group'] for r in rows))
    series = list(dict.fromkeys(r['series'] for r in rows))
    ns = len(series)
    step = 0.7 / ns
    palette = plt.get_cmap('tab10')
    for r in rows:
        if np.isnan(r['est']):
            continue
        j = series.index(r['series'])
        y = groups.index(r['group']) + (j - (ns - 1) / 2) * step
        color = palette(j) if ns > 1 else (colors or {}).get(r['group'], 'black')
        xerr = None
        if not (np.isnan(r['lo']) or np.isnan(r['hi'])):
            xerr = [[r['est'] - r['lo']], [r['hi'] - r['est']]]
        ax.errorbar(r['est'], y, xerr=xerr, fmt='o', color=color, ecolor=color, capsize=3, ms=5, lw=1.3)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_ylim(len(groups) - 0.5, -0.5)
    if ref is not None:
        ax.axvline(ref, color='gray', ls=':', lw=1)
    if ns > 1:
        ax.legend(handles=[Line2D([0], [0], marker='o', color=palette(j), ls='', label=s)
                           for j, s in enumerate(series)], loc='best')


def _forest_plot(rows, xlabel, title, ref, path, colors, show):
    ng = len(dict.fromkeys(r['group'] for r in rows))
    ns = len(dict.fromkeys(r['series'] for r in rows))
    fig, ax = plt.subplots(figsize=(6, 1.3 + 0.5 * ng * (1 + 0.35 * (ns - 1))))
    _forest_ax(ax, rows, ref, colors)
    ax.set_xlabel(f'{xlabel} (point estimate and 95% CI)')
    ax.set_title(title)
    _save_figure(fig, path, show)


def _row(group, series, t, metric):
    return {'group': group, 'series': series, 'est': t.loc[metric, 'estimate'],
            'lo': t.loc[metric, 'ci_low'], 'hi': t.loc[metric, 'ci_high']}


# ────────────────────────────────────────────────────────────────────────────
# Stage 1: original model
# ────────────────────────────────────────────────────────────────────────────
def evaluate_original_model(
    model, datasets, output_dir='output', n_boot=2000, alpha=0.05, seed=108,
    default_threshold=0.5, youden_threshold=None, sens80_threshold=None,
    dca_thresholds=None, n_bins=5, show=False,
):
    """
    model : trained estimator (with predict_proba)
    datasets : dict {dataset_name: (X, y)} already preprocessed as in model development
    youden_threshold / sens80_threshold : thresholds defined IN THE DEVELOPMENT data
        (recommended). Youden = threshold maximizing sensitivity + specificity - 1;
        Sens80 = threshold reaching sensitivity >= 80%. If None, they are estimated on each
        external dataset (optimistic) and the CI (confidence interval) of the thresholds
        themselves is reported.
    dca_thresholds : probability thresholds for the DCA (decision curve analysis).
    """
    out_dir = os.path.join(output_dir, 'stage1_original_model')
    os.makedirs(out_dir, exist_ok=True)
    th_dca = np.arange(0.05, 0.96, 0.01) if dca_thresholds is None else np.asarray(dca_thresholds)

    tables, thresholds_by_dataset, probs, long_tables = {}, {}, {}, []
    for k, (dataset_name, (X, y)) in enumerate(datasets.items()):
        y = np.asarray(y).astype(int)
        p = model.predict_proba(X)[:, 1]
        probs[dataset_name] = p
        color = plt.get_cmap('tab10')(k)

        t_youden, t_sens80 = _optimal_thresholds(y, p)
        rederive = youden_threshold is None and sens80_threshold is None
        thr = {'default': default_threshold,
               'youden': t_youden if youden_threshold is None else youden_threshold,
               'sens80': t_sens80 if sens80_threshold is None else sens80_threshold}
        thresholds_by_dataset[dataset_name] = thr

        print(f"\n[{dataset_name}] n={len(y)}, prevalence={y.mean():.4f}, thresholds={ {a: round(b, 4) for a, b in thr.items()} }"
              + ("  (Youden/Sens80 estimated on this dataset: optimistic)" if rederive else ""))
        t = _bootstrap_metrics(y, {ORIGINAL: p[None, :]}, thr, n_boot, alpha, seed,
                               rederive=rederive)[0][ORIGINAL]
        tables[dataset_name] = t
        t.to_csv(os.path.join(out_dir, f'{dataset_name}_metrics.csv'), float_format=CSV_FMT)
        long_tables.append(_long_table({ORIGINAL: t}, dataset_name))
        print(t.to_string(float_format=FMT))

        # per-dataset curves
        grid, point, lo, hi = _roc_bootstrap(y, p, n_boot, alpha, seed)
        auc = t.loc['auroc']
        _plot_roc(grid, point, lo, hi,
                  f"AUROC = {auc['estimate']:.4f} ({auc['ci_low']:.4f}–{auc['ci_high']:.4f})",
                  f'{dataset_name}: ROC curve (original model)', color,
                  os.path.join(out_dir, f'{dataset_name}_roc.png'), show)
        annotation = (f"CITL {t.loc['citl','estimate']:.4f} ({t.loc['citl','ci_low']:.4f}; {t.loc['citl','ci_high']:.4f})\n"
                      f"Slope {t.loc['slope','estimate']:.4f} ({t.loc['slope','ci_low']:.4f}; {t.loc['slope','ci_high']:.4f})")
        _plot_calibration(y, p, f'{dataset_name}: calibration curve (original model)', color,
                          os.path.join(out_dir, f'{dataset_name}_calibration.png'),
                          n_bins, alpha, show, annotation)
        d = _dca_bootstrap(y, p[None, :], th_dca, n_boot, alpha, seed)
        d.to_csv(os.path.join(out_dir, f'{dataset_name}_dca.csv'), index=False, float_format=CSV_FMT)
        _plot_dca(d, f'{dataset_name}: decision curve (original model)', color,
                  os.path.join(out_dir, f'{dataset_name}_dca.png'), show)

    # one plot per metric, comparing the datasets
    dataset_colors = {b: plt.get_cmap('tab10')(i) for i, b in enumerate(datasets)}
    for m in BASIC_METRICS:
        rows = [_row(b, ORIGINAL, tables[b], m) for b in datasets]
        _forest_plot(rows, LABELS[m], f'{LABELS[m]}: original model by external dataset',
                     REFERENCE_LINES.get(m), os.path.join(out_dir, f'forest_{m}.png'), dataset_colors, show)
    for m in THRESHOLD_METRICS:
        rows = [_row(b, THRESHOLD_LABELS[th_name], tables[b], f'{m}@{th_name}')
                for b in datasets for th_name in ('default', 'youden', 'sens80')]
        _forest_plot(rows, LABELS[m], f'{LABELS[m]}: original model by external dataset',
                     None, os.path.join(out_dir, f'forest_{m}.png'), None, show)

    pd.concat(long_tables).to_csv(os.path.join(out_dir, 'stage1_all_metrics.csv'), index=False, float_format=CSV_FMT)
    return {'tables': tables, 'thresholds': thresholds_by_dataset, 'probs': probs}


# ────────────────────────────────────────────────────────────────────────────
# Stage 2: calibrators
# ────────────────────────────────────────────────────────────────────────────
def evaluate_calibrators(
    model, datasets, output_dir='output', methods=('intercept', 'platt'),
    n_splits=5, n_repeats=20, n_boot=2000, alpha=0.05, seed=108, threshold=0.5,
    dca_thresholds=None, n_bins=5, show=False,
):
    """
    For each dataset, evaluates each calibrator with repeated stratified cross-validation
    (n_splits folds x n_repeats repetitions). Metrics are reported separately per
    calibrator, with CIs (confidence intervals), plus the paired difference against the
    original model.

    threshold : fixed threshold (defined a priori) used for sensitivity, specificity, PPV
        (positive predictive value), NPV (negative predictive value) and accuracy, applied to
        the probabilities of each version.
    Returns, per dataset: metrics, differences, final calibrator parameters (fitted on the
    whole dataset, for intercept-only and Platt) and OOF (out-of-fold) probabilities.
    """
    out_dir = os.path.join(output_dir, 'stage2_calibration')
    os.makedirs(out_dir, exist_ok=True)
    th_dca = np.arange(0.05, 0.96, 0.01) if dca_thresholds is None else np.asarray(dca_thresholds)
    thr = {'default': threshold}

    results, long_tables, long_delta_tables = {}, [], []
    for dataset_name, (X, y) in datasets.items():
        y = np.asarray(y).astype(int)
        p = model.predict_proba(X)[:, 1]
        if min(np.bincount(y)) < n_splits:
            raise ValueError(f"{dataset_name}: the minority class has fewer cases than n_splits={n_splits}.")
        print(f"\n[{dataset_name}] n={len(y)}, prevalence={y.mean():.4f} — cross-validated recalibration "
              f"({n_splits} folds x {n_repeats} repeats)")

        P_dict = {ORIGINAL: p[None, :]}
        for m in methods:
            P_dict[METHODS[m]] = _oof_recalibrated(y, p, m, n_splits, n_repeats, seed)

        tables, deltas = _bootstrap_metrics(y, P_dict, thr, n_boot, alpha, seed, ref=ORIGINAL)
        metrics_to_show = BASIC_METRICS + [f'{m}@default' for m in THRESHOLD_METRICS]
        print("\nMetrics (cross-validated), estimate (95% CI):")
        print(_readable_table(tables, metrics_to_show).to_string())
        print("\nPaired difference vs original model, estimate (95% CI):")
        print(_readable_table(deltas, BASIC_METRICS).to_string())

        for name, t in tables.items():
            t.to_csv(os.path.join(out_dir, f'{dataset_name}_{_slug(name)}_metrics.csv'), float_format=CSV_FMT)
        for name, t in deltas.items():
            t.to_csv(os.path.join(out_dir, f'{dataset_name}_{_slug(name)}_delta_vs_original.csv'), float_format=CSV_FMT)
        long_tables.append(_long_table(tables, dataset_name))
        long_delta_tables.append(_long_table(deltas, dataset_name))

        # parameters of the final calibrator (whole dataset) with CIs (confidence intervals)
        finals = {}
        for m in methods:
            if m in ('intercept', 'platt'):
                finals[METHODS[m]] = _final_parameters(y, p, m, n_boot, alpha, seed)
        if finals:
            fp = pd.concat(finals, names=['calibrator', 'parameter'])
            fp.to_csv(os.path.join(out_dir, f'{dataset_name}_final_calibrator_parameters.csv'), float_format=CSV_FMT)
            print("\nFinal calibrator parameters (fit on the whole dataset), estimate (95% CI):")
            print(fp.to_string(float_format=FMT))

        # one plot per metric, comparing the versions
        for m in BASIC_METRICS:
            rows = [_row(v, 'estimate', tables[v], m) for v in P_dict]
            _forest_plot(rows, LABELS[m], f'{dataset_name}: {LABELS[m]} by calibration method (cross-validated)',
                         REFERENCE_LINES.get(m), os.path.join(out_dir, f'{dataset_name}_forest_{m}.png'), COLORS, show)
        for m in THRESHOLD_METRICS:
            rows = [_row(v, 'estimate', tables[v], f'{m}@default') for v in P_dict]
            _forest_plot(rows, f'{LABELS[m]} at threshold {threshold:.4f}',
                         f'{dataset_name}: {LABELS[m]} by calibration method (cross-validated)',
                         None, os.path.join(out_dir, f'{dataset_name}_forest_{m}.png'), COLORS, show)

        # paired differences (one panel per metric)
        fig, axes = plt.subplots(2, 3, figsize=(12, 5))
        for ax, m in zip(axes.ravel(), BASIC_METRICS):
            rows = [{'group': v, 'series': 'estimate', 'est': deltas[v].loc[m, 'estimate'],
                     'lo': deltas[v].loc[m, 'ci_low'], 'hi': deltas[v].loc[m, 'ci_high']} for v in deltas]
            _forest_ax(ax, rows, ref=0, colors=COLORS)
            ax.set_title(LABELS[m])
            ax.set_xlabel('Change vs original model')
            if ax not in axes[:, 0]:
                ax.set_yticklabels([])
        fig.suptitle(f'{dataset_name}: paired change vs original model (cross-validated, 95% CI)')
        _save_figure(fig, os.path.join(out_dir, f'{dataset_name}_forest_delta_vs_original.png'), show)

        # calibration curve and DCA (decision curve analysis) per version
        for name, P in P_dict.items():
            color = COLORS[name]
            t = tables[name]
            annotation = (f"CITL {t.loc['citl','estimate']:.4f} ({t.loc['citl','ci_low']:.4f}; {t.loc['citl','ci_high']:.4f})\n"
                          f"Slope {t.loc['slope','estimate']:.4f} ({t.loc['slope','ci_low']:.4f}; {t.loc['slope','ci_high']:.4f})")
            p_plot = P.mean(axis=0)  # mean over the repetitions (for plotting only)
            _plot_calibration(y, p_plot, f'{dataset_name}: calibration curve – {name}', color,
                              os.path.join(out_dir, f'{dataset_name}_calibration_{_slug(name)}.png'),
                              n_bins, alpha, show, annotation)
            d = _dca_bootstrap(y, P, th_dca, n_boot, alpha, seed)
            d.to_csv(os.path.join(out_dir, f'{dataset_name}_dca_{_slug(name)}.csv'), index=False, float_format=CSV_FMT)
            _plot_dca(d, f'{dataset_name}: decision curve – {name}', color,
                      os.path.join(out_dir, f'{dataset_name}_dca_{_slug(name)}.png'), show)

        results[dataset_name] = {'metrics': tables, 'deltas': deltas,
                                 'final_calibrator_parameters': finals, 'oof_probs': P_dict}

    pd.concat(long_tables).to_csv(os.path.join(out_dir, 'stage2_all_metrics.csv'), index=False, float_format=CSV_FMT)
    pd.concat(long_delta_tables).to_csv(os.path.join(out_dir, 'stage2_all_deltas.csv'), index=False, float_format=CSV_FMT)
    return results


def run_evaluation(model, datasets, output_dir='output', **kwargs):
    """
    Runs stage 1 and stage 2. kwargs accepted by both: n_boot (number of bootstrap
    resamples), alpha (significance level), seed, dca_thresholds (decision curve analysis
    thresholds), n_bins, show. For stage-specific options, call the functions separately.
    """
    common = ('n_boot', 'alpha', 'seed', 'dca_thresholds', 'n_bins', 'show')
    stage1_only = ('default_threshold', 'youden_threshold', 'sens80_threshold')
    stage2_only = ('methods', 'n_splits', 'n_repeats', 'threshold')
    k1 = {k: v for k, v in kwargs.items() if k in common + stage1_only}
    k2 = {k: v for k, v in kwargs.items() if k in common + stage2_only}
    if 'default_threshold' in kwargs and 'threshold' not in k2:
        k2['threshold'] = kwargs['default_threshold']  # same threshold in both stages
    e1 = evaluate_original_model(model, datasets, output_dir, **k1)
    e2 = evaluate_calibrators(model, datasets, output_dir, **k2)
    return {'stage1': e1, 'stage2': e2}



def _calibration_fast(y, p):
    """Returns (CITL with slope fixed at 1, calibration slope) via a fast NLL optimisation.

    Used only inside the bootstrap loop of `bootstrap_metrics` (the two-stage
    evaluation module above uses the statsmodels GLM instead).
    """
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))

    # slope: logistic regression of y on logit(p) (unpenalised)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y).coef_[0, 0]

    # calibration-in-the-large: intercept with lp as offset (slope = 1)
    def nll(a):
        q = np.clip(expit(a + lp), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    citl = minimize_scalar(nll, bounds=(-5, 5), method="bounded").x
    return citl, slope


def _single_model_metrics(y, p, threshold):
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    prev = y.mean()
    brier = np.mean((p - y) ** 2)
    brier_ref = prev * (1 - prev)  # model that always predicts the prevalence

    pred = (p >= threshold).astype(int)
    tp = np.sum((pred == 1) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    div = lambda a, b: a / b if b > 0 else np.nan

    sens = div(tp, tp + fn)
    ppv = div(tp, tp + fp)
    citl, slope = _calibration_fast(y, p)

    return {
        "auroc": roc_auc_score(y, p),
        "brier": brier,
        "scaled_brier": 1 - brier / brier_ref,
        "citl": citl,
        "slope": slope,
        "sensitivity": sens,
        "specificity": div(tn, tn + fp),
        "ppv": ppv,
        "npv": div(tn, tn + fn),
        "accuracy": (tp + tn) / len(y),
        "f1": div(2 * ppv * sens, ppv + sens) if not (np.isnan(ppv) or np.isnan(sens)) else np.nan,
    }


def bootstrap_metrics(y_true, y_prob, threshold=0.5, n_boot=2000, alpha=0.05, seed=108):
    """DataFrame with the point estimate and percentile CI of each metric."""
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)

    point = _single_model_metrics(y, p, threshold)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:  # resample without both classes
            continue
        samples.append(_single_model_metrics(y[idx], p[idx], threshold))

    boot = pd.DataFrame(samples)
    table = pd.DataFrame({
        "estimate": pd.Series(point),
        "ci_low": boot.quantile(alpha / 2),
        "ci_high": boot.quantile(1 - alpha / 2),
    })
    table.attrs["n_boot_valid"] = len(boot)
    return table


def bootstrap_auc_difference(y_true, p_a, p_b, n_boot=2000, alpha=0.05, seed=108):
    """CI of the AUROC difference (A - B) with a paired bootstrap (same rows)."""
    y = np.asarray(y_true)
    p_a, p_b = np.asarray(p_a, float), np.asarray(p_b, float)
    n = len(y)
    rng = np.random.default_rng(seed)

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))

    diffs = np.array(diffs)
    return {
        "dif_auc": roc_auc_score(y, p_a) - roc_auc_score(y, p_b),
        "ci_low": np.quantile(diffs, alpha / 2),
        "ci_high": np.quantile(diffs, 1 - alpha / 2),
    }


#
REBALANCING = ("with_smotenc", "without_smotenc")
ALGORITHMS = ("logistic_regression", "random_forest", "xgboost",
              "lightgbm", "catboost")

RESULTS_FILE_DEFAULT = "model_factory_v6_results.md"
PROJECT_ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 108


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Scaling + imputation (mean/most_frequent) + categorical casting.

    Reproduces the order of the v5 `_preprocess()` (scale → impute → cast),
    but fitted PER FOLD when used as a pipeline step — no leakage of the
    validation statistics into training.

    Returns a DataFrame with the SAME column names/order as the input
    (required for SMOTENC(categorical_features='auto') and for
    CatBoostClassifier(cat_features=[...])).
    """
    def __init__(self, numerical, categorical):
        self.numerical = numerical
        self.categorical = categorical

    def fit(self, X, y=None):
        X = self._as_dataframe(X)
        num, cat = self.numerical, self.categorical

        self.scaler_ = StandardScaler()
        X[num] = self.scaler_.fit_transform(X[num])

        self.mean_imputer_ = SimpleImputer(strategy='mean')
        X[num] = self.mean_imputer_.fit_transform(X[num])

        self.mode_imputer_ = SimpleImputer(strategy='most_frequent')
        X[cat] = self.mode_imputer_.fit_transform(X[cat])
        return self

    def transform(self, X):
        X = self._as_dataframe(X)
        num, cat = self.numerical, self.categorical

        X[num] = self.scaler_.transform(X[num])
        X[num] = self.mean_imputer_.transform(X[num])
        X[cat] = self.mode_imputer_.transform(X[cat])

        X[cat] = X[cat].astype('int').astype('category')
        return X

    @staticmethod
    def _as_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)


def _detect_subsets(categorical, numerical):
    """Detects 'sexo', 'idade*' and 'duracao*' in the given column lists."""
    sex = next((c for c in categorical if "sexo" in c.lower()), None)
    age = next((n for n in numerical if "idade" in n.lower()), None)
    duration = next((n for n in numerical
                     if "duracao" in n.lower() or "duração" in n.lower()), None)

    missing = [name for name, col in
               {"sex": sex, "age": age, "duration": duration}.items()
               if col is None]
    if missing:
        raise ValueError(
            f"Could not automatically detect the columns {missing}. "
            f"categorical={list(categorical)} | numerical={list(numerical)}. "
            f"Pass the subsets explicitly, e.g. "
            f"subsets={{'sex_age': {{'categorical': ['sexo'], "
            f"'numerical': ['idade_atual']}}}}"
        )

    return {
        "sex_age": {
            "categorical": [sex],
            "numerical": [age],
        },
        "sex_age_duration": {
            "categorical": [sex],
            "numerical": [age, duration],
        },
    }


def _build_scenarios(categorical, numerical, subsets=None):
    """
    Builds the 2×N factorial dictionary.

    Returns {scenario: {"smotenc", "rebalancing", "subset",
                        "categorical", "numerical"}} keyed by
    "{rebalancing}|{subset}".
    """
    if subsets is None:
        subsets = _detect_subsets(categorical, numerical)

    sets = {
        "all": {"categorical": list(categorical),
                "numerical": list(numerical)},
    }
    for name, cols in subsets.items():
        if name == "all":
            raise ValueError(
                "The name 'all' is reserved; use another name for the subset."
            )
        sets[name] = {
            "categorical": list(cols["categorical"]),
            "numerical": list(cols["numerical"]),
        }

    # ---- validate the sets ----
    for name, cols in sets.items():
        cat, num = cols["categorical"], cols["numerical"]
        for c in cat:
            if c not in categorical:
                raise ValueError(
                    f"Categorical column '{c}' of set '{name}' is not in "
                    f"`categorical`={list(categorical)}."
                )
        for n in num:
            if n not in numerical:
                raise ValueError(
                    f"Numerical column '{n}' of set '{name}' is not in "
                    f"`numerical`={list(numerical)}."
                )
        duplicated = set(cat) & set(num)
        if duplicated:
            raise ValueError(
                f"Set '{name}' repeats columns in categorical and numerical: "
                f"{sorted(duplicated)}."
            )

    scenarios = {}
    for rebalancing in REBALANCING:
        for set_name, cols in sets.items():
            key = f"{rebalancing}|{set_name}"
            scenarios[key] = {
                "smotenc": rebalancing == "with_smotenc",
                "rebalancing": rebalancing,
                "subset": set_name,
                "categorical": cols["categorical"],
                "numerical": cols["numerical"],
            }
    return scenarios


class ModelFactory:
    def __init__(
        self,
        data,
        target,
        categorical,
        numerical,
        subsets: dict = None,
        algorithms: list = None,
        hyperparameter_grid: dict = None,
        results_path=None,
    ):
        """
        Parameters
        ----------
        data, target, categorical, numerical : same as v3/v4/v5.
        subsets : dict {name: {"categorical": [...], "numerical": [...]}},
            optional. If None, the sets "sex_age" and "sex_age_duration" are
            detected automatically.
        algorithms : subset of ALGORITHMS to train (default: all). Reduces the
            computational cost (6 scenarios × N algorithms).
        hyperparameter_grid : overrides the default grid (same structure as
            v3/v5: {"algorithm": {"model": est, "params": {...}}}). Useful for
            quick tests.
        results_path : path of the output Markdown. Default: project root /
            `model_factory_v6_results.md`.
        """
        self.data = data
        self.target = target
        self.categorical = list(categorical)
        self.numerical = list(numerical)

        if results_path is None:
            results_path = PROJECT_ROOT / RESULTS_FILE_DEFAULT
        self.results_path = Path(results_path)

        if algorithms is None:
            self.algorithms = list(ALGORITHMS)
        else:
            invalid = [m for m in algorithms if m not in ALGORITHMS]
            if invalid:
                raise ValueError(
                    f"Unknown algorithms: {invalid}. "
                    f"Valid: {list(ALGORITHMS)}."
                )
            self.algorithms = list(algorithms)

        self.hyperparameter_grid = hyperparameter_grid
        self.scenarios = _build_scenarios(
            self.categorical, self.numerical, subsets
        )

        self.y_test = None        # Series shared by every scenario
        self.results = None       # {scenario: {algorithm: {...}}}
        self.summary = None       # DataFrame
        self.nested_table = None  # DataFrame (ranked by outer AUC)
        self.comparisons = None   # DataFrame
        self.execution_config = None

    # ────────────────────────────────────────────────────────────────────
    # Predictors / target
    # ────────────────────────────────────────────────────────────────────
    def _split_predictors_and_target(self, *, data, target):
        y = data[target]
        X = data.drop([target], axis='columns')
        return X, y

    # ────────────────────────────────────────────────────────────────────
    # Model utilities
    # ────────────────────────────────────────────────────────────────────
    def _train_model(self, *, model, train_features, train_target):
        model.fit(train_features, train_target)
        return model

    def _evaluate_model(self, *, test_target, prediction):
        try:
            auc = roc_auc_score(test_target, prediction)
            report = classification_report(test_target, prediction)
        except Exception as e:
            print(f'Cannot compute ROC-AUC: {e}')
            return None, None
        return auc, report

    def _hyperparameter_grid(self, cat_features):
        """Hyperparameter grid; CatBoost receives the CATEGORICALS OF THE SCENARIO."""
        if self.hyperparameter_grid is not None:
            return self.hyperparameter_grid

        return {
            "logistic_regression": {
                "model": LogisticRegression(max_iter=1000, random_state=42),
                "params": {
                    "model__C": [0.01, 0.1, 1, 10],
                    "model__solver": ["lbfgs", "saga"],
                    "model__class_weight": [None, "balanced"],
                },
            },
            "random_forest": {
                "model": RandomForestClassifier(random_state=RANDOM_STATE),
                "params": {
                    "model__n_estimators": [100, 300],
                    "model__max_depth": [None, 5, 10],
                    "model__min_samples_leaf": [1, 5],
                    "model__class_weight": [None, "balanced"],
                },
            },
            "xgboost": {
                "model": XGBClassifier(
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    enable_categorical=True,
                ),
                "params": {
                    "model__n_estimators": [100, 300],
                    "model__max_depth": [3, 6],
                    "model__learning_rate": [0.01, 0.1],
                    "model__subsample": [0.8, 1.0],
                },
            },
            "lightgbm": {
                "model": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
                "params": {
                    "model__n_estimators": [100, 300],
                    "model__max_depth": [-1, 6],
                    "model__learning_rate": [0.01, 0.1],
                    "model__num_leaves": [31, 63],
                },
            },
            "catboost": {
                "model": CatBoostClassifier(
                    random_state=RANDOM_STATE, verbose=0, cat_features=cat_features
                ),
                "params": {
                    "model__iterations": [100, 300],
                    "model__depth": [4, 6],
                    "model__learning_rate": [0.01, 0.1],
                    "model__l2_leaf_reg": [1, 5],
                },
            },
        }

    def _build_pipeline(self, *, name, scenario, configuration):
        """
        Pipeline = Preprocessor + (optional) SMOTENC + model.

        Preprocessing (scale/impute/cast) runs INSIDE every fold (fit per fold).
        Without SMOTENC the 'smotenc' step simply does not exist — the
        parameters already use the 'model__' prefix, so the grid works the same.
        """
        if scenario["smotenc"] and not scenario["categorical"]:
            raise ValueError(
                f"Scenario with SMOTENC and no categorical columns: {scenario}. "
                f"SMOTENC requires at least one categorical variable."
            )
        steps = [(
            'pre',
            Preprocessor(
                numerical=scenario["numerical"],
                categorical=scenario["categorical"],
            ),
        )]
        if scenario["smotenc"]:
            steps.append((
                'smotenc',
                SMOTENC(
                    sampling_strategy='minority',
                    categorical_features='auto',
                    random_state=RANDOM_STATE,
                ),
            ))
        steps.append(('model', configuration["model"]))
        return Pipeline(steps)

    @staticmethod
    def _safe_name(text):
        return text.replace("|", "_").replace("-", "_").replace(" ", "_")

    @staticmethod
    def _ci_value(ci_table, metric, field):
        try:
            return float(ci_table.loc[metric, field])
        except KeyError:
            return float('nan')

    @staticmethod
    def _markdown_table(df: pd.DataFrame, decimals: int = 4) -> str:
        numeric_cols = df.select_dtypes(include="number").columns
        rounded = df.copy()
        rounded[numeric_cols] = rounded[numeric_cols].round(decimals)
        try:
            return rounded.to_markdown(index=False)
        except Exception:
            return "```\n" + rounded.to_string(index=False) + "\n```"

    @staticmethod
    def _build_grid_table(cv_results, best_params: str) -> pd.DataFrame:
    
        rows = []
        for idx, params in enumerate(cv_results["params"]):
            row = dict(params)
            row["mean_roc_auc_cv"] = cv_results["mean_test_score"][idx]
            row["std_roc_auc_cv"] = cv_results["std_test_score"][idx]
            row["rank"] = int(cv_results["rank_test_score"][idx])
            row["best"] = bool(params == best_params)
            rows.append(row)
        grid = pd.DataFrame(rows)
        return grid.sort_values("rank").reset_index(drop=True)

   
    def _nested_cv(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        pipeline,
        configuration: dict,
        scenario_key: str,
        algorithm_name: str,
        cv_outer: int = 5,
        cv_inner: int = 3,
        metric: str = "roc_auc",
        n_jobs: int = -1,
    ) -> dict:
        """
        Nested CV: for every fold of the OUTER CV, a GridSearchCV (inner CV)
        is fitted on that fold's training subset and evaluated on the outer
        validation fold. Returns the AUC per fold (mean/std), the AUC pooled
        on the OOF predictions and the best parameters of each outer fold —
        more information than a plain `cross_val_score(gs_est, ..., cv=outer_cv)`.
        """
        outer_cv = StratifiedKFold(
            n_splits=cv_outer, shuffle=True, random_state=RANDOM_STATE
        )
        inner_cv = StratifiedKFold(
            n_splits=cv_inner, shuffle=True, random_state=RANDOM_STATE
        )

        aucs = []
        oof = np.zeros(len(y_train), dtype=float)
        fold_rows = []

        for i_fold, (tr_idx, val_idx) in enumerate(
            outer_cv.split(X_train, y_train), 1
        ):
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=configuration["params"],
                scoring=metric,
                cv=inner_cv,
                n_jobs=n_jobs,
                refit=True,
            )
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            grid.fit(X_tr, y_tr)

            prob = grid.best_estimator_.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, prob)

            aucs.append(auc)
            oof[val_idx] = prob
            fold_rows.append({
                "fold": i_fold,
                "n_train": len(tr_idx),
                "n_validation": len(val_idx),
                "auc": auc,
                "best_params": json.dumps(
                    {k: str(v) for k, v in grid.best_params_.items()},
                    ensure_ascii=False,
                ),
            })
            print(f"    outer fold {i_fold}/{cv_outer} | AUC {auc:.4f} | "
                  f"params {grid.best_params_}")

        auc_oof = roc_auc_score(y_train, oof)
        print(f"  → outer AUC (nested): {np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
              f"  | pooled OOF AUC: {auc_oof:.4f}")

        return {
            "auc_folds": aucs,
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),   # ddof=0, same as cross_val_score
            "auc_oof": float(auc_oof),
            "prob_oof": oof,
            "folds_table": pd.DataFrame(fold_rows),
            "cv_outer": cv_outer,
            "cv_inner": cv_inner,
        }

   
    def _grid_search_and_evaluate(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        scenario_key: str,
        scenario: dict,
        algorithm_name: str,
        pipeline,
        configuration: dict,
        cv_inner: int = 3,
        metric: str = "roc_auc",
        n_jobs: int = -1,
        threshold: float = 0.5,
        n_boot: int = 2000,
        nested: dict = None,
    ) -> dict:
        """
        GridSearchCV (inner CV) on the WHOLE training set with refit=True — the
        final model of (scenario, algorithm) — and holdout evaluation with
        bootstrap CI. Equivalent to the final step of the example notebook.
        """
        inner_cv = StratifiedKFold(
            n_splits=cv_inner, shuffle=True, random_state=RANDOM_STATE
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=configuration["params"],
            scoring=metric,
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=True,
            verbose=1,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_cv_score = grid.best_score_

        prediction = best_model.predict(X_test)
        prob = best_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, prob)

        print(f"Computing bootstrap CI ({n_boot} resamples)...")
        ci_table = bootstrap_metrics(y_test, prob, threshold=threshold, n_boot=n_boot)

        report = classification_report(y_test, prediction)
        if nested is not None:
            print(f"Outer AUC (nested): {nested['auc_mean']:.4f} ± "
                  f"{nested['auc_std']:.4f}")
        print(f"\nBest parameters: {best_params}")
        print(f"Final CV AUC: {best_cv_score:.4f}")
        print(f"Test AUC    : {test_auc:.4f}")
        print(f"\n{report}")
        print(f"\nTest metrics with 95% CI (threshold = {threshold}):")
        print(ci_table.round(4))

        grid_table = self._build_grid_table(grid.cv_results_, best_params)
        print(f"\nAll combinations tested ({len(grid_table)}):")
        print(grid_table.round(4).to_string(index=False))

        return {
            "best_model": best_model,
            "best_test_auc": test_auc,
            "best_cv_auc": best_cv_score,
            "best_params": best_params,
            "nested": nested,
            "report": report,
            "test_probs": prob,
            "ci": ci_table,
            "grid": grid_table,
            "scenario": scenario_key,
            "smotenc": scenario["smotenc"],
            "rebalancing": scenario["rebalancing"],
            "subset": scenario["subset"],
        }

   
    def _build_summary(self, results: dict) -> pd.DataFrame:
        rows = []
        for scenario_key, by_model in results.items():
            scenario = self.scenarios[scenario_key]
            for algorithm_name, res in by_model.items():
                ci = res["ci"]
                nested = res.get("nested") or {}
                rows.append({
                    "scenario": scenario_key,
                    "rebalancing": scenario["rebalancing"],
                    "smotenc": scenario["smotenc"],
                    "subset": scenario["subset"],
                    "algorithm": algorithm_name,
                    "n_categorical": len(scenario["categorical"]),
                    "n_numerical": len(scenario["numerical"]),
                    "auc_nested": nested.get("auc_mean", float('nan')),
                    "auc_nested_std": nested.get("auc_std", float('nan')),
                    "auc_nested_oof": nested.get("auc_oof", float('nan')),
                    "auc_cv": res["best_cv_auc"],
                    "auc_test": self._ci_value(ci, "auroc", "estimate"),
                    "auc_ci_low": self._ci_value(ci, "auroc", "ci_low"),
                    "auc_ci_high": self._ci_value(ci, "auroc", "ci_high"),
                    "brier": self._ci_value(ci, "brier", "estimate"),
                    "brier_ci_low": self._ci_value(ci, "brier", "ci_low"),
                    "brier_ci_high": self._ci_value(ci, "brier", "ci_high"),
                    "scaled_brier": self._ci_value(ci, "scaled_brier", "estimate"),
                    "calib_intercept": self._ci_value(ci, "citl", "estimate"),
                    "calib_slope": self._ci_value(ci, "slope", "estimate"),
                    "sensitivity": self._ci_value(ci, "sensitivity", "estimate"),
                    "specificity": self._ci_value(ci, "specificity", "estimate"),
                    "ppv": self._ci_value(ci, "ppv", "estimate"),
                    "npv": self._ci_value(ci, "npv", "estimate"),
                    "accuracy": self._ci_value(ci, "accuracy", "estimate"),
                    "f1": self._ci_value(ci, "f1", "estimate"),
                })
        return pd.DataFrame(rows)

    def _build_nested_table(self, results: dict) -> pd.DataFrame:
        """Scenario × algorithm ranking by outer AUC (nested CV)."""
        rows = []
        for scenario_key, by_model in results.items():
            scenario = self.scenarios[scenario_key]
            for algorithm_name, res in by_model.items():
                nested = res.get("nested")
                if nested is None:
                    continue
                rows.append({
                    "scenario": scenario_key,
                    "rebalancing": scenario["rebalancing"],
                    "subset": scenario["subset"],
                    "algorithm": algorithm_name,
                    "auc_nested": nested["auc_mean"],
                    "auc_nested_std": nested["auc_std"],
                    "auc_nested_oof": nested["auc_oof"],
                })
        if not rows:
            return pd.DataFrame()
        return (pd.DataFrame(rows)
                .sort_values("auc_nested", ascending=False)
                .reset_index(drop=True))

    
    def _contrast_pairs(self):
        """Default pairs: SMOTENC effect and variable-set effect."""
        subsets = []
        for scenario in self.scenarios.values():
            if scenario["subset"] not in subsets:
                subsets.append(scenario["subset"])

        pairs = []
        for s in subsets:
            pairs.append((
                f"with_smotenc|{s}", f"without_smotenc|{s}", "smotenc_effect",
            ))
        for rebalancing in REBALANCING:
            for s in subsets:
                if s == "all":
                    continue
                pairs.append((
                    f"{rebalancing}|all", f"{rebalancing}|{s}",
                    "variable_set_effect",
                ))
        return pairs

    def compare_scenarios(
        self, algorithm: str = None, n_boot: int = 2000, pairs: list = None
    ) -> pd.DataFrame:
        """
        Paired ΔAUC (A − B) with 95% bootstrap CI, using the SAME y_test.

        algorithm : restricts to a single algorithm (default: all in the scenario).
        pairs     : optional list of (scenario_a, scenario_b, contrast_type).
        """
        if self.results is None:
            raise RuntimeError("Run run() before comparing scenarios.")
        if pairs is None:
            pairs = self._contrast_pairs()

        algorithm_names = [algorithm] if algorithm else list(self.algorithms)
        rows = []
        for algorithm_name in algorithm_names:
            for scenario_a, scenario_b, kind in pairs:
                for key in (scenario_a, scenario_b):
                    if key not in self.results:
                        raise KeyError(
                            f"Scenario '{key}' was not run "
                            f"(available scenarios: {list(self.results)})."
                        )
                if algorithm_name not in self.results[scenario_a]:
                    continue
                res_a = self.results[scenario_a][algorithm_name]
                res_b = self.results[scenario_b][algorithm_name]

                diff = bootstrap_auc_difference(
                    self.y_test,
                    res_a["test_probs"],
                    res_b["test_probs"],
                    n_boot=n_boot,
                )
                row = {
                    "algorithm": algorithm_name,
                    "contrast_type": kind,
                    "scenario_a": scenario_a,
                    "scenario_b": scenario_b,
                    "auc_a": res_a["best_test_auc"],
                    "auc_b": res_b["best_test_auc"],
                    "dif_auc": diff["dif_auc"],
                    "ci_low": diff["ci_low"],
                    "ci_high": diff["ci_high"],
                    "significant": bool(diff["ci_low"] > 0 or diff["ci_high"] < 0),
                    "n_boot": n_boot,
                }
                rows.append(row)
                print(
                    f"ΔAUC {algorithm_name}: {scenario_a} − {scenario_b} = "
                    f"{diff['dif_auc']:.4f} "
                    f"[95% CI {diff['ci_low']:.4f}; {diff['ci_high']:.4f}]"
                    + ("  *" if row["significant"] else "")
                )

        self.comparisons = pd.DataFrame(rows)
        return self.comparisons

    def compare_auc(
        self, scenario: str, algorithm_a: str, algorithm_b: str, n_boot: int = 2000
    ) -> dict:
        """ΔAUC between TWO ALGORITHMS within the same scenario (bootstrap CI)."""
        if self.results is None or scenario not in self.results:
            raise RuntimeError(
                f"Scenario '{scenario}' unavailable; run run() first."
            )
        diff = bootstrap_auc_difference(
            self.y_test,
            self.results[scenario][algorithm_a]["test_probs"],
            self.results[scenario][algorithm_b]["test_probs"],
            n_boot=n_boot,
        )
        print(
            f"ΔAUC ({algorithm_a} − {algorithm_b}) in [{scenario}] = "
            f"{diff['dif_auc']:.4f} [95% CI {diff['ci_low']:.4f}; {diff['ci_high']:.4f}]"
        )
        return diff

    
    def _build_markdown(self) -> str:
        """Builds the Markdown section of the current execution (does not write)."""
        cfg = self.execution_config or {}
        sections = []
        sections.append(
            f"# Execution — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # ---- configuration ----
        sections.append("\n## 1. Configuration\n")
        sections.append("| Item | Value |")
        sections.append("|---|---|")
        sections.append(f"| Data | {self.data.shape[0]} rows × "
                        f"{self.data.shape[1]} columns |")
        sections.append(f"| Target | `{self.target}` |")
        sections.append(f"| Categorical | {', '.join(self.categorical)} |")
        sections.append(f"| Numerical | {', '.join(self.numerical)} |")
        sections.append(f"| Train/test split | "
                        f"{cfg.get('split', '80/20, stratify, random_state=42')} |")
        sections.append(f"| Outer CV (nested) | "
                        f"{cfg.get('cv_outer', '—')} stratified folds |")
        sections.append(f"| Inner CV (grid) | "
                        f"{cfg.get('cv_inner', '—')} stratified folds |")
        sections.append(f"| Search metric | {cfg.get('metric', 'roc_auc')} |")
        sections.append(f"| Threshold | {cfg.get('threshold', 0.5)} |")
        sections.append(f"| Bootstrap | {cfg.get('n_boot', '—')} resamples |")
        sections.append(f"| Scenarios × algorithms | "
                        f"{len(self.scenarios)} × {len(self.algorithms)} |")
        sections.append(
            "\nPreprocessing (scale/impute/cast) and SMOTENC are pipeline "
            "steps: fit PER FOLD (no leakage in the CV)."
        )

        # ---- nested CV ----
        if self.nested_table is not None and len(self.nested_table):
            sections.append(
                "\n## 2. Nested Cross-Validation — algorithm comparison\n"
            )
            sections.append(
                "Outer AUC (mean ± std of the outer folds) and pooled OOF AUC. "
                "Comparisons are made on the TRAINING set only; the test set is "
                "used only in the final evaluation."
            )
            sections.append("")
            sections.append(self._markdown_table(self.nested_table))
            best = self.nested_table.iloc[0]
            sections.append(
                f"\n**Best combination according to nested CV:** "
                f"`{best['scenario']}` + `{best['algorithm']}` "
                f"(outer AUC {best['auc_nested']:.4f} ± "
                f"{best['auc_nested_std']:.4f})."
            )

        # ---- summary ----
        if self.summary is not None:
            sections.append("\n## 3. Summary — scenario × algorithm\n")
            sections.append(self._markdown_table(self.summary))

        # ---- comparisons ----
        if self.comparisons is not None and len(self.comparisons):
            sections.append("\n## 4. Paired comparisons (ΔAUC with 95% CI)\n")
            sections.append(self._markdown_table(self.comparisons))
            sections.append(
                "\n`significant` = the 95% CI of the difference does not cross zero."
            )

        # ---- detail per scenario/algorithm ----
        sections.append("\n## 5. Details per scenario × algorithm\n")
        for scenario_key, by_model in (self.results or {}).items():
            scenario = self.scenarios[scenario_key]
            sections.append(f"\n### {scenario_key}\n")
            sections.append(f"- SMOTENC: **{'yes' if scenario['smotenc'] else 'no'}**")
            sections.append(f"- Categorical: {', '.join(scenario['categorical']) or '—'}")
            sections.append(f"- Numerical: {', '.join(scenario['numerical']) or '—'}")

            for algorithm_name, res in by_model.items():
                nested = res.get("nested") or {}
                sections.append(f"\n#### {algorithm_name}\n")
                sections.append("| Metric | Value |")
                sections.append("|---|---|")
                if nested:
                    sections.append(
                        f"| Outer AUC (nested) | "
                        f"{nested['auc_mean']:.4f} ± {nested['auc_std']:.4f} |"
                    )
                    sections.append(
                        f"| OOF AUC (pooled) | {nested['auc_oof']:.4f} |"
                    )
                sections.append(f"| AUC (final CV) | {res['best_cv_auc']:.4f} |")
                sections.append(f"| AUC (test) | {res['best_test_auc']:.4f} |")

                if nested and nested.get("folds_table") is not None:
                    sections.append("\n**Nested CV — per outer fold**\n")
                    sections.append(
                        self._markdown_table(nested["folds_table"])
                    )

                sections.append("\n**Best parameters (final grid)**\n")
                sections.append("```json")
                sections.append(
                    json.dumps(
                        {k: str(v) for k, v in res["best_params"].items()},
                        indent=2, ensure_ascii=False,
                    )
                )
                sections.append("```")

                sections.append("\n**Test metrics with 95% CI**\n")
                sections.append(self._markdown_table(res["ci"]))

                sections.append("\n**Classification report (threshold 0.5)**\n")
                sections.append("```")
                sections.append(res["report"].rstrip())
                sections.append("```")

                grid = res.get("grid")
                if grid is not None and len(grid):
                    sections.append(
                        f"\n**Full grid — {len(grid)} combinations "
                        f"(sorted by rank)**\n"
                    )
                    sections.append(self._markdown_table(grid))

        sections.append("")
        return "\n".join(sections)

    def save_results_md(self, path=None) -> str:
        """
        Appends the section of the current execution to the results Markdown
        (creates the file if it does not exist). Returns the written path.
        """
        if self.results is None:
            raise RuntimeError("Run run() before writing the Markdown.")

        destination = Path(path) if path is not None else self.results_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        content = self._build_markdown()
        if destination.exists() and destination.stat().st_size > 0:
            content = "\n\n---\n\n" + content
            with open(destination, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(destination, "w", encoding="utf-8") as f:
                f.write(content)

        return str(destination)

    # ────────────────────────────────────────────────────────────────────
    # Orchestration
    # ────────────────────────────────────────────────────────────────────
    def run(
        self,
        test_size: float = 0.2,
        cv_outer: int = 5,
        cv_inner: int = 3,
        metric: str = "roc_auc",
        n_jobs: int = -1,
        threshold: float = 0.5,
        n_boot: int = 2000,
    ) -> dict:
        
        total = len(self.scenarios) * len(self.algorithms)
        self.execution_config = {
            "test_size": test_size,
            "split": (f"{int((1 - test_size) * 100)}/{int(test_size * 100)}, "
                      f"stratify, random_state=42"),
            "cv_outer": cv_outer,
            "cv_inner": cv_inner,
            "metric": metric,
            "n_jobs": n_jobs,
            "threshold": threshold,
            "n_boot": n_boot,
        }

        print("[ 1/5 ] Splitting predictors and target...")
        X, y = self._split_predictors_and_target(
            data=self.data, target=self.target
        )

        print(
            f"[ 2/5 ] Splitting train/test "
            f"({int((1 - test_size) * 100)}/{int(test_size * 100)}) "
            f"— single split shared by all {len(self.scenarios)} scenarios..."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y,
        )
        self.y_test = y_test

        print(
            f"[ 3/5 ] Nested CV on the training set "
            f"(outer={cv_outer} × inner={cv_inner}) + "
            f"[ 4/5 ] final grid + test — "
            f"{len(self.scenarios)} scenarios × {len(self.algorithms)} algorithms "
            f"= {total} combinations..."
        )
        results = {}
        for i, (scenario_key, scenario) in enumerate(self.scenarios.items(), 1):
            print("\n" + "#" * 80)
            print(f"# Scenario {i}/{len(self.scenarios)}: {scenario_key}")
            print(f"#   categorical={scenario['categorical']} | "
                  f"numerical={scenario['numerical']}")
            print("#" * 80)

            columns = scenario["categorical"] + scenario["numerical"]
            X_tr_scen = X_train[columns].copy()
            X_te_scen = X_test[columns].copy()

            grid = self._hyperparameter_grid(scenario["categorical"])
            results[scenario_key] = {}

            for algorithm_name in self.algorithms:
                configuration = grid[algorithm_name]
                print(f"\n{'=' * 60}")
                print(f"  {algorithm_name.upper()}  [{scenario_key}]")
                print(f"{'=' * 60}")

                pipeline = self._build_pipeline(
                    name=algorithm_name, scenario=scenario,
                    configuration=configuration,
                )

                print(f"\n  --- Nested CV (outer={cv_outer} × "
                      f"inner={cv_inner}) ---")
                nested = self._nested_cv(
                    X_train=X_tr_scen,
                    y_train=y_train,
                    pipeline=pipeline,
                    configuration=configuration,
                    scenario_key=scenario_key,
                    algorithm_name=algorithm_name,
                    cv_outer=cv_outer,
                    cv_inner=cv_inner,
                    metric=metric,
                    n_jobs=n_jobs,
                )

                print(f"\n  --- Final grid (refit on the whole training set) + "
                      f"test evaluation ---")
                results[scenario_key][algorithm_name] = self._grid_search_and_evaluate(
                    X_train=X_tr_scen,
                    y_train=y_train,
                    X_test=X_te_scen,
                    y_test=y_test,
                    scenario_key=scenario_key,
                    scenario=scenario,
                    algorithm_name=algorithm_name,
                    pipeline=pipeline,
                    configuration=configuration,
                    cv_inner=cv_inner,
                    metric=metric,
                    n_jobs=n_jobs,
                    threshold=threshold,
                    n_boot=n_boot,
                    nested=nested,
                )

        self.results = results
        self.summary = self._build_summary(results)
        self.nested_table = self._build_nested_table(results)

        print("\n" + "=" * 30 + " NESTED CV — RANKING (OUTER AUC) " + "=" * 30)
        print(self.nested_table.round(4).to_string(index=False))
        if len(self.nested_table):
            best = self.nested_table.iloc[0]
            print(
                f"\nBest combination according to nested CV: "
                f"{best['scenario']} + {best['algorithm']} "
                f"(outer AUC {best['auc_nested']:.4f} ± "
                f"{best['auc_nested_std']:.4f})"
            )

        print("\n" + "=" * 30 + " SUMMARY — SCENARIO × ALGORITHM " + "=" * 30)
        print(self.summary.round(4).to_string(index=False))

        print("\n" + "=" * 30 + " PAIRED COMPARISONS (ΔAUC) " + "=" * 30)
        comparisons = self.compare_scenarios(n_boot=n_boot)
        print(comparisons.round(4).to_string(index=False))

        markdown_file = self.save_results_md()
        print("\n" + "-" * 30 + " RESULTS WRITTEN " + "-" * 30)
        print(f"Markdown: {markdown_file}")

        return {
            "results": results,
            "summary": self.summary,
            "nested_table": self.nested_table,
            "comparisons": comparisons,
            "y_test": y_test,
            "markdown_file": markdown_file,
        }


"""# Data Processing Helpers"""


def encode_target_as_zero_one(*, data, target, control=1):
   
    data = data.copy()

    data[target] = np.where(
        data[target].isna(),
        data[target],
        np.where(data[target] == control, 0, 1),
    )

    return data


def categorize_income_by_minimum_wage(value, minimum_wage=678):
    
    if pd.isna(value):
        return None
    elif value <= minimum_wage * 0.25:
        return 1
    elif value <= minimum_wage * 0.50:
        return 2
    elif value <= minimum_wage * 1.00:
        return 3
    elif value <= minimum_wage * 2.00:
        return 4
    elif value <= minimum_wage * 3.00:
        return 5
    elif value <= minimum_wage * 5.00:
        return 6
    else:
        return 7


def generate_preprocessor(categorical, numerical):
   
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical),
        ('categorical', categorical_pipeline, categorical),
    ]).set_output(transform="pandas")

    return preprocessor


def load_all_external_datasets(preprocessor, target, columns=None):
    
    external_datasets = {}

    candidate_datasets = {
        'PNS_2013': data_to_validate_pns_2013,
        'Hospital': data_to_validate_hospital,
    }

    for name, df in candidate_datasets.items():
        if df is None:
            print(f"Warning: dataset '{name}' is not available - skipping.")
            continue
        if target not in df.columns:
            print(f"Warning: target column '{target}' not found in '{name}' - skipping.")
            continue

        y = df[target]
        X = df.drop([target], axis='columns')
        if columns is not None:
            missing = [c for c in columns if c not in X.columns]
            if missing:
                print(f"Warning: columns {missing} not found in '{name}' - skipping.")
                continue
            X = X[columns]
        X_transformed = preprocessor.transform(X)
        external_datasets[name] = (X_transformed, y)

    return external_datasets


"""# Dataset Import"""

raw_pns_2019 = pl.read_csv("./pns_2019.csv")
raw_pns_2013 = pd.read_csv("./pns_2013.csv")
raw_hospital = pd.read_csv("./hospital.csv")

"""# Data Processing

## Processing the PNS 2019 data
"""

filter_pns_2019 = raw_pns_2019.filter(pl.col('C008') >= '18', pl.col('Q03001') == '1', pl.col('Q03002') != '1')

filter_pns_2019 = filter_pns_2019.to_pandas()

filter_pns_2019 = filter_pns_2019[
    (filter_pns_2019["V0025A"] == 1) &  # Answered the individual questionnaire
    (filter_pns_2019["M001"] == 1) &   # Interview of the selected adult
    (filter_pns_2019["C008"] >= 18) &  # Age >= 18
    (filter_pns_2019["Q03001"] == 1) & # Diagnosis of diabetes
    (filter_pns_2019["Q03002"] != 1) & # Diabetes outside of pregnancy
    (filter_pns_2019["Q055014"] == 2)  # No foot ulcers or amputations
]

target = "limitacao_funcional"
filter_pns_2019['duracao'] = (
    filter_pns_2019['C008'] - filter_pns_2019['Q031']
)

data_to_model_pns_2019 = filter_pns_2019.rename(columns={
    'C001': 'numero_de_co_habitantes',            # number of co-residents
    'C006': 'sexo',                               # sex
    'C009': 'etnia',                              # ethnicity
    'C008': 'idade_atual',                        # current age
    'VDF004': 'faixa_salario_minimo_per_capita',  # per-capita income bracket (minimum wages)
    'Q00201': 'hipertensao',                      # hypertension
    'Q03802': 'uso_insulina',                     # insulin use
    'Q058': "limitacao_funcional"
})

data_to_model_pns_2019 = encode_target_as_zero_one(data=data_to_model_pns_2019, target='etnia', control=1)
data_to_model_pns_2019 = encode_target_as_zero_one(data=data_to_model_pns_2019, target=target)
data_to_model_pns_2019 = encode_target_as_zero_one(data=data_to_model_pns_2019, target='hipertensao', control=2)
data_to_model_pns_2019 = encode_target_as_zero_one(data=data_to_model_pns_2019, target='uso_insulina', control=2)
data_to_model_pns_2019 = encode_target_as_zero_one(data=data_to_model_pns_2019, target='sexo')

# Column groups used by the preprocessor
numerical = ['numero_de_co_habitantes', 'idade_atual', 'faixa_salario_minimo_per_capita', 'duracao']
categorical = ['sexo', 'etnia', 'hipertensao', 'uso_insulina']

selected_columns = list()
selected_columns.append(target)
selected_columns.extend(numerical)
selected_columns.extend(categorical)

data_to_model_pns_2019 = data_to_model_pns_2019[selected_columns]

# Keep only elderly participants (age 60 or older)
data_to_model_pns_2019 = data_to_model_pns_2019.query('idade_atual >= 60')

"""## Processing the PNS 2013 data


"""

filtered_pns_2013 = raw_pns_2013[
    (raw_pns_2013["V0025"] == '1') &    # Resident aged 18+ selected to answer the individual questionnaire
    (raw_pns_2013["M001"] == '1') &     # Interview with the selected adult was actually Conducted
    (raw_pns_2013["C008"].astype('Int64') >= 18).fillna(False) &    # Resident's age on the reference date is 18 or older
    (raw_pns_2013["Q05506"] == '2') &   # No history of foot ulcer/wound due to diabetes (1=Yes, 2=No)
    (raw_pns_2013["Q05507"] == '2')     # No history of limb amputation due to diabetes (1=Yes, 2=No)
]


categorical_2013 = [
    'C001',    # Number of people in the household (can be treated as numerical or categorical depending on the model)
    'C006',    # Sex
    'C009',    # Race/Color
    'Q002',    # Hypertension diagnosis
    'Q03402'   # Insulin use (replacing Q03802)
]

numerical_2013 = [
    'C008',    # Age
    'Q031',    # Age at first diabetes diagnosis
    'VDF003'   # Continuous per-capita household income (replacing VDF004)
]

columns_2013 = []

columns_2013.extend(categorical_2013)
columns_2013.extend(numerical_2013)
columns_2013.extend(['Q058'])

selected_pns_2013 = filtered_pns_2013[columns_2013].copy()

selected_pns_2013[numerical_2013] = selected_pns_2013[numerical_2013].astype('Int64')
selected_pns_2013['duracao'] = selected_pns_2013['C008'] - selected_pns_2013['Q031']

binarized_pns_2013 = selected_pns_2013.copy()

binarized_pns_2013 = binarized_pns_2013.rename(columns={
    'C001': 'numero_de_co_habitantes',
    'C006': 'sexo',
    'C009': 'etnia',
    'C008': 'idade_atual',
    'VDF003': 'rendimento_per_capita_continuo',  # Replaces VDF004 (which doesn't exist in 2013)
    'Q002': 'hipertensao',                       # In 2013 it's Q002, without the "01" suffix
    'Q03402': 'uso_insulina',                    # In 2013 insulin use is mapped to Q03402
    'Q058': "limitacao_funcional"

})


target = "limitacao_funcional"
binarized_pns_2013 = encode_target_as_zero_one(data=binarized_pns_2013, target='sexo')
binarized_pns_2013 = encode_target_as_zero_one(data=binarized_pns_2013, target='hipertensao', control='3')
binarized_pns_2013 = encode_target_as_zero_one(data=binarized_pns_2013, target='etnia')
binarized_pns_2013 = encode_target_as_zero_one(data=binarized_pns_2013, target='uso_insulina', control='2')
binarized_pns_2013 = encode_target_as_zero_one(data=binarized_pns_2013, target=target)


binarized_pns_2013['faixa_salario_minimo_per_capita'] = binarized_pns_2013['rendimento_per_capita_continuo'].apply(categorize_income_by_minimum_wage)

data_to_validate_pns_2013 = binarized_pns_2013.copy()

data_to_validate_pns_2013.drop(['Q031', 'rendimento_per_capita_continuo'], axis=1, inplace=True)

data_to_validate_pns_2013 = data_to_validate_pns_2013[data_to_validate_pns_2013['idade_atual'] >= 60]

"""## Processing the Hospital data

"""

selected_columns = [
    'idade_atual', 'sexo', 'etnia', 'renda_familiar_em_reais',
    'numero_de_co_habitantes', 'hipertensao', 'ano_do_diagnostico',
    'data_da_avaliacao', 'uso_insulina', 'sppb_pontuacao_final',
]
selected_df = raw_hospital[selected_columns].copy()

selected_df['renda_per_capita'] = selected_df['renda_familiar_em_reais'] / selected_df['numero_de_co_habitantes']
selected_df['faixa_salario_minimo_per_capita'] = selected_df['renda_per_capita'].apply(categorize_income_by_minimum_wage)
selected_df['duracao'] = pd.to_datetime(selected_df['data_da_avaliacao']).dt.year.astype('Int64') - selected_df['ano_do_diagnostico']


selected_df['limitacao_funcional'] = np.where(
    selected_df['sppb_pontuacao_final'].isna(),
    selected_df['sppb_pontuacao_final'],
    np.where(selected_df['sppb_pontuacao_final'] >= 10, 0, 1),
)
selected_df = selected_df.dropna(subset=['limitacao_funcional'])
selected_df['limitacao_funcional'] = selected_df['limitacao_funcional'].astype(int)

data_to_validate_hospital = selected_df.drop(
    [
        'ano_do_diagnostico', 'renda_per_capita', 'renda_familiar_em_reais',
        'data_da_avaliacao', 'sppb_pontuacao_final',
    ],
    axis=1,
)


data_to_validate_hospital = encode_target_as_zero_one(
    data=data_to_validate_hospital, target='sexo', control=2)
data_to_validate_hospital = encode_target_as_zero_one(
    data=data_to_validate_hospital, target='etnia', control=1)

"""# Model Selection 
"""

data = data_to_model_pns_2019

TARGET = "limitacao_funcional"

print(data.isna().sum())

numerical = ['numero_de_co_habitantes', 'idade_atual', 'faixa_salario_minimo_per_capita', 'duracao']
categorical = ['sexo', 'etnia', 'hipertensao', 'uso_insulina']
print(TARGET)

factory = ModelFactory(
    data=data,
    target=TARGET,
    categorical=categorical,
    numerical=numerical,
)

selection = factory.run(cv_outer=5, cv_inner=3, n_boot=2000)

"""# Final Model
"""

ranking = selection['nested_table']
best_row = ranking.iloc[0]
best_scenario_key = best_row['scenario']
best_algorithm = best_row['algorithm']

print(f"Selected combination: {best_scenario_key} + {best_algorithm}")
print(f"Outer AUC (nested): {best_row['auc_nested']:.4f} ± {best_row['auc_nested_std']:.4f}")

scenario = factory.scenarios[best_scenario_key]
winner = selection['results'][best_scenario_key][best_algorithm]
feature_columns = scenario['categorical'] + scenario['numerical']

# ── Rebuild the winning pipeline and refit it on 100% of PNS 2019 ──────────
configuration = factory._hyperparameter_grid(scenario['categorical'])[best_algorithm]
final_pipeline = factory._build_pipeline(
    name=best_algorithm, scenario=scenario, configuration=configuration,
)
final_pipeline.set_params(**winner['best_params'])

X_full = data[feature_columns]
y_full = data[TARGET]
final_pipeline.fit(X_full, y_full)

preprocessor = final_pipeline.named_steps['pre']
final_model = final_pipeline.named_steps['model']

# ── Save model, preprocessor and input column order ────────────────────────
MODEL_FOLDER = "modelos_salvos"
PREPROC_PATH = os.path.join(MODEL_FOLDER, 'preprocessador_pns2019.joblib')
MODEL_JOBLIB_PATH = os.path.join(MODEL_FOLDER, f'{best_algorithm}_pns2019.joblib')
MODEL_NATIVE_PATH = os.path.join(MODEL_FOLDER, f'{best_algorithm}_pns2019.json')
PIPELINE_PATH = os.path.join(MODEL_FOLDER, 'pipeline_final_pns2019.joblib')
COLUMNS_PATH = os.path.join(MODEL_FOLDER, 'colunas_entrada.json')

os.makedirs(MODEL_FOLDER, exist_ok=True)
joblib.dump(preprocessor, PREPROC_PATH)
joblib.dump(final_model, MODEL_JOBLIB_PATH)
joblib.dump(final_pipeline, PIPELINE_PATH)
with open(COLUMNS_PATH, 'w', encoding='utf-8') as f:
    json.dump(feature_columns, f, ensure_ascii=False, indent=2)

# Native serialization when the estimator supports it (XGBoost / CatBoost)
if hasattr(final_model, 'save_model'):
    try:
        final_model.save_model(MODEL_NATIVE_PATH)
        print(f'Native model saved to: {MODEL_NATIVE_PATH}')
    except Exception as exc:
        print(f'Native model save skipped ({exc}); joblib copy is available.')

print(f'Preprocessor saved to: {PREPROC_PATH}')
print(f'Model (joblib) saved to: {MODEL_JOBLIB_PATH}')
print(f'Full pipeline saved to: {PIPELINE_PATH}')
print(f'Input columns saved to: {COLUMNS_PATH}')

"""# Optional TabPFN benchmark


"""

API_TOKEN = os.environ.get('TABPFN_TOKEN')
if not API_TOKEN:
    print('TABPFN_TOKEN not set — skipping the optional TabPFN benchmark.')
elif not _HAS_TABPFN:
    print('tabpfn-client not installed — skipping the optional TabPFN benchmark.')
else:
    set_access_token(API_TOKEN)

    X_tab = generate_preprocessor(categorical, numerical).fit_transform(
        data.drop([TARGET], axis='columns')
    )

    tabpfn_model = TabPFNClassifier()
    tabpfn_model.fit(X_tab, y_full)

    tabpfn_train_prob = tabpfn_model.predict_proba(X_tab)[:, 1]
    tabpfn_train_auc = roc_auc_score(y_full, tabpfn_train_prob)
    print(f'TabPFN ROC AUC (training data): {tabpfn_train_auc:.4f}')

    tabpfn_fpr, tabpfn_tpr, _ = roc_curve(y_full, tabpfn_train_prob)
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.plot(tabpfn_fpr, tabpfn_tpr, color='black', lw=1.5, label=f'ROC Curve (AUC = {tabpfn_train_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', label='Random Guess')
    ax.set_xlabel('1 - Specificity (FPR)')
    ax.set_ylabel('Sensitivity (TPR)')
    ax.set_title('TabPFN: ROC Curve (training data)')
    ax.legend()
    plt.tight_layout()
    plt.show()

"""# External Validation of the Final Model 

Stage 1 — the frozen model on each external cohort (discrimination,
calibration, threshold metrics, DCA; 95% CIs from 2000 bootstrap resamples).
Stage 2 — intercept-only and Platt recalibration estimated out-of-fold,
reported with paired differences versus the original model.
"""

external_datasets = load_all_external_datasets(
    preprocessor, target=TARGET, columns=feature_columns,
)

evaluation = run_evaluation(
    model=final_model,
    datasets=external_datasets,
    output_dir='output',
)

print('\nStage 1 tables : output/stage1_original_model/')
print('Stage 2 tables : output/stage2_calibration/')
