\
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# optional
try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

try:
    from catboost import CatBoostClassifier
    HAVE_CAT = True
except Exception:
    HAVE_CAT = False

try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    HAVE_ORD = True
except Exception:
    HAVE_ORD = False

from .utils import load_config, ensure_dirs, set_seed, save_table
from .preprocessing import load_and_sort, fill_and_clean
from .scoring import add_clinical_scores, add_targets, feature_lists, LABELS
from .evaluation import eval_from_preds, macro_auprc_from_proba, plot_pr_curves_by_class

def split_data(df: pd.DataFrame, seed: int, test_size: float):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df['PersonnelID']))
    train = df.iloc[train_idx].reset_index(drop=True)
    test  = df.iloc[test_idx].reset_index(drop=True)
    return train, test

def train_and_eval_all(cfg: Dict):
    set_seed(cfg['random_seed'])
    paths = cfg['paths']
    cont_cols, score_cols = feature_lists()
    features = cont_cols + score_cols
    classes = np.array([0,1,2,3])

    # load data
    df = load_and_sort(paths['data_csv'])
    df = fill_and_clean(df)
    df = add_clinical_scores(df)
    df = add_targets(df)

    train, test = split_data(df, cfg['random_seed'], cfg['test_size'])

    X_tr, y_tr = train[features].values, train['NextRiskCode'].values
    X_te, y_te = test[features].values,  test['NextRiskCode'].values

    results = []

    # XGBoost
    xgb_model = xgb.XGBClassifier(**cfg['models']['xgboost'])
    xgb_model.fit(X_tr, y_tr)
    y_pred = xgb_model.predict(X_te)
    m, _ = eval_from_preds(y_te, y_pred, LABELS)
    proba = xgb_model.predict_proba(X_te)
    mauprc, _ = macro_auprc_from_proba(y_te, proba, classes)
    results.append(dict(Model="XGBoost", **m, Macro_AUPRC=mauprc))

    # Random Forest
    rf = RandomForestClassifier(**cfg['models']['random_forest'])
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    m, _ = eval_from_preds(y_te, y_pred, LABELS)
    proba = rf.predict_proba(X_te)
    mauprc, _ = macro_auprc_from_proba(y_te, proba, classes)
    results.append(dict(Model="Random Forest", **m, Macro_AUPRC=mauprc))

    # Logistic Regression
    lr = LogisticRegression(**cfg['models']['logistic_regression'])
    lr.fit(X_tr, y_tr)
    y_pred = lr.predict(X_te)
    m, _ = eval_from_preds(y_te, y_pred, LABELS)
    proba = lr.predict_proba(X_te)
    mauprc, _ = macro_auprc_from_proba(y_te, proba, classes)
    results.append(dict(Model="Logistic Regression", **m, Macro_AUPRC=mauprc))

    # SVM (Linear)
    svm = SVC(**cfg['models']['svm_linear'])
    svm.fit(X_tr, y_tr)
    y_pred = svm.predict(X_te)
    m, _ = eval_from_preds(y_te, y_pred, LABELS)
    proba = svm.predict_proba(X_te)
    mauprc, _ = macro_auprc_from_proba(y_te, proba, classes)
    results.append(dict(Model="SVM (Linear)", **m, Macro_AUPRC=mauprc))

    # LightGBM (Monotonic)
    if HAVE_LGB:
        params = cfg['models']['lightgbm'].copy()
        use_mono = params.pop("use_monotone", True)
        if use_mono:
            # +1 for risk-upward features (HR, Temp, Sys, Dia), -1 for SpO2; scores mirror the same
            mono = [1,1,1,1,-1] + [1,1,1,1,-1]
            params['monotone_constraints'] = mono
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_model = lgb.train(params, lgb_train, num_boost_round=params.get("n_estimators", 200))
        proba = lgb_model.predict(X_te)
        y_pred = np.argmax(proba, axis=1)
        m, _ = eval_from_preds(y_te, y_pred, LABELS)
        mauprc, _ = macro_auprc_from_proba(y_te, np.asarray(proba), classes)
        results.append(dict(Model="LightGBM (Monotonic)" if use_mono else "LightGBM", **m, Macro_AUPRC=mauprc))

    # CatBoost
    if HAVE_CAT:
        cat_params = cfg['models']['catboost'].copy()
        cat = CatBoostClassifier(**cat_params)
        cat.fit(X_tr, y_tr)
        proba = cat.predict_proba(X_te)
        y_pred = np.argmax(proba, axis=1)
        m, _ = eval_from_preds(y_te, y_pred, LABELS)
        mauprc, _ = macro_auprc_from_proba(y_te, np.asarray(proba), classes)
        results.append(dict(Model="CatBoost", **m, Macro_AUPRC=mauprc))

    # Ordinal Logistic Regression (optional)
    if HAVE_ORD and cfg['models'].get('ordinal_logit', {}).get('use', False):
        ord_model = OrderedModel(y_tr, X_tr, distr='logit')
        ord_res = ord_model.fit(method='bfgs', disp=False)
        proba = ord_res.predict(X_te).values
        y_pred = np.argmax(proba, axis=1)
        m, _ = eval_from_preds(y_te, y_pred, LABELS)
        mauprc, _ = macro_auprc_from_proba(y_te, proba, classes)
        results.append(dict(Model="Ordinal Logistic Regression", **m, Macro_AUPRC=mauprc))

    # Naïve baseline
    def rule_based(row):
        if row['SpO2'] < 95: return 3
        elif row['SystolicBP'] > 139 or row['HeartRate'] > 100: return 2
        elif row['SystolicBP'] < 90 or row['HeartRate'] < 60:  return 1
        else: return 0
    y_pred_rule = test.apply(rule_based, axis=1).values
    proba_rule = np.zeros((len(y_pred_rule), 4), dtype=float)
    proba_rule[np.arange(len(y_pred_rule)), y_pred_rule] = 1.0
    from .evaluation import eval_from_preds
    m, _ = eval_from_preds(y_te, y_pred_rule, LABELS)
    from .evaluation import macro_auprc_from_proba
    mauprc, _ = macro_auprc_from_proba(y_te, proba_rule, classes)
    results.append(dict(Model="Naïve Baseline", **m, Macro_AUPRC=mauprc))

    res_df = pd.DataFrame(results)
    for c in ['Accuracy','Macro_F1','Extreme_Recall','Macro_AUPRC']:
        res_df[c] = res_df[c].astype(float).round(3)

    # Save table and fig
    ensure_dirs(paths['tables_dir'], paths['figures_dir'])
    save_table(res_df, paths['tables_dir'] + "table11_model_comparison.csv")

    # Representative PRC (use XGBoost proba)
    plot_pr_curves_by_class(y_te, xgb_model.predict_proba(X_te), LABELS, paths['figures_dir'] + "figure6_prc_by_class.png")

    return res_df
