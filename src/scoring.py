\
import numpy as np
import pandas as pd
from typing import List, Tuple

RAW2SCORE = {
    'HeartRate'  : ('HR_score',   lambda x: 0 if 60 <= x <= 100 else (1 if 40 <= x < 60 else 2)),
    'BodyTemp'   : ('Temp_score', lambda x: 0 if 36 <= x <= 37  else (1 if 35 <= x < 36 or 37 < x <= 38 else 2)),
    'SystolicBP' : ('Sys_score',  lambda x: 0 if 90 <= x <= 119 else (1 if 120 <= x <= 139 else 2)),
    'DiastolicBP': ('Dia_score',  lambda x: 0 if 60 <  x <= 85  else (1 if 85  < x <= 89 else 2)),
    'SpO2'       : ('SpO2_score', lambda x: 0 if x >= 95 else 2),
}

AHP_MATRIX = np.array([
    [1,   1/2, 1,   2,   3],
    [2,   1,   2,   3,   4],
    [1,   1/2, 1,   2,   2],
    [1/2, 1/3, 1/2, 1,   2],
    [1/3, 1/4, 1/2, 1/2, 1]
], dtype=float)

LABELS = ['Low','Medium','High','Extreme']

def add_clinical_scores(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RAW2SCORE if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for scoring: {missing}")
    for raw_col, (score_col, scorer) in RAW2SCORE.items():
        df[score_col] = df[raw_col].apply(scorer)
    return df

def compute_ahp_weights() -> np.ndarray:
    w = (AHP_MATRIX.prod(axis=1)**(1/5)); w /= w.sum()
    return w

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    score_cols = [v[0] for v in RAW2SCORE.values()]
    w = compute_ahp_weights()
    df['WeightedRisk']     = df[score_cols].dot(w)
    df['NextWeightedRisk'] = df.groupby('PersonnelID')['WeightedRisk'].shift(-1)
    df = df.dropna(subset=['NextWeightedRisk']).reset_index(drop=True)
    ranks = df['NextWeightedRisk'].rank(method='first')
    df['NextRiskLevel'] = pd.qcut(ranks, 4, labels=LABELS)
    df['NextRiskCode']  = df['NextRiskLevel'].cat.codes
    return df

def feature_lists() -> Tuple[list, list]:
    score_cols = [v[0] for v in RAW2SCORE.values()]
    cont_cols  = list(RAW2SCORE.keys())
    return cont_cols, score_cols
