

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb

# Load data
df = pd.read_csv(r"C:\Users\Administrator\SalePrediction\data\processed_handband_data.csv")
df = df.dropna(subset=['PersonnelID']).reset_index(drop=True)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.sort_values(['PersonnelID','Timestamp'], inplace=True)

# Clinical scoring
def score_hr(x):   return 0 if 60 <= x <= 100 else (1 if 40 <= x < 60 else 2)
def score_temp(x): return 0 if 36 <= x <= 37 else (1 if 35 <= x < 36 or 37 < x <= 38 else 2)
def score_sys(x):  return 0 if 90 <= x <= 119 else (1 if 120 <= x <= 139 else 2)
def score_dia(x):  return 0 if 60 < x <= 85 else (1 if 85 < x <= 89 else 2)
def score_spo2(x): return 0 if x >= 95 else 2

df['HR_score']   = df['HeartRate'].apply(score_hr)
df['Temp_score'] = df['BodyTemp'].apply(score_temp)
df['Sys_score']  = df['SystolicBP'].apply(score_sys)
df['Dia_score']  = df['DiastolicBP'].apply(score_dia)
df['SpO2_score'] = df['SpO2'].apply(score_spo2)
score_cols = ['HR_score', 'Temp_score', 'Sys_score', 'Dia_score', 'SpO2_score']

# WeightedRisk
A = np.array([
    [1, 1/2, 1,   2,   3],
    [2, 1,   2,   3,   4],
    [1, 1/2, 1,   2,   2],
    [1/2,1/3,1/2,1,   2],
    [1/3,1/4,1/2,1/2,1]
])
w = (A.prod(axis=1)**(1/5))
w /= w.sum()
df['WeightedRisk'] = df[score_cols].dot(w)

# Next-step target
df['NextWeightedRisk'] = df.groupby('PersonnelID')['WeightedRisk'].shift(-1)
df = df.dropna(subset=['NextWeightedRisk']).reset_index(drop=True)

# Quartile binning with qcut
ranks = df['NextWeightedRisk'].rank(method='first')
df['NextRiskLevel'] = pd.qcut(
    ranks,
    4,
    labels=['Low','Medium','High','Extreme']
)
df['NextRiskCode'] = df['NextRiskLevel'].cat.codes

print("NextRiskLevel distribution:\n", df['NextRiskLevel'].value_counts())

# Train/test split
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['PersonnelID']))
train = df.iloc[train_idx].reset_index(drop=True)
test  = df.iloc[test_idx].reset_index(drop=True)

# Features and labels
cont_cols = ['HeartRate', 'BodyTemp', 'SystolicBP', 'DiastolicBP', 'SpO2']
features = cont_cols + score_cols
X_tr, y_tr = train[features], train['NextRiskCode']
X_te, y_te = test [features], test ['NextRiskCode']

# Train XGBoost
model = xgb.XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss', use_label_encoder=False)
model.fit(X_tr, y_tr)

# Confusion matrix
y_pred = model.predict(X_te)
cm = confusion_matrix(y_te, y_pred, labels=[0,1,2,3])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low','Medium','High','Extreme'],
            yticklabels=['Low','Medium','High','Extreme'])
plt.title('Next-Step Risk Confusion Matrix')
plt.tight_layout()
plt.savefig('cm_nextstep_fixed.png', dpi=300)
plt.close()

# Classification report
print(classification_report(y_te, y_pred, labels=[0,1,2,3],
    target_names=['Low','Medium','High','Extreme'], zero_division=0))

# ROC curves
y_score = model.predict_proba(X_te)
plt.figure(figsize=(6,4))
for cls in range(4):
    fpr, tpr, _ = roc_curve((y_te==cls).astype(int), y_score[:,cls])
    plt.plot(fpr, tpr, label=f"{['Low','Medium','High','Extreme'][cls]} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('Next-Step Risk ROC Curves')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_nextstep_fixed.png', dpi=300)
plt.close()

print("Next-step prediction with qcut binning complete.")
