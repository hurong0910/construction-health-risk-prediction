
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ================= 工具函数：鲁棒分箱 =================
def robust_bins(series, labels):
    n_labels = len(labels)
    # 先尝试qcut
    try:
        result = pd.qcut(series, n_labels, labels=labels, duplicates='drop')
        actual_bins = result.unique().size
        if actual_bins < n_labels:
            raise Exception("qcut箱数少于label数")
        return result
    except Exception as e:
        print(f"qcut分箱失败或箱数不足，改用等距cut: {e}")
        bins = np.linspace(series.min(), series.max(), n_labels+1)
        bins = np.unique(bins)
        if bins.size <= 2:  # 只有一个bin或全是相同值
            return pd.Series([labels[0]] * len(series), index=series.index, dtype='category')
        n_actual_bins = len(bins) - 1
        use_labels = labels[:n_actual_bins]
        return pd.cut(series, bins=bins, labels=use_labels, include_lowest=True)

# ================ 结果评价函数 ======================
def eval_and_print(y_true, y_pred, note, labellist):
    print(f"\n==== {note} ====")
    report = classification_report(y_true, y_pred, labels=list(range(len(labellist))), target_names=labellist, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, labels=list(range(len(labellist))), target_names=labellist, zero_division=0))
    macro_f1 = report['macro avg']['f1-score']
    acc = accuracy_score(y_true, y_pred)
    extreme_rec = report[labellist[-1]]['recall']
    return acc, macro_f1, extreme_rec

# ================= 数据准备 =========================
df = pd.read_csv(r"C:\Users\Administrator\SalePrediction\data\processed_handband_data.csv")
df = df.dropna(subset=['PersonnelID']).reset_index(drop=True)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.sort_values(['PersonnelID','Timestamp'], inplace=True)

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
cont_cols = ['HeartRate', 'BodyTemp', 'SystolicBP', 'DiastolicBP', 'SpO2']

A = np.array([
    [1, 1/2, 1,   2,   3],
    [2, 1,   2,   3,   4],
    [1, 1/2, 1,   2,   2],
    [1/2,1/3,1/2,1,   2],
    [1/3,1/4,1/2,1/2,1]
])
w = (A.prod(axis=1)**(1/5))
w /= w.sum()

labels = ['Low','Medium','High','Extreme']
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
results = []

# ============ 1. 全流程主模型 ============
df['WeightedRisk'] = df[score_cols].dot(w)
df['NextWeightedRisk'] = df.groupby('PersonnelID')['WeightedRisk'].shift(-1)
df_main = df.dropna(subset=['NextWeightedRisk']).reset_index(drop=True)
df_main['NextRiskLevel'] = robust_bins(df_main['NextWeightedRisk'], labels)
df_main['NextRiskCode'] = df_main['NextRiskLevel'].cat.codes
print("主模型risk分布：\n", df_main['NextRiskLevel'].value_counts())

features = cont_cols + score_cols
train_idx, test_idx = next(gss.split(df_main, groups=df_main['PersonnelID']))
train = df_main.iloc[train_idx].reset_index(drop=True)
test  = df_main.iloc[test_idx].reset_index(drop=True)
X_tr, y_tr = train[features], train['NextRiskCode']
X_te, y_te = test [features], test ['NextRiskCode']

model = xgb.XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss', random_state=42)
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)
acc, macro_f1, extreme_rec = eval_and_print(y_te, y_pred, '全流程主模型 (AHP+分级+聚类+下一步预测+XGB)', labels)
results.append(dict(setting='主模型', acc=acc, macro_f1=macro_f1, extreme_rec=extreme_rec))

# ============ 2. AHP等权消融 ============
w_eq = np.ones(5)/5
df['WeightedRisk_eq'] = df[score_cols].dot(w_eq)
df['NextWeightedRisk_eq'] = df.groupby('PersonnelID')['WeightedRisk_eq'].shift(-1)
df_eq = df.dropna(subset=['NextWeightedRisk_eq']).reset_index(drop=True)
df_eq['NextRiskLevel_eq'] = robust_bins(df_eq['NextWeightedRisk_eq'], labels)
df_eq['NextRiskCode_eq'] = df_eq['NextRiskLevel_eq'].cat.codes
print("AHP等权分布：\n", df_eq['NextRiskLevel_eq'].value_counts())

train_idx, test_idx = next(gss.split(df_eq, groups=df_eq['PersonnelID']))
train_eq = df_eq.iloc[train_idx].reset_index(drop=True)
test_eq  = df_eq.iloc[test_idx].reset_index(drop=True)
X_tr_eq, y_tr_eq = train_eq[features], train_eq['NextRiskCode_eq']
X_te_eq, y_te_eq = test_eq [features], test_eq ['NextRiskCode_eq']

model_eq = xgb.XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss', random_state=42)
model_eq.fit(X_tr_eq, y_tr_eq)
y_pred_eq = model_eq.predict(X_te_eq)
acc, macro_f1, extreme_rec = eval_and_print(y_te_eq, y_pred_eq, 'AHP等权消融 (AHP等权)', labels)
results.append(dict(setting='AHP等权', acc=acc, macro_f1=macro_f1, extreme_rec=extreme_rec))

# ============ 3. 无临床分级消融 ============
df['WeightedRisk_raw'] = df[cont_cols].dot(w[:5])
df['NextWeightedRisk_raw'] = df.groupby('PersonnelID')['WeightedRisk_raw'].shift(-1)
df_raw = df.dropna(subset=['NextWeightedRisk_raw']).reset_index(drop=True)
df_raw['NextRiskLevel_raw'] = robust_bins(df_raw['NextWeightedRisk_raw'], labels)
df_raw['NextRiskCode_raw'] = df_raw['NextRiskLevel_raw'].cat.codes
print("无分级分布：\n", df_raw['NextRiskLevel_raw'].value_counts())

train_idx, test_idx = next(gss.split(df_raw, groups=df_raw['PersonnelID']))
train_raw = df_raw.iloc[train_idx].reset_index(drop=True)
test_raw  = df_raw.iloc[test_idx].reset_index(drop=True)
X_tr_raw, y_tr_raw = train_raw[cont_cols], train_raw['NextRiskCode_raw']
X_te_raw, y_te_raw = test_raw [cont_cols], test_raw ['NextRiskCode_raw']

model_raw = xgb.XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss', random_state=42)
model_raw.fit(X_tr_raw, y_tr_raw)
y_pred_raw = model_raw.predict(X_te_raw)
acc, macro_f1, extreme_rec = eval_and_print(y_te_raw, y_pred_raw, '无临床分级消融 (只用原始数据)', labels)
results.append(dict(setting='无临床分级', acc=acc, macro_f1=macro_f1, extreme_rec=extreme_rec))

# ============ 4. 无聚类分层消融 ============
df['CompositeScore'] = df[score_cols].sum(axis=1)
df['NextCompositeScore'] = df.groupby('PersonnelID')['CompositeScore'].shift(-1)
df_sum = df.dropna(subset=['NextCompositeScore']).reset_index(drop=True)
bins = np.linspace(df_sum['NextCompositeScore'].min(), df_sum['NextCompositeScore'].max(), 5)
bins = np.unique(bins)
use_labels = labels[:len(bins)-1]
df_sum['NextRiskLevel_sum'] = pd.cut(df_sum['NextCompositeScore'], bins=bins, labels=use_labels, include_lowest=True)
df_sum['NextRiskCode_sum'] = df_sum['NextRiskLevel_sum'].cat.codes
print("无聚类分层分布：\n", df_sum['NextRiskLevel_sum'].value_counts())

train_idx, test_idx = next(gss.split(df_sum, groups=df_sum['PersonnelID']))
train_sum = df_sum.iloc[train_idx].reset_index(drop=True)
test_sum  = df_sum.iloc[test_idx].reset_index(drop=True)
X_tr_sum, y_tr_sum = train_sum[features], train_sum['NextRiskCode_sum']
X_te_sum, y_te_sum = test_sum [features], test_sum ['NextRiskCode_sum']

model_sum = xgb.XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss', random_state=42)
model_sum.fit(X_tr_sum, y_tr_sum)
y_pred_sum = model_sum.predict(X_te_sum)
acc, macro_f1, extreme_rec = eval_and_print(y_te_sum, y_pred_sum, '无聚类分层消融 (分级累加)', use_labels)
results.append(dict(setting='无聚类', acc=acc, macro_f1=macro_f1, extreme_rec=extreme_rec))

# ============ 5. 无下一步预测消融 ============
df['RiskLevel_now'] = robust_bins(df['WeightedRisk'], labels)
df['RiskCode_now'] = df['RiskLevel_now'].cat.codes
print("无下一步预测分布：\n", df['RiskLevel_now'].value_counts())

train_idx, test_idx = next(gss.split(df, groups=df['PersonnelID']))
train_now = df.iloc[train_idx].reset_index(drop=True)
test_now  = df.iloc[test_idx].reset_index(drop=True)
X_tr_now, y_tr_now = train_now[features], train_now['RiskCode_now']
X_te_now, y_te_now = test_now [features], test_now ['RiskCode_now']

model_now = xgb.XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss', random_state=42)
model_now.fit(X_tr_now, y_tr_now)
y_pred_now = model_now.predict(X_te_now)
acc, macro_f1, extreme_rec = eval_and_print(y_te_now, y_pred_now, '无下一步预测消融 (只用当前时刻)', labels)
results.append(dict(setting='无下一步预测', acc=acc, macro_f1=macro_f1, extreme_rec=extreme_rec))

# ============ 6. RF替换XGB消融 ============
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_tr, y_tr)
y_pred_rf = rf.predict(X_te)
acc, macro_f1, extreme_rec = eval_and_print(y_te, y_pred_rf, '模型替换消融 (RF替换XGB)', labels)
results.append(dict(setting='RF替换XGB', acc=acc, macro_f1=macro_f1, extreme_rec=extreme_rec))

# ========== 汇总输出 ============
results_df = pd.DataFrame(results)
print("\n==== 消融实验主要结果总表 ====")
print(results_df.to_string(index=False))
results_df.to_csv("ablation_summary.csv", index=False)
