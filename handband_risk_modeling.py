import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

# 1. 数据加载和排序
df = pd.read_csv(r"C:\Users\Administrator\SalePrediction\data\processed_handband_data.csv")
df = df.dropna(subset=['PersonnelID']).reset_index(drop=True)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.sort_values(['PersonnelID','Timestamp'], inplace=True)

# 2. 临床打分
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

# 3. AHP权重
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

# 4. 下一步风险（目标标签）
df['NextWeightedRisk'] = df.groupby('PersonnelID')['WeightedRisk'].shift(-1)
df = df.dropna(subset=['NextWeightedRisk']).reset_index(drop=True)

# 5. 四分位分层/编码
ranks = df['NextWeightedRisk'].rank(method='first')
df['NextRiskLevel'] = pd.qcut(
    ranks,
    4,
    labels=['Low','Medium','High','Extreme']
)
df['NextRiskCode'] = df['NextRiskLevel'].cat.codes
print("NextRiskLevel distribution:\n", df['NextRiskLevel'].value_counts())

# 6. 训练/测试划分
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['PersonnelID']))
train = df.iloc[train_idx].reset_index(drop=True)
test  = df.iloc[test_idx].reset_index(drop=True)

# 7. 特征和标签
features = cont_cols + score_cols
X_tr, y_tr = train[features], train['NextRiskCode']
X_te, y_te = test [features], test ['NextRiskCode']
labels = ['Low','Medium','High','Extreme']

# 8. 多模型对比
models = {
    "XGBoost": xgb.XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss', use_label_encoder=False, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000, multi_class='multinomial', random_state=42),
    "SVM (Linear)": SVC(kernel='linear', probability=True, random_state=42)
}
results = []

for name, clf in models.items():
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    report = classification_report(y_te, y_pred, labels=[0,1,2,3], target_names=labels, output_dict=True, zero_division=0)
    acc = accuracy_score(y_te, y_pred)
    macro_f1 = report['macro avg']['f1-score']
    extreme_rec = report['Extreme']['recall']
    results.append(dict(Model=name, Accuracy=acc, Macro_F1=macro_f1, Extreme_Recall=extreme_rec))
    print(f"\n=== {name} ===")
    print(classification_report(y_te, y_pred, labels=[0,1,2,3], target_names=labels, zero_division=0))

# 9. 对比表格与可视化
df_res = pd.DataFrame(results)
print("\nModel Comparison Table:")
print(df_res.to_string(index=False))
df_res.to_csv("model_comparison.csv", index=False)

plt.figure(figsize=(8,5))
bar_width = 0.22
x = np.arange(len(df_res))
plt.bar(x - bar_width, df_res['Accuracy'], width=bar_width, label='Accuracy')
plt.bar(x, df_res['Macro_F1'], width=bar_width, label='Macro F1')
plt.bar(x + bar_width, df_res['Extreme_Recall'], width=bar_width, label='Extreme Recall')
plt.xticks(x, df_res['Model'])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("model_performance_comparison.png", dpi=150)
plt.close()

print("All model training and comparison finished.")
