import pandas as pd

df = pd.read_csv('processed_handband_data.csv')

summary = []

summary.append("===== Dataset Summary =====")
summary.append(f"Total samples: {len(df)}")
summary.append(f"Unique personnel: {df['PersonnelID'].nunique()}")
summary.append(f"Time range: {df['Timestamp'].min()} ~ {df['Timestamp'].max()}")

summary.append("\n===== Descriptive Statistics =====")
desc_stats = df[['HeartRate', 'BodyTemp', 'SystolicBP', 'DiastolicBP', 'SpO2']].describe().round(2)
summary.append(str(desc_stats))

summary.append("\n===== Missing Values =====")
missing = df[['HeartRate', 'BodyTemp', 'SystolicBP', 'DiastolicBP', 'SpO2']].isnull().sum()
summary.append(str(missing))

summary.append("\n===== Main Model Risk Distribution (Test Set) =====")
try:
    risk_dist = test['NextRiskLevel'].value_counts()
    summary.append(str(risk_dist))
except:
    summary.append("Test set risk level not calculated yet.")

summary.append("\n===== Personnel Sample Count (Top 5) =====")
worker_count = df['PersonnelID'].value_counts()
summary.append(str(worker_count.head()))

# 汇总输出
output = "\n".join(summary)
print(output)
with open("dataset_descriptive_summary.txt", "w") as f:
    f.write(output)
