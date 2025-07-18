import pandas as pd

# 1. 明确列出所有要合并的 CSV 路径
file_paths = [
    r"C:\Users\Administrator\SalePrediction\data\手环数据1.csv",
    r"C:\Users\Administrator\SalePrediction\data\手环数据2.csv",
    r"C:\Users\Administrator\SalePrediction\data\手环数据3.csv",
    r"C:\Users\Administrator\SalePrediction\data\手环数据4.csv",
    r"C:\Users\Administrator\SalePrediction\data\手环数据5.csv",
]

df_list = []
for fp in file_paths:
    try:
        tmp = pd.read_csv(fp, encoding='utf-8')
        print(f"Read {fp} with utf-8")
    except Exception:
        tmp = pd.read_csv(fp, encoding='gbk')
        print(f"Read {fp} with gbk")
    df_list.append(tmp)

# 2. 合并
df = pd.concat(df_list, ignore_index=True)

# 3. 删除无关列
cols_to_drop = [
    '标段名称','人员名称','用户联系方式','IMEI','ICCID',
    '上传频率(分钟)','定位类型','求助信息','预警信息','预警信息.1',
    '手环是否拆除','是否预警','备注','所属班组','体表温度'
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')

# 4. 重命名中英文列为英文
rename_map = {
    'ID':'RecordID','所属标段ID':'SectionID','人员ID':'PersonnelID',
    '经度':'Longitude','纬度':'Latitude','记录时间':'Timestamp',
    '身体温度':'BodyTemp','心率':'HeartRate','收缩压':'SystolicBP',
    '舒张压':'DiastolicBP','血氧':'SpO2','电量':'Battery','步数':'StepCount'
}
df.rename(columns=rename_map, inplace=True)

# 5. 检查一下此时的列名，确保核心列存在
print("Columns after rename:", df.columns.tolist())

# 6. 核心生理特征缺失剔除
core = ['HeartRate','BodyTemp','SystolicBP','DiastolicBP','SpO2']
df.dropna(subset=core, inplace=True)
df.drop(columns=['国际移动设备识别码IMEI', '集成电路卡识别码ICCID'], inplace=True, errors='ignore')
# 7. 非核心数值列（Battery, StepCount, Latitude, Longitude）用 0 填充
for col in ['Battery','StepCount','Latitude','Longitude']:
    if col in df.columns:
        df[col].fillna(0, inplace=True)

# 8. （可选）异常值剔除：IQR 法
for col in core + ['Battery','StepCount']:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[df[col].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)]
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
print("Invalid timestamps:", df['Timestamp'].isnull().sum())
# 如果有少量无法解析，可用前向/后向填充：
df['Timestamp'].fillna(method='ffill', inplace=True)

# 9. 数据概览
print(df.info())
print(df.describe())
import matplotlib.pyplot as plt
import seaborn as sns

# 3.1 直方图
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
cols = ['HeartRate','BodyTemp','SystolicBP','DiastolicBP','SpO2','Battery','StepCount']
for ax, col in zip(axes.flatten(), cols):
    sns.histplot(df[col], bins=30, kde=True, ax=ax)
    ax.set_title(f'{col} Distribution')
plt.tight_layout()
plt.savefig('feature_histograms.png', dpi=300)

# 3.2 相关性热力图
plt.figure(figsize=(8,6))
corr = df[cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.savefig('feature_correlation_heatmap.png', dpi=300)

# 10. 保存清洗后数据
output_path = r"C:\Users\Administrator\SalePrediction\data\processed_handband_data.csv"
df.to_csv(output_path, index=False)
print("Processed data saved to:", output_path)
