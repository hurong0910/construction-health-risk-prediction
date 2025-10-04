\
import pandas as pd
import numpy as np
from .utils import load_config

def load_and_sort(data_csv: str) -> pd.DataFrame:
    df = pd.read_csv(data_csv)
    if 'PersonnelID' not in df.columns:
        raise KeyError("Expected 'PersonnelID' column in data.")
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.sort_values(['PersonnelID','Timestamp']).reset_index(drop=True)
    return df

def fill_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # drop rows missing core signals
    core = ['HeartRate','BodyTemp','SystolicBP','DiastolicBP','SpO2']
    df = df.dropna(subset=[c for c in core if c in df.columns]).copy()
    for col in ['Battery','StepCount','Latitude','Longitude']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df
