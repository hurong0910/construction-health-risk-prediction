# Sample Dataset for Clinically-Guided Machine Learning Framework

This folder contains a **de-identified sample dataset** derived from wearable signals collected from construction workers.  
The dataset is provided to support **reproducibility of the results** presented in the following paper:

**Hu, R., et al. (2025). _A Clinically-Guided Machine Learning Framework for Predicting Health Risk in Construction Workers Using Wearable Data_. IEEE Access (under review).**

---

## 📊 Dataset Description

- **File**: `sample_dataset.csv`  
- **Size**: Subset of the full dataset (approx. X records, Y features)  
- **Variables included**:
  - `Timestamp`: Data collection time (UTC)  
  - `HeartRate`: Heart rate (bpm)  
  - `BodyTemp`: Body temperature (°C)  
  - `SystolicBP`: Systolic blood pressure (mmHg)  
  - `DiastolicBP`: Diastolic blood pressure (mmHg)  
  - `SpO2`: Blood oxygen saturation (%)  
  - `StepCount`: Step count (per interval)  
  - `Battery`: Device battery level (%)  
  - `Longitude`, `Latitude`: GPS coordinates (de-identified, noise-added)  

⚠️ All personally identifiable information (PII) has been removed. Spatial features are anonymized to prevent re-identification.

---

## 📜 License and Usage

- This dataset is released under the **CC BY-NC 4.0 License**.  
- **Academic and non-commercial use only**.  
- When using this dataset, you **must cite** the following reference:

```
Hu, R., et al. (2025).
A Clinically-Guided Machine Learning Framework for Predicting Health Risk in Construction Workers Using Wearable Data.
IEEE Access (under review).
```

---

## 📥 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/hurong0910/construction-health-risk-prediction.git
   ```
2. Navigate to the `data/` folder.
3. Load the dataset in Python:
   ```python
   import pandas as pd
   df = pd.read_csv("data/sample_dataset.csv")
   print(df.head())
   ```

---

## 🔗 Citation Reminder

If you use this dataset in your work (research papers, theses, presentations, or derivative datasets), please cite the above reference and provide a link to this repository.

---
