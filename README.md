# ECG Signal Classification for ST-Segment Abnormalities

This project focuses on analyzing ECG (Electrocardiogram) and applying machine learning techniques to classify **ST-segment abnormalities**.

The task is framed as a **binary classification problem**:
- **Class 0:** ST Depression  
- **Class 1:** ST Elevation  

These conditions are clinically relevant indicators of cardiac ischemia and myocardial infarction.

---

## Project Overview

The pipeline follows a standard biomedical signal processing workflow:

1. Load ECG signals from annotated records using the WFDB library  
2. Preprocess raw ECG signals to remove noise and inconsistencies  
3. Extract meaningful features from ECG time-series data  
4. Train and evaluate machine learning models for classification  
5. Analyze and store model performance metrics  

---

## Dataset and Labels

- ECG records are loaded using `wfdb.rdsamp`
- Labels are assigned based on record identifiers corresponding to known conditions:
  - ST Depression → Label `0`
  - ST Elevation → Label `1`
- Records outside the selected ranges are ignored
