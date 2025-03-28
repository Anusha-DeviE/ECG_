import wfdb
import numpy as np

# Define two-class labels
ST_DEPRESSION = list(range(300, 323))  # Class 0
ST_ELEVATION = list(range(323, 328))   # Class 1

def load_ecg_data(record_number):
    record_path = f"data/{record_number}"
    signals, fields = wfdb.rdsamp(record_path)
    
    # Assign class labels
    if record_number in ST_DEPRESSION:
        label = 0  # ST Depression
    elif record_number in ST_ELEVATION:
        label = 1  # ST Elevation
    else:
        label = -1  # Ignore other cases

    return signals, label
