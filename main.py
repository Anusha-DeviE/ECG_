import numpy as np
import pandas as pd
from preprocess import load_ecg_data
from feature_extract import extract_features

all_features = []

for record_num in range(300, 328):  # Loop through all records
    signals, label = load_ecg_data(record_num)

    if label == -1:
        continue  # Ignore records outside our class range

    fs = 360  # Original sampling rate
    segment_length = 10 * fs  # 10-second window
    num_segments = len(signals) // segment_length

    data_segments = np.array_split(signals[:num_segments * segment_length], num_segments)

    # Extract features for each segment
    for segment in data_segments:
        features = extract_features(segment, label, fs)
        all_features.append(features)

# Save to CSV
df_features = pd.DataFrame(all_features)
df_features.to_csv("ecg_features.csv", index=False)

print("Feature extraction completed. Features saved as 'ecg_features.csv'")
print(df_features.head())  




