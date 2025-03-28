import numpy as np
import pandas as pd
from scipy.signal import welch

def extract_features(segment, label, fs=180):
    features = {}

    # Time-Domain Features
    features['mean'] = np.mean(segment)
    features['std_dev'] = np.std(segment)
    features['rms'] = np.sqrt(np.mean(segment**2))

    # Frequency-Domain Features (Power Spectral Density)
    freqs, psd = welch(segment, fs=fs, nperseg=max(2,len(segment)//2))
    features['max_psd'] = np.max(psd)
    features['mean_psd'] = np.mean(psd)

    # Add class label
    features['label'] = label

    return features
