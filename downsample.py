import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate

# Step 1: Load the ECG signal
record_name = "307"  # Modify with the correct record
record_path = f"data/{record_name}"  # Ensure the correct path

record = wfdb.rdrecord(record_path)  # Load ECG record
original_signal = record.p_signal[:, 0]  # Extract Lead 0 ECG signal
sampling_rate = record.fs  # Original sampling frequency

# Step 2: Downsample the signal (reduce by a factor of 2)
downsample_factor = 2  # Modify this based on the requirement
downsampled_signal = decimate(original_signal, downsample_factor, ftype='fir')
new_sampling_rate = sampling_rate // downsample_factor  # Update sampling frequency

 #Check if downsampling was done correctly
expected_length = len(original_signal) // downsample_factor
actual_length = len(downsampled_signal)
print(f"Expected Downsampled Length: {expected_length}, Actual: {actual_length}")

# Step 3: Create time axis
original_time = np.linspace(0, len(original_signal) / sampling_rate, len(original_signal))
downsampled_time = np.linspace(0, len(downsampled_signal) / new_sampling_rate, len(downsampled_signal))

# Step 4: Plot both signals for comparison
 #Plot both signals for comparison
plt.figure(figsize=(10, 5))
plt.plot(original_time[:1000], original_signal[:1000], label="Original ECG (360 Hz)", color='b')

# Fix slicing to match available points in downsampled signal
plt.plot(downsampled_time[:min(1000, len(downsampled_signal))], 
         downsampled_signal[:min(1000, len(downsampled_signal))], 
         label=f"Downsampled ECG ({new_sampling_rate} Hz)", color='r', linestyle='dashed')


plt.xlabel("Time (seconds)")
plt.ylabel("ECG Amplitude (mV)")
plt.title("Comparison of Original and Downsampled ECG")
plt.legend()
plt.show()

# Step 5: Print confirmation
print(f"Original Signal Length: {len(original_signal)}")
print(f"Downsampled Signal Length: {len(downsampled_signal)}")
print(f"New Sampling Rate: {new_sampling_rate} Hz")
