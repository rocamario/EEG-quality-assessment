import pandas as pd
import numpy as np
from scipy.signal import welch

# Extract frequency domain features
def extract_frequency_domain_features(data, fs=250):
    """
    Extracts frequency domain features from EEG signal data.
    Parameters:
    data (np.ndarray): A NumPy array where each row represents a 2-second window of the EEG signal, 
                       with each row containing 500 data points.
    fs (int): Sampling frequency of the EEG signal. Default is 250 Hz.
    Returns:
    pd.DataFrame: A DataFrame containing the extracted frequency domain features.
    """

    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 40)
    }

    features = []

    for window in data:
        f, Pxx = welch(window, fs=fs, nperseg=fs*2)
        band_powers = {}
        for band, (low, high) in bands.items():
            band_power = np.trapz(Pxx[(f >= low) & (f <= high)], f[(f >= low) & (f <= high)])
            band_powers[f'{band}_power'] = band_power
        features.append(band_powers)

    return pd.DataFrame(features)

def extract_relative_power(data, fs=250):
    """
    Extracts relative power features from EEG signal data.
    Parameters:
    data (np.ndarray): A NumPy array where each row represents a 2-second window of the EEG signal, 
                        with each row containing 500 data points.
    fs (int): Sampling frequency of the EEG signal. Default is 250 Hz.
    Returns:
    pd.DataFrame: A DataFrame containing the extracted relative power features.
    """

    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 40)
    }

    features = []

    for window in data:
        f, Pxx = welch(window, fs=fs, nperseg=fs*2)
        total_power = np.trapz(Pxx, f)
        band_powers = {}
        for band, (low, high) in bands.items():
            band_power = np.trapz(Pxx[(f >= low) & (f <= high)], f[(f >= low) & (f <= high)])
            band_powers[f'{band}_relative_power'] = band_power / total_power
        features.append(band_powers)

    return pd.DataFrame(features)

def extract_frequency_domain_features_multichannel(data, fs=250):
    """
    Extracts absolute frequency domain features from multi-channel EEG signal data.
    Parameters:
    data (np.ndarray): A NumPy array with shape (channel, segment, 500), where each segment represents a 2-second window of the EEG signal.
    fs (int): Sampling frequency of the EEG signal. Default is 250 Hz.
    Returns:
    dict: A dictionary containing the extracted frequency domain features with keys as {band}_power and values as numpy arrays of shape (channel, segment).
    """

    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 40)
    }

    num_channels, num_segments, _ = data.shape
    features = {f'{band}_power': np.zeros((num_channels, num_segments)) for band in bands}

    for ch in range(num_channels):
        for seg in range(num_segments):
            window = data[ch, seg, :]
            f, Pxx = welch(window, fs=fs, nperseg=fs*2)
            for band, (low, high) in bands.items():
                band_power = np.trapz(Pxx[(f >= low) & (f <= high)], f[(f >= low) & (f <= high)])
                features[f'{band}_power'][ch, seg] = band_power

    return features

def extract_relative_power_multichannel(data, fs=250):
    """
    Extracts relative power features from multi-channel EEG signal data.
    Parameters:
    data (np.ndarray): A NumPy array with shape (channel, segment, 500), where each segment represents a 2-second window of the EEG signal.
    fs (int): Sampling frequency of the EEG signal. Default is 250 Hz.
    Returns:
    dict: A dictionary containing the extracted relative power features with keys as {band}_relative_power and values as numpy arrays of shape (channel, segment).
    """

    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 40)
    }

    num_channels, num_segments, _ = data.shape
    relative_power_features = {f'{band}_relative_power': np.zeros((num_channels, num_segments)) for band in bands}

    for ch in range(num_channels):
        for seg in range(num_segments):
            window = data[ch, seg, :]
            f, Pxx = welch(window, fs=fs, nperseg=fs*2)
            total_power = np.trapz(Pxx, f)
            for band, (low, high) in bands.items():
                band_power = np.trapz(Pxx[(f >= low) & (f <= high)], f[(f >= low) & (f <= high)])
                relative_power_features[f'{band}_relative_power'][ch, seg] = band_power / total_power

    return relative_power_features