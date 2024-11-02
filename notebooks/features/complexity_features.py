import antropy as ant
import pandas as pd
import numpy as np

def extract_entropy_features(signal):
    """
    Extracts entropy features from a 2D signal array.
    Parameters:
    signal (numpy.ndarray): A 2D array where each row represents a signal.
    Returns:
    pandas.DataFrame: A DataFrame containing the extracted entropy features:
        - 'shannon_entropy': List of Shannon entropy values for each signal.
        - 'sample_entropy': List of sample entropy values for each signal.
        - 'spectral_entropy': List of spectral entropy values for each signal.
    """
    
    features = {
        'shannon_entropy': [],
        'sample_entropy': [],
        'spectral_entropy': []
    }
    
    # Iterate over the last dimension
    for i in range(signal.shape[0]):
        features['shannon_entropy'].append(ant.perm_entropy(signal[i, ...]))
        features['sample_entropy'].append(ant.sample_entropy(signal[i, ...]))
        features['spectral_entropy'].append(ant.spectral_entropy(signal[i, ...], sf=250, method='welch', normalize=True))
    
    return pd.DataFrame(features)

def extract_multichannel_entropy_features(signal):
    """
    Extracts multichannel entropy features from a given EEG signal.
    Parameters:
    signal (numpy.ndarray): A 3D numpy array of shape (channels, windows, samples) representing the EEG signal.
    Returns:
    dict: A dictionary containing the following keys:
        - 'shannon_entropy': A 2D numpy array of shape (channels, windows) with Shannon entropy values.
        - 'sample_entropy': A 2D numpy array of shape (channels, windows) with Sample entropy values.
        - 'spectral_entropy': A 2D numpy array of shape (channels, windows) with Spectral entropy values.
    """

    channels, windows, _ = signal.shape
    
    features = {
        'shannon_entropy': np.zeros((channels, windows)),
        'sample_entropy': np.zeros((channels, windows)),
        'spectral_entropy': np.zeros((channels, windows))
    }
    
    for ch in range(channels):
        for win in range(windows):
            features['shannon_entropy'][ch, win] = ant.perm_entropy(signal[ch, win, :])
            features['sample_entropy'][ch, win] = ant.sample_entropy(signal[ch, win, :])
            features['spectral_entropy'][ch, win] = ant.spectral_entropy(signal[ch, win, :], sf=250, method='welch', normalize=True)
    
    return features