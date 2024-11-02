import numpy as np
import pywt
import pandas as pd

def extract_wavelet_energy_features(signal, wavelet='db4', max_level=3):  # db4 commonly used for EEG signals
    """
    Extracts wavelet energy features from a 2D numpy array signal.
    Parameters:
    signal (numpy.ndarray): A 2D numpy array where each row represents a signal segment.
    wavelet (str): The type of wavelet to use for decomposition. Default is 'db4'.
    max_level (int): The maximum level of wavelet decomposition. Default is 3.
    Returns:
    pandas.DataFrame: A DataFrame containing the wavelet energy features for each segment.
                        Each column corresponds to the energy of a specific sub-band.
    """
    
    features = []

    for segment in signal:
        wp = pywt.WaveletPacket(data=segment, wavelet=wavelet, maxlevel=max_level)
        feature_vector = []
        for node in wp.get_level(max_level, 'freq'):
            # Calculate energy of each node
            energy = np.sum(np.square(node.data))
            feature_vector.append(energy)
        features.append(feature_vector)

    # Name the features according to the sub-band
    feature_names = [f'energy_band_{i}' for i in range(len(features[0]))]
    df_features = pd.DataFrame(features, columns=feature_names)
    
    return df_features

def extract_wavelet_energy_features_multichannel(signal, wavelet='db4', max_level=3):
    """
    Extracts wavelet energy features from a 3D numpy array signal.
    Parameters:
    signal (numpy.ndarray): A 3D numpy array with shape (channels, windows, 500).
    wavelet (str): The type of wavelet to use for decomposition. Default is 'db4'.
    max_level (int): The maximum level of wavelet decomposition. Default is 3.
    Returns:
    dict: A dictionary where keys are feature names and values are numpy arrays of shape (channels, windows).
    """
    
    channels, windows, _ = signal.shape
    features_dict = {}

    for ch in range(channels):
        channel_features = []
        for win in range(windows):
            segment = signal[ch, win, :]
            wp = pywt.WaveletPacket(data=segment, wavelet=wavelet, maxlevel=max_level)
            feature_vector = []
            for node in wp.get_level(max_level, 'freq'):
                # Calculate energy of each node
                energy = np.sum(np.square(node.data))
                feature_vector.append(energy)
            channel_features.append(feature_vector)
        
        # Convert list of features to numpy array and store in dictionary
        feature_names = [f'energy_band_{i}' for i in range(len(channel_features[0]))]
        for i, feature_name in enumerate(feature_names):
            if feature_name not in features_dict:
                features_dict[feature_name] = np.zeros((channels, windows))
            features_dict[feature_name][ch, :] = np.array([cf[i] for cf in channel_features])
    
    return features_dict