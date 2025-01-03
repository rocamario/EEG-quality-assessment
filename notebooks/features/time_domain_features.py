import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import hilbert

# Extract time domain features
def extract_time_domain_features(data, return_type='dataframe'):
    """
    Extracts time-domain features from EEG data.
    Parameters:
    data (numpy.ndarray): A 2D or 3D array where each row (or each slice in the case of 3D) represents a 2-second window of EEG data sampled at 250 Hz (i.e., each row has 500 data points).
    return_type (str): The type of the return value, either 'dataframe' or 'numpy'.
    Returns:
    pandas.DataFrame or numpy.ndarray: A DataFrame or ndarray containing the following time-domain features for each row (or slice) of the input data:
        - amplitude: The difference between the maximum and minimum values.
        - mean: The mean value.
        - max: The maximum value.
        - min: The minimum value.
        - stdev: The standard deviation.
        - skewness: The skewness of the data.
        - kurtosis: The kurtosis of the data.
        - hjorth_activity: The Hjorth activity parameter.
        - hjorth_mobility: The Hjorth mobility parameter.
        - hjorth_complexity: The Hjorth complexity parameter.
    """

    is_3d = data.ndim == 3

    amplitude = np.max(data, axis=-1) - np.min(data, axis=-1)
    mean_values = np.mean(data, axis=-1)
    max_values = np.max(data, axis=-1)
    min_values = np.min(data, axis=-1)
    stdev_values = np.std(data, axis=-1)
    skewness_values = skew(data, axis=-1)
    kurtosis_values = kurtosis(data, axis=-1)

    # Hjorth parameters
    def hjorth_parameters(data):
        first_deriv = np.diff(data, axis=-1)
        second_deriv = np.diff(first_deriv, axis=-1)
        var_zero = np.var(data, axis=-1)
        var_d1 = np.var(first_deriv, axis=-1)
        var_d2 = np.var(second_deriv, axis=-1)
        activity = var_zero
        mobility = np.sqrt(var_d1 / var_zero)
        complexity = np.sqrt(var_d2 / var_d1) / mobility
        return activity, mobility, complexity

    hjorth_activity, hjorth_mobility, hjorth_complexity = hjorth_parameters(data)

    features = {
        "amplitude": amplitude,
        "mean": mean_values,
        "max": max_values,
        "min": min_values,
        "stdev": stdev_values,
        "skewness": skewness_values,
        "kurtosis": kurtosis_values,
        "hjorth_activity": hjorth_activity,
        "hjorth_mobility": hjorth_mobility,
        "hjorth_complexity": hjorth_complexity,
    }

    if not is_3d:
        for key in features:
            features[key] = features[key].reshape(-1)

    if return_type == 'dataframe':
        return pd.DataFrame(features)
    elif return_type == 'numpy':
        return features
    else:
       raise ValueError("return_type must be either 'dataframe' or 'numpy'")
    

def extract_amplitude_modulation_features(data, return_type='dataframe'):
    """
    Extracts amplitude modulation features from EEG data using the Hilbert transform.
    Parameters:
    data (numpy.ndarray): A 2D or 3D array where each row (or each slice in the case of 3D) represents a 2-second window of EEG data sampled at 250 Hz (i.e., each row has 500 data points).
    return_type (str): The type of the return value, either 'dataframe' or 'numpy'.
    Returns:
    pandas.DataFrame or numpy.ndarray: A DataFrame or ndarray containing the following amplitude modulation features for each row (or slice) of the input data:
        - envelope_mean: The mean of the amplitude envelope.
        - envelope_std: The standard deviation of the amplitude envelope.
        - envelope_max: The maximum value of the amplitude envelope.
        - envelope_min: The minimum value of the amplitude envelope.
    """

    is_3d = data.ndim == 3

    analytic_signal = hilbert(data, axis=-1)
    amplitude_envelope = np.abs(analytic_signal)

    envelope_mean = np.mean(amplitude_envelope, axis=-1)
    envelope_std = np.std(amplitude_envelope, axis=-1)
    envelope_max = np.max(amplitude_envelope, axis=-1)
    envelope_min = np.min(amplitude_envelope, axis=-1)

    features = {
        "envelope_mean": envelope_mean,
        "envelope_std": envelope_std,
        "envelope_max": envelope_max,
        "envelope_min": envelope_min,
    }

    if not is_3d:
        for key in features:
            features[key] = features[key].reshape(-1)

    if return_type == 'dataframe':
        return pd.DataFrame(features)
    elif return_type == 'numpy':
        return features
    else:
        raise ValueError("return_type must be either 'dataframe' or 'numpy'")