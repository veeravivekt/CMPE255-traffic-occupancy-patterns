import numpy as np
from scipy import stats

def extract_features(data):
    # Time-based features
    morning_rush = np.mean(data[:, :, 30:54], axis=(1, 2))
    evening_rush = np.mean(data[:, :, 90:114], axis=(1, 2))
    mid_day = np.mean(data[:, :, 54:90], axis=(1, 2))
    night = np.mean(data[:, :, [*range(0,30), *range(114,144)]], axis=(1, 2))
    
    # Traffic pattern features
    hourly_means = np.mean(data, axis=1)
    hourly_std = np.std(data, axis=1)
    
    # Advanced statistical features
    skewness = stats.skew(data, axis=(1, 2))
    kurtosis = stats.kurtosis(data, axis=(1, 2))
    
    # Ratios and differences
    rush_ratio = morning_rush / (evening_rush + 1e-6)
    day_night_ratio = (morning_rush + evening_rush) / (night + 1e-6)
    peak_variation = np.max(data, axis=(1, 2)) - np.min(data, axis=(1, 2))
    
    # Temporal patterns
    first_hour = np.mean(data[:, :, :6], axis=(1, 2))
    last_hour = np.mean(data[:, :, -6:], axis=(1, 2))
    
    features = np.concatenate([
        np.mean(data, axis=(1, 2)).reshape(-1, 1),
        np.std(data, axis=(1, 2)).reshape(-1, 1),
        np.max(data, axis=(1, 2)).reshape(-1, 1),
        np.min(data, axis=(1, 2)).reshape(-1, 1),
        morning_rush.reshape(-1, 1),
        evening_rush.reshape(-1, 1),
        mid_day.reshape(-1, 1),
        night.reshape(-1, 1),
        rush_ratio.reshape(-1, 1),
        day_night_ratio.reshape(-1, 1),
        peak_variation.reshape(-1, 1),
        skewness.reshape(-1, 1),
        kurtosis.reshape(-1, 1),
        first_hour.reshape(-1, 1),
        last_hour.reshape(-1, 1),
        np.percentile(data, [25, 50, 75], axis=(1, 2)).T,
        np.var(hourly_means, axis=1).reshape(-1, 1),
        np.var(hourly_std, axis=1).reshape(-1, 1)
    ], axis=1)
    
    return features


def extract_features_from_data(train_data, test_data):
    # Extract features
    X_train = extract_features(train_data)
    X_test = extract_features(test_data)

    return X_train, X_test