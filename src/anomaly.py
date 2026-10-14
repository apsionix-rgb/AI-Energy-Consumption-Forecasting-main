import numpy as np

def detect_anomalies(data):
    mean = data['Energy'].mean()
    std = data['Energy'].std()

    threshold_upper = mean + 2 * std
    threshold_lower = mean - 2 * std

    data['anomaly'] = 0

    data.loc[data['Energy'] > threshold_upper, 'anomaly'] = 1
    data.loc[data['Energy'] < threshold_lower, 'anomaly'] = -1

    return data