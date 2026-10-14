def create_features(data):
    # Time features
    data['hour'] = data.index.hour
    data['day'] = data.index.dayofweek
    data['month'] = data.index.month

    # Lag features (VERY IMPORTANT)
    data['lag_1'] = data['Energy'].shift(1)
    data['lag_24'] = data['Energy'].shift(24)

    # Rolling averages
    data['rolling_mean_24'] = data['Energy'].rolling(window=24).mean()

    # Drop NaN after feature creation
    data = data.dropna()

    return data