def preprocess_data(data):
    data = data.sort_index()

    # Keep only numeric columns
    data = data.select_dtypes(include=['number'])

    # Resample hourly
    data = data.resample('h').mean()

    # FIXED HERE 👇
    data = data.ffill()

    return data