from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_model(data):
    features = ['hour', 'day', 'month', 'lag_1', 'lag_24', 'rolling_mean_24']

    X = data[features]
    y = data['Energy']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test