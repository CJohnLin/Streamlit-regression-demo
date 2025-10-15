import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def generate_data(a=2.0, b=1.0, noise=1.0, n_points=50):
    np.random.seed(42)
    X = np.random.rand(n_points, 1) * 10
    y = a * X + b + np.random.randn(n_points, 1) * noise
    return X, y

def train_and_evaluate(a=2.0, b=1.0, noise=1.0, n_points=50):
    X, y = generate_data(a, b, noise, n_points)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "R2": r2_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred)
    }
