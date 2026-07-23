import numpy as np
from sklearn.linear_model import LinearRegression


def least_squares(data: np.ndarray) -> np.ndarray:
    A = data[:, :-1]
    b = data[:, -1]

    output = np.linalg.lstsq(A, b)

    return output


def regression(data: np.ndarray) -> LinearRegression:
    A = data[:, :-1]
    b = data[:, -1]

    model = LinearRegression(
        fit_intercept=False, tol=1e-15
    )  # stupid version difference. cost me a whole day

    model.fit(A, b)

    return model


def normal(data: np.ndarray) -> np.ndarray:
    A = data[:, :-1]
    b = data[:, -1]

    x = np.linalg.solve(A.T @ A, A.T @ b)
    return x
