import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.cost_history: list[float] = []  # Added to track loss over iterations

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(z)
        
        epsilon = 1e-9
        cost = (-1 / m) * np.sum(
            y * np.log(predictions + epsilon) + 
            (1 - y) * np.log(1 - predictions + epsilon)
        )
        return float(cost)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.cost_history = []

        for i in range(self.num_iterations):
            z = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(z)

            # Compute gradients
            dw = (1 / m) * np.dot(X.T, (predictions - y))
            db = (1 / m) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Record cost
            current_cost = self.compute_cost(X, y)
            self.cost_history.append(current_cost)

            # Print progress every 100 iterations
            if verbose and i % 100 == 0:
                print(f"Iteration {i:4d} | Cost: {current_cost:.6f}")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        z = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(z)
        return np.where(predictions >= threshold, 1, 0)