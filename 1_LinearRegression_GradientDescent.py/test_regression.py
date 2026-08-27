import numpy as np
import matplotlib.pyplot as plt
from linear_regression import gradient_descent

def generate_dummy_data(m: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic linear data with some Gaussian noise.
    Equation: y = 3x + 4 + noise
    """
    np.random.seed(42) # For reproducibility
    
    # Generate features (x) between 0 and 2
    X = 2 * np.random.rand(m, 1)
    
    # Generate targets (y) with true w=3 and true b=4, plus noise
    y = 3 * X[:, 0] + 4 + np.random.randn(m)
    
    return X, y

def main():
    # 1. Prepare Data
    print("Generating data...")
    X_train, y_train = generate_dummy_data(m=100)
    
    # 2. Initialize Parameters & Hyperparameters
    w_init = np.zeros(X_train.shape[1])
    b_init = 0.0
    learning_rate = 0.1
    iterations = 50
    
    # 3. Train the Model
    print(f"Running Gradient Descent for {iterations} iterations...")
    w_final, b_final, J_hist = gradient_descent(
        X_train, y_train, w_init, b_init, learning_rate, iterations
    )
    
    print(f"Optimized weights (w): {w_final}")
    print(f"Optimized bias (b): {b_final:.4f}")
    
    # 4. Plotting Results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Learning Curve (Cost vs Iterations)
    plt.subplot(1, 2, 1)
    plt.plot(range(len(J_hist)), J_hist, color='blue', linewidth=2)
    plt.title('Learning Curve: Cost vs. Iterations', fontsize=12)
    plt.xlabel('Iterations', fontsize=10)
    plt.ylabel('Cost J(w, b)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Regression Line vs Actual Data
    plt.subplot(1, 2, 2)
    plt.scatter(X_train, y_train, color='red', marker='x', label='Actual Data')
    
    # Calculate predictions for the regression line
    predictions = np.dot(X_train, w_final) + b_final
    plt.plot(X_train, predictions, color='green', label=f'Fit: y = {w_final[0]:.2f}x + {b_final:.2f}')
    
    plt.title('Linear Regression Fit', fontsize=12)
    plt.xlabel('Feature (x)', fontsize=10)
    plt.ylabel('Target (y)', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()