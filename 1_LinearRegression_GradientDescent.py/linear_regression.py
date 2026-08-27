import numpy as np

def compute_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    Computes the Mean Squared Error (MSE) cost function for linear regression.
    
    Args:
        X (np.ndarray): Feature matrix of shape (m, n) where m is the number of examples 
                        and n is the number of features.
        y (np.ndarray): Target values of shape (m,).
        w (np.ndarray): Model parameters (weights) of shape (n,).
        b (float): Model parameter (bias).
        
    Returns:
        float: The total cost.
    """
    m = X.shape[0]
    predictions = np.dot(X, w) + b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    
    return float(cost)

def compute_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:
    """
    Computes the gradient of the cost function with respect to w and b.
    
    Args:
        X (np.ndarray): Feature matrix of shape (m, n).
        y (np.ndarray): Target values of shape (m,).
        w (np.ndarray): Model parameters (weights) of shape (n,).
        b (float): Model parameter (bias).
        
    Returns:
        tuple[np.ndarray, float]: 
            - dj_dw: The gradient of the cost w.r.t. the parameters w, shape (n,).
            - dj_db: The gradient of the cost w.r.t. the parameter b, float.
    """
    m = X.shape[0]
    predictions = np.dot(X, w) + b
    
    # Vectorized gradient computation
    dj_dw = (1 / m) * np.dot(X.T, (predictions - y))
    dj_db = (1 / m) * np.sum(predictions - y)
    
    return dj_dw, dj_db

def gradient_descent(X: np.ndarray, y: np.ndarray, w_in: np.ndarray, b_in: float, 
                     learning_rate: float, num_iters: int) -> tuple[np.ndarray, float, list[float]]:
    """
    Performs batch gradient descent to learn w and b.
    
    Args:
        X (np.ndarray): Feature matrix of shape (m, n).
        y (np.ndarray): Target values of shape (m,).
        w_in (np.ndarray): Initial model parameters (weights) of shape (n,).
        b_in (float): Initial model parameter (bias).
        learning_rate (float): The step size for gradient descent (alpha).
        num_iters (int): Number of iterations to run gradient descent.
        
    Returns:
        tuple[np.ndarray, float, list[float]]:
            - w: Updated weights after gradient descent.
            - b: Updated bias after gradient descent.
            - J_history: History of cost values to analyze convergence.
    """
    w = np.copy(w_in)
    b = b_in
    J_history = []
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        
        # Update parameters
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        
        # Save cost for monitoring convergence
        if i < 100000:  # Prevent memory exhaustion
            cost = compute_cost(X, y, w, b)
            J_history.append(cost)
            
    return w, b, J_history

if __name__ == "__main__":
    # Create simple dummy data
    X_test = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_test = np.array([2.0, 4.0, 6.0, 8.0])  # y = 2x
    
    # Initial parameters
    w_init = np.array([0.0])
    b_init = 0.0
    
    # Run gradient descent
    w_final, b_final, j_hist = gradient_descent(
        X_test, y_test, w_init, b_init, learning_rate=0.01, num_iters=1000
    )
    
    # Print results
    print("--- Training Complete ---")
    print(f"Final Weight (w): {w_final[0]:.4f}")
    print(f"Final Bias (b):   {b_final:.4f}")
    print(f"Initial Cost:     {j_hist[0]:.4f}")
    print(f"Final Cost:       {j_hist[-1]:.4f}")