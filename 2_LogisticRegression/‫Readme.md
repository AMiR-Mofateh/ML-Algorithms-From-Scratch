# Logistic Regression & Classification

This module contains a pure NumPy implementation of Logistic Regression trained via Gradient Descent.

## Cost Function Comparison: Linear vs. Logistic Regression

Selecting the appropriate cost function is critical for the convergence of the Gradient Descent algorithm.

### Linear Regression (Mean Squared Error)
For Linear Regression, the standard cost function is the Mean Squared Error (MSE):

$$ J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$

Where $f_{w,b}(x^{(i)}) = w \cdot x^{(i)} + b$. This surface is strictly convex, ensuring Gradient Descent easily converges to the global minimum. 

### Logistic Regression (Binary Cross-Entropy)
In Logistic Regression, the prediction function incorporates the non-linear sigmoid activation: $f_{w,b}(x^{(i)}) = \frac{1}{1 + e^{-(w \cdot x^{(i)} + b)}}$. 

If we apply MSE to this non-linear function, the resulting cost surface becomes non-convex (full of local minima), preventing Gradient Descent from finding the optimal weights reliably. 

To restore a convex optimization landscape, we use the **Binary Cross-Entropy (Log Loss)** function:

$$ J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f_{w,b}(x^{(i)})) + (1 - y^{(i)}) \log(1 - f_{w,b}(x^{(i)})) \right] $$

This formulation heavily penalizes confident but incorrect predictions, producing a smooth, bowl-shaped cost function that guarantees convergence to the global minimum.