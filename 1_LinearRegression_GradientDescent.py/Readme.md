# Linear Regression & Gradient Descent (From Scratch)

A clean, vectorized implementation of Multivariate Linear Regression and Gradient Descent from scratch using **Python** and **NumPy**, built without any high-level machine learning frameworks.

---

## 📌 Mathematical Foundation

### 1. Model Prediction (Hypothesis)
The linear regression model predicts target values using a linear combination of input features:
$$ f_{w,b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b $$

Where:
* $\mathbf{w}$ is the parameter weight vector of shape $(n,)$.
* $b$ is the scalar bias parameter.
* $\mathbf{x}$ is the feature vector of shape $(n,)$.

---

### 2. Cost Function (Mean Squared Error)
To quantify how well our parameters $\mathbf{w}$ and $b$ fit the training data, we compute the **Mean Squared Error (MSE)** cost function $J(\mathbf{w}, b)$:

$$ J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(\mathbf{x}^{(i)}) - y^{(i)} \right)^2 $$

Where $m$ is the total number of training examples.

---

### 3. Gradient Descent
To minimize $J(\mathbf{w}, b)$, parameters are iteratively updated in the direction of steepest descent, governed by the learning rate $\alpha$:

$$ w_j = w_j - \alpha \frac{\partial J(\mathbf{w}, b)}{\partial w_j} $$
$$ b = b - \alpha \frac{\partial J(\mathbf{w}, b)}{\partial b} $$

#### Partial Derivatives (Gradients)
The gradients with respect to the weights and bias are computed as follows:

$$ \frac{\partial J(\mathbf{w}, b)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)} $$

$$ \frac{\partial J(\mathbf{w}, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(\mathbf{x}^{(i)}) - y^{(i)} \right) $$

---

## 🛠 Vectorized Implementation

Rather than using explicit Python loops over $m$ examples, the code utilizes vectorized matrix operations via `NumPy` for high performance:

* **Predictions:** $\mathbf{\hat{Y}} = \mathbf{X}\mathbf{w} + b$
* **Weight Gradient:** $\nabla_{\mathbf{w}} J = \frac{1}{m} \mathbf{X}^T (\mathbf{\hat{Y}} - \mathbf{Y})$
* **Bias Gradient:** $\frac{\partial J}{\partial b} = \frac{1}{m} \sum (\mathbf{\hat{Y}} - \mathbf{Y})$

---

## 🚀 How to Run

1. **Activate Environment & Install Dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
