# ML Algorithms From Scratch

This repository contains bare-bones implementations of fundamental Machine Learning algorithms from scratch, strictly using Python and `NumPy`.

## 1. Linear Regression & Gradient Descent

### Cost Function (Mean Squared Error)
The model computes the cost $J(w, b)$ to measure the accuracy of predictions against the actual targets:
$$ J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$

### Gradient Descent Algorithm
To minimize the cost function, we update the parameters $w$ and $b$ iteratively using the learning rate $\alpha$:
$$ w_j = w_j - \alpha \frac{\partial J(w, b)}{\partial w_j} $$
$$ b = b - \alpha \frac{\partial J(w, b)}{\partial b} $$

Where the gradients are calculated as:
$$ \frac{\partial J(w, b)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) x_j^{(i)} $$
$$ \frac{\partial J(w, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) $$