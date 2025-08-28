#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
data = {
    "shear": [2160.70, 1680.15, 2318.00, 2063.30, 2209.50, 1710.30,
              1786.70, 2577.00, 2359.90, 2258.70, 2167.20, 2401.55, 1781.80,
              2338.75, 1767.30, 2055.50, 2416.40, 2202.50, 2656.20, 1755.70],
    "age": [15.50, 23.75, 8.00, 17.00, 5.50, 19.00, 24.00, 2.50, 7.50,
            11.00, 13.00, 3.75, 25.00, 9.75, 22.00, 18.00, 6.00, 12.50, 2.00,
            21.50]
}
X = np.array(data["age"]).reshape(-1, 1)  
y = np.array(data["shear"])  

def stochastic_gradient_descent(X, y, initial_learning_rate=0.001, decay_rate=0.01, n_iteration=1000):
    n = len(y)
    X_b = np.c_[np.ones(n), X]  
    theta = np.random.randn(X_b.shape[1])  
    grad_clip_threshold = 10  #
    for iteration in range(n_iteration):
        indices = np.random.permutation(n)
        for i in indices:  
            X_i = X_b[i:i+1] 
            y_i = y[i:i+1]  
            gradients = 2 * X_i.T.dot(X_i.dot(theta) - y_i)
            gradients = np.clip(gradients, -grad_clip_threshold, grad_clip_threshold)
            learning_rate = initial_learning_rate / (1 + decay_rate * iteration)
            theta -= learning_rate * gradients
    return theta
theta_sgd = stochastic_gradient_descent(X, y)

print("\nStochastic Gradient Descent:")
print("Intercept:", theta_sgd[0])  
print("Slope:", theta_sgd[1]) 

