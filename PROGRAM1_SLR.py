#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
data = {
    "shear": [2160.70, 1680.15, 2318.00, 2063.30, 2209.50, 1710.30, 1786.70, 2577.00,
              2359.90, 2258.70, 2167.20, 2401.55, 1781.80, 2338.75, 1767.30, 2055.50,
              2416.40, 2202.50, 2656.20, 1755.70],
    
    "age": [15, 50, 23, 75, 8.00, 17.00, 5.50, 19.00, 24.00, 2.50, 7.50, 11.00,
            13.00, 3.75, 25.00, 9.75, 22.00, 18.00, 6.00, 12.50]
}

df = pd.DataFrame(data)

X = df['age']
y = df['shear']

X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()


intercept = results.params[0]
slope = results.params[1]

print("Intercept:", intercept)
print("Slope:", slope)


# In[ ]:




