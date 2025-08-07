import numpy as np
import matplotlib.pyplot as plt
from Linear_regression.multiple_linear import LinearRegression
x=np.array([1,2,3,4])  
x=x.reshape(-1, 1)  # Reshaping x to be a column
y=1+x**2
x=np.c_[x,x**2]  # Adding polynomial features

model = LinearRegression(learning_rate=0.01, iterations=10000)
model.fit(x, y)
plt.scatter(x[:, 0], y, color='red', label='Data Points')  # Scatter plot of the data points
plt.plot(x[:, 0], model.predict(x), color='blue', label='Fitted Line')  # Plotting the fitted line
plt.xlabel('x')
plt.ylabel('y')
plt.show()
