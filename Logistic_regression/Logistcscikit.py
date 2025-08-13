import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

x= np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y= np.array([0, 0, 1, 1])
scalar=StandardScaler()
x_norm=scalar.fit_transform(x)
model=LogisticRegression()
model.fit(x_norm,y)
predictions=model.predict(x_norm)
cost_history=model.decision_function(x_norm)
plt.plot(range(len(cost_history)), cost_history, label='Cost History')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost History over Epochs')
plt.show()
