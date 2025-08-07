from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

class LinearRegressionModel:
    def __init__(self):
        self.scalar= StandardScaler()
        
        ### Chnage eta0 for other datasets, higher values for small numbers
        
        self.reg= SGDRegressor()
    
    def normalize(self,x):
        x_norm=self.scalar.fit_transform(x)
        return x_norm
        
    def fit(self,x,y):
        self.reg.fit(x,y)
    
    def predict(self,x):
        f=self.reg.predict(x)
        return f
    
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(9)
    x=90*np.random.rand(90,1)
    y=50+2*x+(10*np.random.rand(90,1))
    y.ravel()
    
    model= LinearRegressionModel()
    x_norm= model.normalize(x)
    model.fit(x_norm,y)
    f= model.predict(x_norm)
    plt.scatter(x[:,0],y, label='Data points')
    plt.plot(x[:, 0], f, color='red', label='Regression line')
    plt.show()
    
        
        