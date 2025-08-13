import numpy as np
from Linear_regression.multiple_linear import LinearRegression
from Logistic_regression.Logisticbase import LogisticRegressionModel
import matplotlib.pyplot as plt

class RegularizationModel:
    def __init__(self, x, y,model_type, lamda, alpha=0.01, epochs=1000):
        self.x = x
        self.y = y
        self.model_type = model_type
        self.alpha = alpha
        self.epochs = epochs
        self.model = None
        self.lamda = lamda
        self.cost_history = []

    def initialize_model(self):
        if self.model_type == 'linear':
            self.model = LinearRegression(self.x, self.y, w=np.zeros(self.x.shape[1]), b=0.0)
        elif self.model_type == 'logistic':
            w = np.zeros(self.x.shape[1])
            b = 0.0
            self.model = LogisticRegressionModel(self.x, self.y, w, b)
            
    def cost(self):
        return self.model.cost_function()+self.lamda/(2*self.x.shape[0])*np.sum(np.square(self.model.w))
        
    
    def fit(self):
        self.initialize_model()
        self.model.normalization()  # Normalize the feature matrix
        for i in range(self.epochs):
            dw,db=self.model.compute_gradient()
            dw=dw+self.lamda/self.x.shape[0]*self.model.w
            self.model.w =self.model.w-self.alpha*dw
            self.model.b = self.model.b-self.alpha*db
            if i % 100 == 0:
                cost = self.cost()
                self.cost_history.append(cost)
                
if __name__ == "__main__":
    x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])
    
    # Example for Linear Regression with Regularization
    linear_model = RegularizationModel(x, y, model_type='linear', lamda=0.1)
    linear_model.fit()
    print("Linear Regression Weights:", linear_model.model.w)
    
    # Example for Logistic Regression with Regularization
    logistic_model = RegularizationModel(x, y, model_type='logistic', lamda=0.1)
    logistic_model.fit()
    print("Logistic Regression Weights:", logistic_model.model.w)
    
    plt.plot(range(len(linear_model.cost_history)), linear_model.cost_history, label='Linear Cost History')
    plt.plot(range(len(logistic_model.cost_history)), logistic_model.cost_history, label='Logistic Cost History')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Cost History over Epochs')
    plt.show()