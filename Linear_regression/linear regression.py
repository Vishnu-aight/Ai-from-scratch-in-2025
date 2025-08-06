import numpy as np
import matplotlib.pyplot as plt
import math

class LinearRegression:
    def __init__(self,learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
        cost_history = []
        self.cost_history = cost_history
        
    def fit(self, x, y):
        self.w = 0
        self.b = 0
        for i in range(self.iterations):
            dw,db= self.compute_gradient(x, y)
            alpha = self.learning_rate
            self.w=self.w-alpha*dw
            self.b=self.b-alpha*db
            if i<10000:
                if i%math.ceil(self.iterations/10)==0:
                    cost = self.cost_function(x, y)
                    self.cost_history.append(cost)
                    
    def cost_function(self,x, y):
        m=len(x)
        return np.sum((1/(2*m) * ((y-(self.w*x+self.b))**2)))

    def compute_gradient(self,x,y):
        m=len(x)
        f= self.w*x + self.b
        f = f.reshape(-1, 1)  # Ensure f is a column vector
        dw=(f-y).dot(x)/m
        db=np.sum(f-y)/m
        return dw, db

    def plot_cost_history(self):
        plt.plot(self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost Function History')
        plt.show()
        
        
if __name__ == "__main__":
    # Example usage
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 5, 7, 11])
    
    model = LinearRegression(learning_rate=0.01, iterations=10000)
    model.fit(x, y)
    
    print("Weights:", model.w)
    print("Bias:", model.b)
    
    model.plot_cost_history()