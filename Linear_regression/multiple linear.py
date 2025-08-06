import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self,learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
        cost_history = []
        self.cost_history = cost_history
        
    def fit(self, x, y):
        m,n = x.shape
        self.w = np.zeros(n)
        self.b = 0
        for i in range(self.iterations):
            dw,db= self.compute_gradient(x, y)
            alpha = self.learning_rate
            self.w=self.w-alpha*dw
            self.b=self.b-alpha*db
            if i<10000:
               cost = self.cost_function(x, y)
               self.cost_history.append(cost) 
                    
                    
    def cost_function(self,x, y):
        m=len(x)
        return np.sum((1/(2*m) * ((y-(np.dot(x,self.w)+self.b))**2)))

    def compute_gradient(self,x,y):
        m=len(x)
        f= np.dot(x,self.w) + self.b
        dw=1/m*(np.dot(x.T, (f-y)))
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
    x = np.array([[1, 2, 3, 4, 5],[1, 1, 1, 1, 1], [1, 2, 3, 4, 5], [4,5,6,7,8]])
    y = np.array([2, 3, 5, 7])
    
    model = LinearRegression(learning_rate=0.01, iterations=10000)
    model.fit(x, y)
    
    print("Weights:", model.w)
    print("Bias:", model.b)
    
    model.plot_cost_history()