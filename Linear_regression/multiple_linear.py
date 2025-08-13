import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self,x,y,w,b, learning_rate=0.01, iterations=10000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = w
        self.b = b
        self.x= x.copy()
        self.y = y
        cost_history = []
        self.mu= None
        self.sigma = None
        self.cost_history = cost_history
        
    def fit(self):
        m,n = self.x.shape
        self.w = np.zeros(n)
        self.b = 0
        y=self.y.ravel()
        self.normalization()  # Normalize the feature matrix
        for i in range(self.iterations):
            dw,db= self.compute_gradient()
            alpha = self.learning_rate
            self.w=self.w-alpha*dw
            self.b=self.b-alpha*db
            if i<10000:
               cost = self.cost_function()
               self.cost_history.append(cost) 
                    
                    
    def cost_function(self):
        m=len(self.x)
        return np.sum((1/(2*m) * ((self.y-(np.dot(self.x,self.w)+self.b))**2)))

    def compute_gradient(self):
        m=len(self.x)
        f= np.dot(self.x,self.w) + self.b
        dw=1/m*(np.dot(self.x.T, (f-self.y)))
        db=np.sum(f-self.y)/m
        return dw, db

    def plot_cost_history(self):
        plt.plot(self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost Function History')
        plt.show()
        
    def normalization(self):
        """Normalize the feature matrix."""
        self.mu=np.mean(self.x,axis=0)
        self.sigma=np.std(self.x,axis=0)
        self.x=(self.x-self.mu)/(self.sigma + 1e-8)  # Adding a small value to avoid division by zero
    
    def predict(self,x):
        """Make predictions using the learned weights."""
        x_normalized = (x - self.mu) / (self.sigma + 1e-8)
        return np.dot(x_normalized, self.w) + self.b
    
if __name__ == "__main__":
    # Example usage
    x = np.array([[1, 2, 3, 4, 5],[1, 1, 1, 1, 1], [1, 2, 3, 4, 5], [4,5,6,7,8]])
    y = np.array([2, 3, 5, 7])
    w = np.zeros(x.shape[1])
    b = 0.0
    
    model = LinearRegression(x,y,w,b,learning_rate=0.01, iterations=10000)
    model.fit()
    model.predict(x)
    
    print("Weights:", model.w)
    print("Bias:", model.b)
    
    model.plot_cost_history()