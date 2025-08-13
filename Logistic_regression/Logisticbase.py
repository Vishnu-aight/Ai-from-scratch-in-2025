import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionModel:
    def __init__(self,x,y,w,b):
        self.x=x.copy()
        self.y=y
        self.w=w
        self.b=b
        self.sigma=None
        self.mu=None
        self.cost_history=[]
    
    def sigmoid(self):
        z=np.dot(self.x,self.w)+self.b
        return 1/(1+(np.exp(-z)))
    
    
    def loss(self):
        f=self.sigmoid()
        loss=-self.y*(np.log(f))-(1-self.y)*(np.log(1-f))
        return loss
    
    def cost_function(self):
        loss=self.loss()
        m=len(self.y)
        return 1/m*np.sum(loss)
    
    def compute_gradient(self):
        f=self.sigmoid()
        m=len(self.y)
        dw=1/m*(np.dot(self.x.T,(f-self.y)))
        db=1/m*(np.sum(f-self.y))
        return dw, db
    
    def fit(self,learning_rate=0.01, epochs=1000):
        self.normalization()
        for i in range(epochs):
            dw,db=self.compute_gradient()
            self.w=self.w-learning_rate*dw
            self.b=self.b-learning_rate*db
            self.cost_history.append(self.cost_function())
            
    def predict(self, x):
        x= (x-self.mu)/(self.sigma + 1e-8)
        y=np.dot(x,self.w)+self.b
        z=1/(1+np.exp(-y))
        return (z >= 0.5).astype(int)
    
    def normalization(self):
        self.mu = np.mean(self.x, axis=0)
        self.sigma = np.std(self.x, axis=0)
        self.x=(self.x - self.mu) / (self.sigma + 1e-8)
    
    
if __name__ == "__main__":
    x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])
    w = np.zeros(x.shape[1])
    b = 0.0

    model = LogisticRegressionModel(x, y, w, b)
    model.fit(learning_rate=0.01, epochs=1000)
    predictions = model.predict(x)
    print("Predictions:", predictions)
    plt.plot(model.cost_history)
    plt.show()
        
    

    