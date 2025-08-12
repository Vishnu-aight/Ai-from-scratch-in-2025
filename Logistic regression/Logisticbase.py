import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionModel:
    def __init__(self,x,y,w,b):
        self.x=x
        self.y=y
        self.w=w
        self.b=b
        self.cost_history=[]
    
    def sigmoid(self):
        z=np.dot(self.x,self.w)+self.b
        return 1/(1+(np.exp(-z)))
    
    
    def loss(self):
        f=self.sigmoid()
        loss=-y*(np.log(f))-(1-y)*(np.log(1-f))
        return loss
    
    def cost(self):
        loss=self.loss()
        m=len(self.y)
        return 1/m*np.sum(loss)
    
    def gradient(self):
        f=self.sigmoid()
        m=len(self.y)
        dw=1/m*(np.dot(self.x.T,(f-self.y)))
        db=1/m*(np.sum(f-self.y))
        return dw, db
    
    def fit(self,learning_rate=0.01, epochs=1000):
        for i in range(epochs):
            dw,db=self.gradient()
            self.w=self.w-learning_rate*dw
            self.b=self.b-learning_rate*db
            self.cost_history.append(self.cost())
            
    def predict(self, x):
        x= self.normalize(x)
        y=np.dot(x,self.w)+self.b
        z=1/(1+np.exp(-y))
        return (z >= 0.5).astype(int)
    
    def normalize(self, x):
        return (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    
    
if __name__ == "__main__":
    x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])
    w = np.zeros(x.shape[1])
    b = 0.0

    model = LogisticRegressionModel(x, y, w, b)
    model.x = model.normalize(model.x)
    model.fit(learning_rate=0.01, epochs=1000)
    predictions = model.predict(x)
    print("Predictions:", predictions)
    plt.plot(model.cost_history)
    plt.show()
        
    

    