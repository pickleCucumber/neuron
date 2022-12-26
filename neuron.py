from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


#сигмоидная функция\sigmoid function
def sigmoid(x):
  return 1/(1+np.exp(-x))

#производная сигмоиды\derivative sigmoid
def sigmoid_derivative(x):
  return sigmoid(x)*(1-sigmoid(x))

#среднеквадратичная ошибка\MSE
def loss(y_pred, y):
  y_pred=y_pred.reshape(-1,1)
  y=np.array(y).reshape(-1,1)
  return 0.5*np.mean((y_pred-y)**2)

class Neuron:
  def _init_(self, w=None, b=0):
    #w- вектор весов
    #b- смещение
    self.w=w
    self.b=b
    
  def acivate(self, x):
    retunr sigmoid(x)
    
  #рассчитывает ответ нейрона при предъявдении набора объектов
  def forward_pass(self, X):
    n=X.reshape[0]
    y_pred=np.zeros((n,1))
    y_pred=self.activate(X @ self.w.reshape(X.shape[1], 1)+ self.b)
    return y_pred.reshape(-1, 1)
  
  #обновляет значения весов нейрона в соответствии с этим объектом, меняет веса с помощью градиентного спуска
  def backward_pass(self, X, y, y_pred, learning_rate=0.1):
    n=len(y)
    y=np.array(y).reshape(-1,1)
    sigma=self.activate(X@self.w+self.b)
    self.w=self.w-learning_rate*(X.T@((sigma-y)*sigma*(1-sigma)))/n
    self.b=self.b-learning_rate*np.mean((sigma-y)*sigma*(1-sigma))
  def fit(self, X, y, num_epoch=5000):
    self.w=np.zeros((X.shape[1],1))
    self.b=0 #смещение
    loss_values=[]
    
    for i in range(num_epochs):
      #предсказание с текущими весами
      y_pred=self.forward_pass(X)
      #функция потерь с текущими весами
      loss_values.append(loss(y_pred, y))
      #обновляем веса в соответствии с тем, где ошиблись раньше
      self.backward_pass(X, y, y_pred)
    return loss_values
