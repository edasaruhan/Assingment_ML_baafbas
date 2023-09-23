import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

iris = load_iris()
X = iris.data
y = iris.target

y_binary = np.where(y==0,1,0)


X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

def cost_func(y_true,y_pred):
	epsilon = 1e-15
	y_pred = np.clip(y_pred,epsilon,1-epsilon)
	return - (y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))

def training_func(X,y,learning_rate,iterations,reg_parameter):
	m,n=X.shape
	W = np.zeros(n)
	b= 0

	for _ in range(iterations):
		z = np.dot(X,W) + b
		A = 1/(1+np.exp(-z))

		dW = (1/m)*np.dot(X.T,(A-y))+(reg_parameter/m)*W
		db = np.mean(A-y)

		W -= learning_rate*dW
		b -= learning_rate*db
	return W,b

