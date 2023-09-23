import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

iris = load_iris()
X = iris.data
y = iris.target 

