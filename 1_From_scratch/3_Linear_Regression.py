import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

class LR:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self,X_train,y_train):

        num = 0
        den = 0

        for i in range(X_train.shape[0]):

            num = num + ((X_train[i] - X_train.mean())*(y_train[i] - y_train.mean()))
            den = den + ((X_train[i] - X_train.mean())*(X_train[i] - X_train.mean()))

        self.m = num/den
        self.b = y_train.mean() - (self.m * X_train.mean())

    def predict(self,X_test):
        return self.m * X_test + self.b
    
df = pd.read_csv('Required Datasets/placement.csv')

x = df['cgpa']
y = df['package']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

sc = StandardScaler()
x_train = sc.fit_transform(x_train.to_frame())
x_test = sc.transform(x_test.to_frame())

model = LR()
model.fit(x_train,y_train)
predictions = model.predict(x_test)

error = mean_absolute_percentage_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(error)
print(r2)