import numpy as np
import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

data = pd.read_csv('./dataset/Breast_cancer_data.csv')
x = data.drop(columns = ['diagnosis'])
y = data['diagnosis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1234)

# mergeTrain = pd.concat([x_train, y_train], axis=1)
# trainData = pd.DataFrame(mergeTrain)
# trainData.to_csv('train-data.csv')

# mergeTest = pd.concat([x_test, y_test], axis=1)
# testData = pd.DataFrame(mergeTest)
# testData.to_csv('test-data.csv')



def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

regressor = LogisticRegression(lr=0.0001, n_iters=15000)
regressor.fit(x_train, y_train)
predictions = regressor.predict(x_test)

print("LR classification accuracy:", accuracy(y_test, predictions))
plt.plot(list(range(len(regressor.cost))), regressor.cost)
plt.show()
# print(predictions)