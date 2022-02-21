import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets

data = pd.read_csv('BreastCancer.csv', index_col=False)

x = data.drop(columns = ['diagnosis'])
y = data['diagnosis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1234)

mergeTrain = pd.concat([x_train, y_train], axis=1)
trainData = pd.DataFrame(mergeTrain)
trainData.to_csv('train.csv', index=False)

mergeTest = pd.concat([x_test, y_test], axis=1)
testData = pd.DataFrame(mergeTest)
testData.to_csv('test.csv', index=False)

