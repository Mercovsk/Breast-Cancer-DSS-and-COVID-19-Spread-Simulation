from django.shortcuts import render
from django.http import HttpResponse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #to split the dataset
import matplotlib.pyplot as plt
import pickle

from logistic_regression import LogisticRegression 
from pathlib import Path

from plotly.offline import plot
import plotly.graph_objs as go
from plotly.graph_objs import Scatter
import plotly.graph_objects as gos
import plotly.express as px
from virus import Virus 
#from sklearn import metrics


def home(request):
    return render(request, 'webapp/index.html')

def train(request):

    return render(request, 'webapp/train.html', {"show":'hide'})

def predict(request):
    return render(request, 'webapp/predict.html')

def simulation(request):
    return render(request, 'webapp/simulation.html')
    
def toSimulate(request):

    population = int(request.GET['population'])  # kuhain yung mga values
    r0 = float(request.GET['r0'])
    init_infected = int(request.GET['init_infected'])
    mild_recovery = float(request.GET['mild_recovery'])
    severe_recovery = float(request.GET['severe_recovery'])
    fatality_rate = float(request.GET['fatality_rate'])

    simuInput = [population, r0, init_infected, severe_recovery, mild_recovery, fatality_rate]

    AIRBORNE_PARAMS = {
    "r0": simuInput[1],
    "incubation": 5,
    "percent_mild": simuInput[4],
    "mild_recovery": (7, 14),
    "percent_severe": simuInput[3],
    "severe_recovery": (21, 42),
    "severe_death": (14, 56),
    "fatality_rate": simuInput[5],
    "serial_interval": 7,
    "init_population": simuInput[0],
    "init_infected": simuInput[2]
}
    disease = Virus(AIRBORNE_PARAMS)
    disease.animate()
    plt.show()

    return render(request, 'webapp/simulation.html')

def toTrain(request):
    main_data = pd.read_csv('BreastCancer.csv')
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    row, column = main_data.shape

    #if model.pickle esixt dont run this codeyyy for presentation muna
    """ if Path('model.pickle').is_file():
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1234)#split dataset to train 
        toTrainCSV(x_train, y_train) #call to save the selected train data to a one dataset .csv
        toTestCSV(x_test, y_test)# call to test CSV
        
        regressor = LogisticRegression(lr=0.0001, n_iters=15000)
        regressor.fit(x_train, y_train)
        filename = 'model.pickle'
        pickle.dump(regressor, open(filename, 'wb')) """
    
########## Breast Cancer Dataset Snippet #############
    data = main_data
    layout = gos.Layout(autosize = True, height = 235, margin = {'l': 5, 'r':5, 't':5, "b": 5})
    fig = gos.Figure(data=[gos.Table(
        header=dict(values=list(data.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[data.mean_radius, data.mean_texture,
                           data.mean_perimeter, data.mean_area, data.mean_smoothness, data.diagnosis],
                   fill_color='lavender',
                   align='left'))
                   
    ], layout = layout
    )
############ Missing Data Percentage ##################
    featureList = ['mean_radius', 'mean_texture',	'mean_perimeter',	'mean_area',	'mean_smoothness']
    layout2 = gos.Layout(autosize = True, height = 150, width = 500, margin = {'l': 5, 'r':5, 't':5, "b": 5})
    featureMiss = main_data[featureList].isin({0}).sum()/row*100
    test = gos.Figure(data=[gos.Table(
        header=dict(values=("feature", "Missing data"),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[featureList, featureMiss],
                   fill_color='lavender',
                   align='left'))
    ], layout = layout2
    )
############## Accuracy #####################
    y_test = test_data['diagnosis']
    x_test = test_data.drop(columns = ['diagnosis'])
    with open('model.pickle', 'rb') as f:
        loadModel= pickle.load(f)
    pred = loadModel.predict(x_test)

################ Cost Fucntion ##################
    
############# Data Describe ####################
    data_des = main_data.describe()

############ Data Correlation ##################

    corr = main_data.corr()
    #dataCor = "webapp/images/Corr_img.png"
    trace = gos.Heatmap(
            x=corr.index,       
            y=corr.index,       
            z=corr.values,      
            colorscale='Viridis', #colorscale to define different colors for different range of values in correlation matrix
    )

    layout = gos.Layout(
    title="Data Correlation",       
    autosize=True,             
    )

    wew = gos.Data([trace])
    heatm = gos.Figure(data=wew, layout=layout)
    heat = heatm.to_html(full_html = False)

############ Data Visualize ##################

    Acc = accuracy(y_test, pred)
    des = data_des.to_html()
    BCdata = fig.to_html(full_html = False)
    dataMiss = test.to_html(full_html = False)
    heat = heatm.to_html(full_html = False)
    show = 'show'
    return render(request, 'webapp/train.html', {"BCdata": BCdata, "dataMiss": dataMiss, "data_des": des, "Acc": Acc, "heat": heat, "show":show})

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def toPredict(request):
    # loadModel = pickle.load(open('model.sav', 'rb'))#load trained model na nakasave
    with open('model.pickle', 'rb') as f:
        loadModel = pickle.load(f)
    

    radius = float(request.GET['radius'])  # kuhain yung mga values
    texture = float(request.GET['texture'])
    perimeter = float(request.GET['perimeter'])
    area = float(request.GET['area'])
    smoothness = float(request.GET['smooth'])

    testInput = [radius, texture, perimeter, area, smoothness]

    predicted = loadModel.predict([testInput])

    if predicted == [1]:
        result = "Breast Cancer Positive"
    elif predicted == [0]:
        result = "Breast Cancer Negative"
    else:
        return "Error"  
    return render(request, 'webapp/predict.html', {"predicted": result, "radius": radius, "texture": texture, "perimeter": perimeter, "area": area, "smoothness": smoothness})
    # print(predicted)

def toTrainCSV(X, Y):
    td = pd.concat([X, Y], axis=1)
    trainData = pd.DataFrame(td)
    trainData.to_csv('train.csv')
    print("Train Data exported as csv")

def toTestCSV(X, Y):
    td = pd.concat([X, Y], axis=1)
    testData = pd.DataFrame(td)
    testData.to_csv('test.csv')
    print('Test Data exported as test.csv')

# def dataCorr(datas):
#     corr = datas.corr()#corelation
#     plt.figure(figsize=(10,10))
#     plt.title('Correlation')
#     sns.heatmap(corr, annot=True, square=True)
#     test = sns.pairplot(data)
#     return = test
