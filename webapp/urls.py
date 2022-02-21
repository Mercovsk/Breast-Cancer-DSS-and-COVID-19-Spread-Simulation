""" from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='webapp-home'),
    path('dss.html/', views.dss, name='webapp-dss'),
    path('dss.html/result', views.result),
    path('simulation.html/', views.simulation, name='webapp-simulation'),
    path('about.html/', views.about, name='webapp-about'),

] """

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='webapp-home'),
    path('train.html/', views.train, name='webapp-dss-train'),
    path('train.html/train', views.train),
    path('train.html/toTrain', views.toTrain),
    path('predict.html/', views.predict, name='webapp-dss-predict'),
    path('predict.html/toPredict', views.toPredict),
    path('simulation.html/', views.simulation, name='webapp-simulation'),
    path('simulation.html/toSimulate', views.toSimulate),


]