# pages/urls.py
from django.urls import path
from .views import homePageView, homePost, results

urlpatterns = [
    path('', homePageView, name='home'),
    path('homePost/', homePost, name='homePost'),
    path('<str:sd>/<str:Q25>/<str:IQR>/<str:spEnt>/<str:sfm>/<str:meanfun>/results/', results, name='results'),
]
