from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('ejercicio5/', views.ejercicio5, name='ejercicio5'),
    path('ejercicio6/', views.ejercicio6, name='ejercicio6'),
    path('ejercicio7/', views.ejercicio7, name='ejercicio7'),
    path('ejercicio8/', views.ejercicio8, name='ejercicio8'),
    path('ejercicio9/', views.ejercicio9, name='ejercicio9'),
    path('ejercicio10/', views.ejercicio10, name='ejercicio10'),
]



