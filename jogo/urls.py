from django.urls import path
from . import views

urlpatterns = [
    path('',           views.index,     name='index'),
    path('novo-jogo/', views.novo_jogo, name='novo_jogo'),
    path('jogo/',      views.jogo,      name='jogo'),
    path('interrogar/',views.interrogar,name='interrogar'),
    path('acusar/',    views.acusar,    name='acusar'),
    path('reiniciar/', views.reiniciar, name='reiniciar'),
]
