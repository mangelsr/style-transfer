from django.urls import path

from .views import combine

urlpatterns = [
    path('', combine, name='index'),
]
