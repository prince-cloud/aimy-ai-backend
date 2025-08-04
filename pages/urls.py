from django.urls import path

from .views import HomePageView, about

urlpatterns = [
    path("", HomePageView.as_view(), name="home"),
    path("about/", about, name="about"),
]
