from rest_framework.routers import DefaultRouter
from django.urls import path
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)
from . import views
from dj_rest_auth.views import LoginView

app_name = "accounts"

router = DefaultRouter()
router.register("departments", views.DepartmentViewset, basename="departments")

urlpatterns = [
    path("signup/", views.SignUpViewset.as_view(), name="signup"),
    path("profile/", views.ProfileView.as_view(), name="profile"),
    path(
        "signup/verify-otp/",
        views.VerifyOTPViewset.as_view(),
        name="signup-verify-otp",
    ),
    path(
        "signup/send-email-otp/",
        views.SendEmailOTPViewset.as_view(),
        name="signup-send-email-otp",
    ),
    path("login/", LoginView.as_view(), name="login"),
    path("token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("token/verify/", TokenVerifyView.as_view(), name="token_verify"),
]

urlpatterns += router.urls
