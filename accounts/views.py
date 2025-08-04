from . import serializers
from rest_framework import status
from rest_framework.response import Response
from rest_framework.generics import CreateAPIView
from rest_framework import permissions as rest_permissions
from django.utils.translation import gettext_lazy as _
from django.db import transaction
from django.conf import settings
from dj_rest_auth.views import LoginView as DJREST_LoginView
from rest_framework.views import APIView
from rest_framework import viewsets
from .models import Department


class DepartmentViewset(viewsets.ModelViewSet):
    serializer_class = serializers.DepartmentSerializer
    queryset = Department.objects.all()
    permission_classes = (rest_permissions.AllowAny,)
    http_method_names = ["get"]


class SignUpViewset(CreateAPIView, DJREST_LoginView):
    """RegisterView takes a post method: Creates a user Account and sends
    AN OTP for user Activation
    """

    serializer_class = serializers.SignUpSerializer

    def get_response_data(self):
        if settings.ACCOUNT_EMAIL_VERIFICATION == "mandatory":
            return {
                "detail": _(
                    "Verification code has been sent to your e-mail or phone number."
                )
            }

    @transaction.atomic
    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save(request)
        # HACK :
        # code to get account token from dj_rest_auth
        self.serializer = serializer
        self.login()
        response = self.get_response()
        # send email to user and admin

        # end dj_rest_auth hack
        return response


class SendEmailOTPViewset(CreateAPIView):
    serializer_class = serializers.SendEmailOTPSerializer

    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.send_email(request)
        return Response(
            {
                "success": True,
                "message": "OTP has been sent to your email",
            }
        )


class VerifyOTPViewset(CreateAPIView):
    serializer_class = serializers.VerifyOTPSerializer

    def post(self, request):
        serializer = self.get_serializer(data=request.data)

        serializer.is_valid(raise_exception=True)
        serializer.verify_otp(request)
        return Response(
            data={
                "success": True,
                "data": serializer.data,
            },
            status=status.HTTP_200_OK,
        )


class ProfileView(APIView):
    serializer_class = serializers.UserSerializer
    permission_classes = (rest_permissions.IsAuthenticated,)

    def get(self, request):
        user = request.user
        serializer = serializers.UserSerializer(
            instance=user,
            many=False,
            context={"request": request},
        )
        return Response(data=serializer.data)
