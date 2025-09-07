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
from rest_framework.parsers import MultiPartParser, FormParser
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


class ProfileImageUpdateView(APIView):
    """Update user profile image"""

    serializer_class = serializers.ProfileImageUpdateSerializer
    permission_classes = (rest_permissions.IsAuthenticated,)
    parser_classes = [MultiPartParser, FormParser]

    def patch(self, request):
        """Update profile image"""
        user = request.user
        serializer = serializers.ProfileImageUpdateSerializer(
            instance=user,
            data=request.data,
            partial=True,
            context={"request": request},
        )

        if serializer.is_valid():
            serializer.save()

            # Return updated user data
            user_serializer = serializers.UserSerializer(
                instance=user,
                context={"request": request},
            )

            return Response(
                {
                    "success": True,
                    "message": "Profile image updated successfully",
                    "user": user_serializer.data,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "success": False,
                "message": "Failed to update profile image",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    def delete(self, request):
        """Delete profile image"""
        user = request.user

        if user.profile_image:
            try:
                # Delete the file from storage
                user.profile_image.delete(save=False)
                user.profile_image = None
                user.save()

                return Response(
                    {
                        "success": True,
                        "message": "Profile image deleted successfully",
                    },
                    status=status.HTTP_200_OK,
                )

            except Exception as e:
                return Response(
                    {
                        "success": False,
                        "message": f"Failed to delete profile image: {str(e)}",
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
        else:
            return Response(
                {
                    "success": False,
                    "message": "No profile image to delete",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )


class ProfileUpdateView(APIView):
    """Update user profile information"""

    serializer_class = serializers.UserProfileUpdateSerializer
    permission_classes = (rest_permissions.IsAuthenticated,)
    parser_classes = [MultiPartParser, FormParser]

    def patch(self, request):
        """Update profile information"""
        user = request.user
        serializer = serializers.UserProfileUpdateSerializer(
            instance=user,
            data=request.data,
            partial=True,
            context={"request": request},
        )

        if serializer.is_valid():
            serializer.save()

            # Return updated user data
            user_serializer = serializers.UserSerializer(
                instance=user,
                context={"request": request},
            )

            return Response(
                {
                    "success": True,
                    "message": "Profile updated successfully",
                    "user": user_serializer.data,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "success": False,
                "message": "Failed to update profile",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )
