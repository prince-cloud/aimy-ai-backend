from rest_framework import serializers
from .models import CustomUser, Department
from dj_rest_auth.serializers import LoginSerializer
from datetime import datetime, timedelta
from django.core.cache import cache
from django.core.exceptions import ObjectDoesNotExist
from helpers import exceptions
from helpers.functions import generate_otp, email_address_exists
from django.db.models import Q
from django.http import HttpRequest
from django.db import transaction
from allauth.account.models import EmailAddress
from accounts.tasks import generic_send_mail, send_otp_email
from allauth.account.adapter import get_adapter
from django.core.exceptions import ValidationError as DjangoValidationError
from loguru import logger


class DepartmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Department
        fields = ["name"]


class SignUpSerializer(serializers.Serializer):
    first_name = serializers.CharField(max_length=50, required=True)
    last_name = serializers.CharField(max_length=15, min_length=3, required=True)
    email = serializers.EmailField(required=True)
    student_id = serializers.CharField(max_length=20, required=True)
    index_number = serializers.CharField(max_length=20, required=True)
    department = serializers.IntegerField(required=True)
    year_of_study = serializers.ChoiceField(choices=CustomUser.YearOfStudy.choices)

    password = serializers.CharField(write_only=True, min_length=6)

    cleaned_data = None

    # Add missing attributes that dj-rest-auth expects
    _has_phone_field = False
    _has_username_field = True
    _has_email_field = True

    def validate_student_id(self, student_id):
        try:
            user_number = CustomUser.objects.get(student_id=student_id)
            if user_number:
                raise exceptions.GeneralException(detail="Student ID already in use")
        except ObjectDoesNotExist:
            return student_id
        return student_id

    def validate_email(self, email):
        email = get_adapter().clean_email(email)
        if email and email_address_exists(email):
            raise exceptions.EmailAlreadyInUseException()
        return email

    def validate_password(self, password):
        return get_adapter().clean_password(password)

    def validate_department(self, department_id):
        try:
            Department.objects.get(id=department_id)
            return department_id
        except Department.DoesNotExist:
            raise serializers.ValidationError("Invalid department ID")

    def get_cleaned_data(self):
        return {
            "first_name": self.validated_data.get("first_name", ""),
            "last_name": self.validated_data.get("last_name", ""),
            "email": self.validated_data.get("email", ""),
            "password": self.validated_data.get("password", ""),
            "student_id": self.validated_data.get("student_id", ""),
            "index_number": self.validated_data.get("index_number", ""),
            "department_id": self.validated_data.get("department", ""),
            "year_of_study": self.validated_data.get("year_of_study", ""),
        }

    @transaction.atomic
    def save(self, request):
        adapter = get_adapter()
        user = adapter.new_user(request)
        self.cleaned_data = self.get_cleaned_data()
        user = adapter.save_user(request, user, self, commit=False)
        if "password" in self.cleaned_data:
            try:
                adapter.clean_password(self.cleaned_data["password"], user=user)
            except DjangoValidationError as exc:
                raise exceptions.InvalidPasswordException(detail=str(exc))
        user.student_id = self.cleaned_data["student_id"]
        user.index_number = self.cleaned_data["index_number"]
        # Get the department object from the ID
        department_id = self.cleaned_data["department_id"]
        try:
            department = Department.objects.get(id=department_id)
            user.department = department
        except Department.DoesNotExist:
            raise exceptions.GeneralException(detail="Invalid department ID")
        user.year_of_study = self.cleaned_data["year_of_study"]
        user.save()
        self.validated_data["email"] = user.email
        # create email address
        EmailAddress.objects.create(email=user.email, user=user)

        self.verify_account()
        return user

    def verify_account(self):
        email = self.validated_data["email"]
        student_id = self.validated_data["student_id"]
        if email:
            user_email = EmailAddress.objects.get(email=email)
            user_email.verified = True
            user_email.set_as_primary(conditional=True)
            user_email.save()
            user_account = CustomUser.objects.get(email=email)
            user_account.is_active = True
            user_account.save()
            user_account.backend = "allauth.account.auth_backends.AuthenticationBackend"
            self.validated_data["user"] = user_account
            return email
        user_account = CustomUser.objects.get(student_id=student_id)
        user_account.backend = "allauth.account.auth_backends.AuthenticationBackend"
        self.validated_data["user"] = user_account
        email = user_account.email
        user_email = EmailAddress.objects.get(email=email)
        user_email.verified = True
        user_email.set_as_primary(conditional=True)
        user_email.save()
        user_account.is_active = True
        user_account.save()
        return email


class UserSerializer(serializers.ModelSerializer):
    department = DepartmentSerializer()
    college = serializers.SerializerMethodField()

    def get_college(self, obj):
        return obj.department.college.name

    class Meta:
        model = CustomUser
        fields = (
            "id",
            "email",
            "first_name",
            "last_name",
            "student_id",
            "department",
            "college",
            "year_of_study",
            "phone_number",
            "profile_image",
        )


class CustomLoginSerializer(LoginSerializer):
    """
    Custom Login serializer to overide default dj-rest-auth login
    """

    def custom_validate(self, username):
        try:
            _username = CustomUser.objects.get(username=username)
            # print("=== username: ", _username)
            if not _username.is_active:
                # automatically generate and send otp to the user account.
                otp_generated = generate_otp(6)
                _username.otp = otp_generated
                _username.otp_expiry = datetime.now() + timedelta(minutes=5)
                _username.save()

                # send otp to the user's email
                try:
                    send_otp_email.delay(
                        recipient_email=_username.email,
                        otp_code=otp_generated,
                        user=_username,
                        site_url="",
                    )
                except Exception as e:
                    # Log the error but don't fail the request
                    from loguru import logger

                    logger.error(
                        f"Failed to send OTP email to {_username.email}: {str(e)}"
                    )

                raise exceptions.InactiveAccountException()
        except ObjectDoesNotExist:
            return username

    def get_client_ip(self, request):
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            ip = x_forwarded_for.split(",")[0]
        else:
            ip = request.META.get("REMOTE_ADDR")
        return ip

    def validate(self, attrs):
        request: HttpRequest = self.context.get("request")
        username = attrs.get("username")
        email = attrs.get("email")
        password = attrs.get("password")

        attempt = cache.get(f"login-attempt/{username}")
        if attempt:
            attempt += 1
        else:
            attempt = 1
        cache.set(f"login-attempt/{username}", attempt, 60 * 5)
        if attempt > 5:
            raise exceptions.TooManyLoginAttemptsException()

        if not (username or email):
            raise exceptions.ProvideUsernameOrPasswordException()

        if username:
            user_qs = CustomUser.objects.filter(
                Q(username=username) | Q(email=username) | Q(student_id=username),
            )
            if user_qs.exists():
                user = user_qs.first()
                if not user.is_active:
                    raise exceptions.AccountDeactivatedException()
                email = user.email
                attrs["email"] = user.email

            else:
                raise exceptions.UsernameDoesNotExistsException()
        elif email:
            user_qs = CustomUser.objects.filter(email=email)
            if user_qs.exists():
                user = user_qs.first()
                if not user.is_active:
                    raise exceptions.AccountDeactivatedException()
                username = user.username
                attrs["username"] = user.username
            else:
                raise exceptions.EmailDoesNotExistsException()

        _ = self.custom_validate(username)
        user: CustomUser = self.get_auth_user(username, email, password)

        if not user:
            raise exceptions.LoginException()

        try:
            user.last_login_ip = self.get_client_ip(request)
            user.save()
        except Exception as e:
            logger.error(f"Error saving last login IP: {str(e)}")
        cache.delete(f"login-attempt/{username}")
        attrs = super().validate(attrs)
        return attrs


class SendEmailOTPSerializer(serializers.Serializer):
    email = serializers.EmailField(allow_blank=False)

    def send_email(self, request):
        """Send email Generate OTP and key and saves them in user account,
        Sends email with an otp to user for Account Activation"""
        email = self.validated_data["email"]
        otp = generate_otp(6)

        # validate if customer with the phone number exists
        if CustomUser.objects.filter(
            email=email,
        ).exists():
            raise exceptions.EmailAlreadyInUseException()

        cache.set(f"otp/email/{email}", otp, 60 * 5)

        # Send OTP email using our custom template
        try:
            # Create a mock user object for the email template
            from django.contrib.auth.models import AnonymousUser

            mock_user = AnonymousUser()
            mock_user.username = email.split("@")[0]  # Use email prefix as username

            # Send the OTP email
            generic_send_mail.delay(
                recipient=email,
                title="OTP",
                payload={
                    "otp_code": otp,
                    "name": "Student",
                    "site_url": "https://aimyai.com",
                    "unsubscribe_url": "https://aimyai.com",
                },
            )

            # send_otp_email.delay(
            #     recipient_email=email,
            #     otp_code=otp,
            #     user={
            #         "first_name": "Student",
            #         "username": "Student"
            #     },
            #     site_url=request.build_absolute_uri("/")[:-1] if request else None,
            # )
        except Exception as e:
            # Log the error but don't fail the request
            from loguru import logger

            logger.error(f"Failed to send OTP email to {email}: {str(e)}")

        return self.validated_data["email"]


class VerifyOTPSerializer(serializers.Serializer):
    email = serializers.EmailField(required=False)
    otp = serializers.CharField(max_length=6, min_length=6, required=True)
    token: str = ""

    def verify_otp(self, request) -> str:
        """Send email Generate OTP and key and saves them in user account,
        Sends email with an otp to user for Account Activation"""
        # email = self.validated_data["email"]
        email = self.validated_data.get("email", None)
        otp = self.validated_data["otp"]

        print("== verify data: ", email, otp)
        # get token details from cache for a maximum of 24 hours
        cache_otp_value = cache.get(f"otp/email/{email}")

        if cache_otp_value != otp:
            raise exceptions.InvalidOTPException()
        return self.token


class ProfileImageUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating user profile image"""

    class Meta:
        model = CustomUser
        fields = ["profile_image"]

    def validate_profile_image(self, value):
        """Validate uploaded profile image"""
        if value:
            # Check file size (max 5MB)
            max_size = 5 * 1024 * 1024  # 5MB
            if value.size > max_size:
                raise serializers.ValidationError("Image size must be less than 5MB")

            # Check file type
            allowed_types = ["jpg", "jpeg", "png", "gif", "webp"]
            file_extension = value.name.split(".")[-1].lower()
            if file_extension not in allowed_types:
                raise serializers.ValidationError(
                    f"Image type must be one of: {', '.join(allowed_types)}"
                )

        return value

    def update(self, instance, validated_data):
        """Update user profile image"""
        # Delete old profile image if it exists and a new one is being uploaded
        if "profile_image" in validated_data and instance.profile_image:
            try:
                # Delete the old file from storage
                instance.profile_image.delete(save=False)
            except Exception as e:
                # Log error but don't fail the update
                logger.error(f"Failed to delete old profile image: {e}")

        return super().update(instance, validated_data)


class UserProfileUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating user profile information (excluding sensitive fields)"""

    class Meta:
        model = CustomUser
        fields = [
            "first_name",
            "last_name",
            "phone_number",
            "year_of_study",
            "profile_image",
        ]

    def validate_profile_image(self, value):
        """Validate uploaded profile image"""
        if value:
            # Check file size (max 5MB)
            max_size = 5 * 1024 * 1024  # 5MB
            if value.size > max_size:
                raise serializers.ValidationError("Image size must be less than 5MB")

            # Check file type
            allowed_types = ["jpg", "jpeg", "png", "gif", "webp"]
            file_extension = value.name.split(".")[-1].lower()
            if file_extension not in allowed_types:
                raise serializers.ValidationError(
                    f"Image type must be one of: {', '.join(allowed_types)}"
                )

        return value

    def update(self, instance, validated_data):
        """Update user profile with proper image handling"""
        # Handle profile image deletion and replacement
        if "profile_image" in validated_data and instance.profile_image:
            try:
                # Delete the old file from storage
                instance.profile_image.delete(save=False)
            except Exception as e:
                # Log error but don't fail the update
                logger.error(f"Failed to delete old profile image: {e}")

        return super().update(instance, validated_data)
