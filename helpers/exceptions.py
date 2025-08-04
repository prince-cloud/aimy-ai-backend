from config.exceptions import BaseException


class GeneralException(BaseException):
    status_code = 400
    default_code = 100
    default_detail = "Sorry an error occured"


class LoginException(BaseException):
    status_code = 400
    default_code = 101
    default_detail = "Invalid username or password"


class AccountExistsException(BaseException):
    status_code = 400
    default_code = 102
    default_detail = "A user with the provided details already exists."


class UserDoesNotExistException(BaseException):
    status_code = 400
    default_code = 103
    default_detail = "A user does not exist with the given credentials"


class InactiveAccountException(BaseException):
    status_code = 400
    default_code = 104
    default_detail = (
        "Account not activated, an OTP has been sent "
        "to your email, please verify your account."
    )


class EmailDoesNotExistsException(BaseException):
    status_code = 400
    default_code = 105
    default_detail = "No user account found for the provided email"


class UsernameDoesNotExistsException(BaseException):
    status_code = 400
    default_code = 106
    default_detail = "No user account found for the provided username"


class AccountDoesNotExistException(BaseException):
    status_code = 400
    default_code = 107
    default_detail = "Account not found"


class AccountAlreadyVerifiedException(BaseException):
    status_code = 400
    default_code = 108
    default_detail = "Account already verified or active"


class EmailAlreadyVerifiedException(BaseException):
    status_code = 400
    default_code = 109
    default_detail = "Email already verified or active"


class EmailNotVerifiedException(BaseException):
    status_code = 400
    default_code = 110
    default_detail = "Email is unverified"


class InvalidOTPException(BaseException):
    status_code = 400
    default_code = 111
    default_detail = "OTP is invalid"


class OTPExpiredException(BaseException):
    status_code = 400
    default_code = 112
    default_detail = "The provided OTP is expired"


class EmailOrUsernameRequiredException(BaseException):
    status_code = 400
    default_code = 113
    default_detail = "At least email or username is required"


class PasswordsDoNotMatchException(BaseException):
    default_code = 114
    default_detail = "The two password fields didn't match."


class InvalidPasswordException(BaseException):
    default_code = 115
    default_detail = "Invalid passwords. Please use prescribed format"


class EmailAlreadyInUseException(BaseException):
    default_code = 116
    default_detail = "The provided email is already in use."


class ProvideUsernameOrPasswordException(BaseException):
    status_code = 400
    default_code = 117
    default_detail = "Provide username or password"


class ChangePasswordException(BaseException):
    status_code = 400
    default_code = 118
    default_detail = "Please reset your password."


class PhoneNumberAlreadyInUseException(BaseException):
    status_code = 400
    default_code = 119
    default_detail = "The provided phone number is already in use."


class InvalidRegistrationToken(BaseException):
    status_code = 400
    default_code = 120
    default_detail = "Invalid account registration token"


class InvalidToken(BaseException):
    status_code = 400
    default_code = 121
    default_detail = "The provided token is invalid."


class InvalidCredentials(BaseException):
    status_code = 400
    default_code = 122
    default_detail = "Invalid Credentials provided"


class TooManyLoginAttemptsException(BaseException):
    status_code = 400
    default_code = 123
    default_detail = "Too many login attempt, please try again, in the next 5 minutes."


class TooManyAttempt(BaseException):
    status_code = 400
    default_code = 124
    default_detail = "Too many attempt, please try again in the next 5 minutes."
