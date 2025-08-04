import random
import string
from accounts.models import CustomUser


def generate_otp(n):
    """
    generate random otp digits provided by length 'n'.
    """
    otp = "".join([str(random.randint(0, 9)) for _ in range(n)])
    return otp


def generate_reference_id(length=10):
    characters = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
    unique_string = "".join(random.choices(characters, k=length))
    return unique_string


def generate_access_code():
    """
    generate random access code
    """
    access_code = "".join([str(random.randint(1, 9)) for _ in range(10)])
    return access_code


def email_address_exists(email):
    return CustomUser.objects.filter(email__iexact=email).exists()
