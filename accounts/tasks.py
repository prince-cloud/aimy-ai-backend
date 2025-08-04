from django.core.mail import EmailMessage
from celery import shared_task
from django.conf import settings
import requests


@shared_task
def generic_send_mail(message, recipient_list, title):
    from_email = settings.DEFAULT_FROM_EMAIL
    try:
        email = EmailMessage(
            subject=title,
            body=message,
            from_email=from_email,
            to=[recipient_list],
        )
        email.send()
        return "Mail Sent"
    except Exception as e:
        print("==== an error occured: ", str(e))


# @celery_app.task
@shared_task
def generic_send_sms(to, body):
    base_url = settings.SMS_API_URL
    username = settings.SMS_API_USERNAME
    password = settings.SMS_API_PASSWORD

    url = f"{base_url}/http/v2/sendsms/{username}/{password}/{to}/{body}"

    try:
        headers = {"Content-Type": "application/json"}
        requests.request("GET", url, headers=headers)
        return "OTP Sent"
    except:
        return "An exception occurred while sending account activation code."
