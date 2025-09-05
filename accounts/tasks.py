from celery import shared_task
from django.conf import settings
import requests
from loguru import logger
from jinja2 import Environment, FileSystemLoader
import os
from typing import Dict
from django.template.loader import render_to_string
from django.core.mail import EmailMultiAlternatives
from django.utils.html import strip_tags


@shared_task
def generic_send_mail(
    recipient, title, payload: Dict[str, str] = {}, template_type: str = "user"
):
    """
    Send generic email using specified template type.

    Args:
        recipient: Email address of the recipient
        title: Email subject
        payload: Dictionary containing template variables
        template_type: Either "user" or "admin" to specify which template to use
    """
    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates"))
    )

    # Choose template based on type
    template_mapping = {
        "user": "otp_email.html",
        "admin": "otp_email.html",  # Can be different admin template
        "reminder": "reminder_email.html",
    }

    template_file = template_mapping.get(template_type, "otp_email.html")
    template = env.get_template(template_file)

    # Add current year to payload
    from datetime import datetime

    payload["current_year"] = datetime.now().year

    html_message = template.render(payload)
    logger.info(f"sending {template_type} email to {recipient}")
    try:
        base_url = "https://0qmusixj1f.execute-api.us-east-1.amazonaws.com/sendEmail"
        body = {
            "recipient": recipient,
            "subject": title,
            "body": html_message,
        }
        email_send = requests.post(
            base_url, json=body, headers={"Content-Type": "application/json"}
        )
        print("== response: ", email_send.text)
        return "Mail Sent"
    except Exception as e:
        logger.warning(f"An error occurred sending email {str(e)}")


@shared_task
def send_otp_email(recipient_email: str, otp_code: str, user, site_url: str = None):
    """
    Send OTP verification email using custom template with KNUST branding.

    Args:
        recipient_email: Email address of the recipient
        otp_code: The OTP code to send
        user: User object for personalization
        site_url: Website URL for footer links
    """
    try:
        # Prepare template context
        context = {
            "user": user,
            "otp_code": otp_code,
            "site_url": site_url or "https://aimyai.com",
            "unsubscribe_url": (
                f"{site_url or 'https://aimyai.com'}/unsubscribe" if site_url else "#"
            ),
        }

        # Render HTML template
        html_content = render_to_string("otp_email.html", context)

        # Render plain text template
        text_content = render_to_string("otp_email.txt", context)

        # Strip HTML tags for plain text
        text_content = strip_tags(text_content)

        # Create email message
        subject = "OTP Verification - Aimy AI"
        from_email = "noreply@aimyai.com"

        msg = EmailMultiAlternatives(
            subject=subject,
            body=text_content,
            from_email=from_email,
            to=[recipient_email],
        )

        # Attach HTML version
        msg.attach_alternative(html_content, "text/html")

        # Send email
        msg.send()

        logger.info(f"OTP email sent successfully to {recipient_email}")
        return "OTP Email Sent Successfully"

    except Exception as e:
        logger.error(f"Error sending OTP email to {recipient_email}: {str(e)}")
        return f"Error sending OTP email: {str(e)}"


@shared_task
def generic_send_sms(to, body):
    # return ""
    url = "https://apps.mnotify.net/smsapi"
    sender_id = settings.MNOTIFY_SENDER_ID
    api_key = settings.MNOTIFY_API_KEY

    params = {
        "key": api_key,
        "to": to,
        "msg": body,
        "sender_id": sender_id,
    }

    try:
        response = requests.post(url, params=params)
        logger.info(f"Response: {response.text}")
        response.raise_for_status()
        logger.info("Message sent successfully!")
        return response.json()  # Assuming API returns JSON
    except requests.RequestException as e:
        logger.error(f"An error occurred sending otp {e}")
        return {"status": "error", "message": str(e)}
