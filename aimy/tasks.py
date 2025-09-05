from celery import shared_task
from django.utils import timezone
from django.conf import settings
from loguru import logger

from .models import Reminder
from .services import ReminderService


@shared_task
def process_due_reminders():
    """
    Celery task to process all due reminders.
    This should be scheduled to run every minute or so.
    """
    try:
        reminder_service = ReminderService()
        due_reminders = reminder_service.get_due_reminders()

        if not due_reminders:
            logger.info("No due reminders to process")
            return {"processed": 0, "failed": 0}

        processed_count = 0
        failed_count = 0

        for reminder in due_reminders:
            try:
                # For now, we'll just mark reminders as sent
                # In a real implementation, you would:
                # 1. Send email notifications if delivery_method includes "email"
                # 2. Create in-app notifications if delivery_method includes "notification"
                # 3. Send push notifications, SMS, etc.

                # Simulate processing
                logger.info(f"Processing reminder {reminder.id}: {reminder.title}")

                # Here you could add actual notification sending logic
                send_reminder_notification(reminder)

                # Mark as sent
                reminder_service.mark_reminder_sent(reminder)
                processed_count += 1

            except Exception as e:
                logger.error(f"Failed to process reminder {reminder.id}: {e}")
                reminder_service.mark_reminder_failed(reminder, str(e))
                failed_count += 1

        logger.info(f"Processed {processed_count} reminders, {failed_count} failed")
        return {"processed": processed_count, "failed": failed_count}

    except Exception as e:
        logger.error(f"Error in process_due_reminders task: {e}")
        return {"error": str(e)}


def send_reminder_notification(reminder: Reminder):
    """
    Send notification for a reminder based on its delivery method.
    """
    try:
        if reminder.delivery_method in ["email", "email_sms", "all"]:
            # Send email notification
            send_reminder_email(reminder)

        if reminder.delivery_method in ["sms", "email_sms", "all"]:
            # Send SMS notification
            send_reminder_sms(reminder)

        if reminder.delivery_method in ["notification", "all"]:
            # Create in-app notification
            create_in_app_notification(reminder)

    except Exception as e:
        logger.error(f"Error sending notification for reminder {reminder.id}: {e}")
        raise


def send_reminder_email(reminder: Reminder):
    """
    Send email notification for a reminder using the existing generic_send_mail.
    """
    try:
        # Check if user has an email
        if not reminder.user.email:
            logger.warning(
                f"User {reminder.user.id} has no email for reminder {reminder.id}"
            )
            return

        # Import and use the existing generic_send_mail function
        from accounts.tasks import generic_send_mail

        # Prepare email data
        user_name = (
            reminder.user.get_full_name()
            or reminder.user.first_name
            or reminder.user.email.split("@")[0]
        )

        # Format the reminder datetime
        formatted_datetime = reminder.reminder_datetime.strftime(
            "%B %d, %Y at %I:%M %p"
        )

        # Prepare payload for email template
        payload = {
            "user_name": user_name,
            "reminder_title": reminder.title,
            "reminder_description": reminder.description
            or "No additional details provided.",
            "reminder_datetime": formatted_datetime,
            "reminder_date": reminder.reminder_datetime.strftime("%B %d, %Y"),
            "reminder_time": reminder.reminder_datetime.strftime("%I:%M %p"),
            "original_message": reminder.original_message,
            "app_name": "Aimy AI",
            "app_url": "https://aimyai.com",  # Update with your actual URL
        }

        # Send the email using the existing function
        result = generic_send_mail.delay(
            recipient=reminder.user.email,
            title=f"🔔 Reminder: {reminder.title}",
            payload=payload,
            template_type="reminder",
        )

        logger.info(
            f"Email sent successfully to {reminder.user.email} for reminder {reminder.id}"
        )

    except Exception as e:
        logger.error(f"Error sending reminder email for {reminder.id}: {e}")
        raise


def send_reminder_sms(reminder: Reminder):
    """
    Send SMS notification for a reminder using MNOTIFY service.
    """
    try:
        # Check if user has a phone number
        if not reminder.user.phone_number:
            logger.warning(
                f"User {reminder.user.email} has no phone number for SMS reminder {reminder.id}"
            )
            return

        # Check if MNOTIFY is configured
        if not all([settings.MNOTIFY_SENDER_ID, settings.MNOTIFY_API_KEY]):
            logger.error("MNOTIFY credentials not configured - cannot send SMS")
            raise Exception("SMS service not configured")

        # Format the SMS message
        message_body = format_sms_reminder_message(reminder)

        # Import and use the existing generic_send_sms function
        from accounts.tasks import generic_send_sms

        # Send the SMS using the existing function
        result = generic_send_sms.delay(
            to=reminder.user.phone_number, body=message_body
        )

        logger.info(
            f"SMS sent successfully to {reminder.user.phone_number} for reminder {reminder.id} via MNOTIFY"
        )

    except Exception as e:
        logger.error(f"Error sending SMS for reminder {reminder.id}: {e}")
        raise


def format_sms_reminder_message(reminder: Reminder) -> str:
    """
    Format the SMS message for a reminder.
    Keep it concise due to SMS character limits.
    """
    user_name = reminder.user.first_name or reminder.user.email.split("@")[0]

    # Format datetime in a readable way
    reminder_time = reminder.reminder_datetime.strftime("%I:%M %p on %b %d")

    message = f"🔔 Reminder: {reminder.title}\n"

    if reminder.description:
        # Truncate description if too long
        description = (
            reminder.description[:80] + "..."
            if len(reminder.description) > 80
            else reminder.description
        )
        message += f"{description}\n"

    message += f"Scheduled for: {reminder_time}\n"
    message += f"- Aimy AI"

    return message


def create_in_app_notification(reminder: Reminder):
    """
    Create in-app notification for a reminder.
    This is a placeholder - integrate with your notification system.
    """
    try:
        # For now, just log the notification that would be created
        logger.info(
            f"Would create in-app notification for user {reminder.user.id}: {reminder.title}"
        )

        # In a real implementation, you would create a notification record
        # or send a real-time notification through WebSockets, etc.

    except Exception as e:
        logger.error(f"Error creating in-app notification for {reminder.id}: {e}")
        raise


@shared_task
def cleanup_old_reminders(days_old: int = 30):
    """
    Clean up old completed/failed reminders to keep the database tidy.
    Run this task less frequently (e.g., weekly).
    """
    try:
        cutoff_date = timezone.now() - timezone.timedelta(days=days_old)

        old_reminders = Reminder.objects.filter(
            status__in=["sent", "failed", "cancelled"], updated_at__lt=cutoff_date
        )

        count = old_reminders.count()
        old_reminders.delete()

        logger.info(f"Cleaned up {count} old reminders")
        return {"cleaned_up": count}

    except Exception as e:
        logger.error(f"Error in cleanup_old_reminders task: {e}")
        return {"error": str(e)}
