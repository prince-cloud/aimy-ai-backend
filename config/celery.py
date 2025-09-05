import os
from celery import Celery
import logging

from celery.schedules import crontab


logger = logging.getLogger(__name__)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("aimy-ai")
app.config_from_object("django.conf:settings", namespace="CELERY")

app.conf.beat_schedule = {
    # "permanently_delete_deactivated_accounts": {
    #     "task": "accounts.tasks.permanently_delete_deactivated_accounts",
    #     "schedule": crontab(
    #         minute="1",
    #         # hour="*/12",
    #     ),
    # },
    "process_due_reminders": {
        "task": "aimy.tasks.process_due_reminders",
        "schedule": crontab(minute="*"),  # Run every minute
    },
    "cleanup_old_reminders": {
        "task": "aimy.tasks.cleanup_old_reminders",
        "schedule": crontab(minute="0", hour="2"),  # Run daily at 2 AM
    },
}


app.autodiscover_tasks()
