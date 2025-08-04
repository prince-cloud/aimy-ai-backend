import os
from celery import Celery
import logging
from celery.schedules import crontab


logger = logging.getLogger(__name__)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("paperless-branch")
app.config_from_object("django.conf:settings", namespace="CELERY")

app.conf.beat_schedule = {
    # "permanently_delete_deactivated_accounts": {
    #     "task": "accounts.tasks.permanently_delete_deactivated_accounts",
    #     "schedule": crontab(
    #         minute="1",
    #         # hour="*/12",
    #     ),
    # },
}


app.autodiscover_tasks()
