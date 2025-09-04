from django.contrib.auth.models import AbstractUser
from django.db import models


class College(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class Department(models.Model):
    college = models.ForeignKey(
        College,
        on_delete=models.CASCADE,
        related_name="departments",
    )
    name = models.CharField(max_length=100)
    description = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class CustomUser(AbstractUser):
    class YearOfStudy(models.TextChoices):
        LEVEL_100 = "100"
        LEVEL_200 = "200"
        LEVEL_300 = "300"
        LEVEL_400 = "400"
        LEVEL_500 = "500"
        LEVEL_600 = "600"
        LEVEL_700 = "700"
        LEVEL_800 = "800"

    student_id = models.CharField(max_length=20, unique=True)
    index_number = models.CharField(max_length=20)
    department = models.ForeignKey(
        Department,
        on_delete=models.CASCADE,
        related_name="users",
        null=True,
        blank=True,
    )
    year_of_study = models.CharField(
        choices=YearOfStudy.choices,
        default=YearOfStudy.LEVEL_100,
        max_length=100,
    )

    def __str__(self):
        return self.email
