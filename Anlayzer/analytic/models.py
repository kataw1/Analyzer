from django.db import models

# for slug to use /+-@#$%^
from django.core.validators import RegexValidator 
from django.utils.text import slugify
from django.contrib.auth.models import User ,auth

# Create your models here.

class Service(models.Model):
    CATEGORY_CHOICES = [
        ('AI', 'Artificial Intelligence'),
        ('DataAnalysis', 'Data Analysis'),
        ('DataScience', 'Data Science'),
        ('Other', 'Other'),  # Example of a default category
    ]


    title = models.CharField(max_length=100)
    description = models.TextField()
    url = models.URLField(null=True)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, default='Other')
    notes  = models.TextField(null=True)
    def __str__(self) -> str:
        return self.title

    

class Contact(models.Model):
    email = models.EmailField()
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.email
    

class Favorite(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    service = models.ForeignKey(Service, on_delete=models.CASCADE)

    def is_favorite_for_user(self, user):
        """
        Check if this service is a favorite for the given user.
        """
        return self.user == user

    def __str__(self) -> str:
        return f"{self.user.username} - {self.service.title}"    
    


# models.py

from django.utils import timezone
from django.utils.crypto import get_random_string

class EmailVerification(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    token = models.CharField(max_length=100)
    created_at = models.DateTimeField(default=timezone.now)

    def save(self, *args, **kwargs):
        if not self.token:
            self.token = get_random_string(length=50)
        return super().save(*args, **kwargs)


class Premium_Service(models.Model):
    CATEGORY_CHOICES = [
        ('AI', 'Artificial Intelligence'),
        ('DataAnalysis', 'Data Analysis'),
        ('DataScience', 'Data Science'),
        ('Other', 'Other'),  # Example of a default category
    ]


    title = models.CharField(max_length=100)
    description = models.TextField()
    url = models.URLField(null=True)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, default='Other')
    notes  = models.TextField(null=True)
    def __str__(self) -> str:
        return self.title