
from django.db import models

# Create your models here...


class Question(models.Model):
    id = models.AutoField(primary_key=True)
    questionContent = models.CharField(max_length=200, unique=True)
    answer = models.CharField(max_length=50)


class Student(models.Model):
    id = models.AutoField(primary_key=True)
    account = models.CharField('account', max_length=50, unique=True)
    password = models.CharField('password', max_length=100)
    email = models.EmailField('Email')
    name = models.CharField('name', max_length=8, default=None)
    registerTime = models.DateTimeField('registerTime', auto_now=True)
    # occupation = models
