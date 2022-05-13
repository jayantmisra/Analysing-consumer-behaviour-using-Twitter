# Generated by Django 4.0.2 on 2022-02-06 16:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('login', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Student',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('account', models.CharField(max_length=50, unique=True, verbose_name='account')),
                ('password', models.CharField(max_length=100, verbose_name='password')),
                ('email', models.EmailField(max_length=254, verbose_name='Email')),
                ('registerTime', models.DateTimeField(auto_now=True, verbose_name='registerTime')),
            ],
        ),
    ]
