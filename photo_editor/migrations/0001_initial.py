# Generated by Django 4.0.2 on 2022-07-26 15:17

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ImageModel',
            fields=[
                ('sid', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False, verbose_name='sid')),
                ('img', models.ImageField(null=True, upload_to='images')),
            ],
        ),
    ]
