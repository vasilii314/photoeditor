# Generated by Django 4.0.6 on 2022-08-13 14:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('photo_editor', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagemodel',
            name='url',
            field=models.URLField(blank=True, null=True),
        ),
    ]
