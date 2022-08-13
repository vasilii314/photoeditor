import uuid

from django.db import models


class ImageModel(models.Model):
    sid = models.UUIDField('sid', primary_key=True, default=uuid.uuid4, editable=False)
    img = models.ImageField(null=True, upload_to='images')
    url = models.URLField(null=True, blank=True)


