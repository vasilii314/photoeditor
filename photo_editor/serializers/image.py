import os
import uuid
from io import BytesIO

from PIL import Image
from django.conf import settings
from django.http import JsonResponse
from rest_framework import serializers
import numpy as np
import base64

from photo_editor.models.image import ImageModel
from photo_editor.utils.maths import Transformations


class ImageCreateSerializer(serializers.ModelSerializer):

    img = serializers.CharField()

    transformations = serializers.ListField(required=False)

    class Meta:
        model = ImageModel
        fields = ['sid', 'img', 'url', 'transformations']

    def validate_transformations(self, value):
        return value

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformations = {
            'GRADIENT': Transformations.grad,
            'HIST_NORMALIZE': Transformations.normalize,
            'BLUR': Transformations.gaussian_blur,
            'SHARPEN': Transformations.laplacian_sharpening,
            'NEGATIVE': Transformations.negative,
        }

    def create(self, validated_data):
        img = validated_data.get('img')
        format, imgstr = img.split(';base64,')
        image = np.asarray(Image.open(BytesIO(base64.b64decode(imgstr))))
        transformations = validated_data.get('transformations', [])
        new_image = image
        for transformation in transformations:
            new_image = self.transformations[transformation](new_image)
        new_image = (((new_image - new_image.min()) / (new_image.max() - new_image.min())) * 255.9).astype(np.uint8)
        new_image = Image.fromarray(new_image)
        if new_image.mode != 'RGB':
            new_image = new_image.convert('RGB')
        image_name = f'{str(uuid.uuid4())}.png'
        path_to_image = f'{os.path.join(settings.MEDIA_ROOT, image_name)}'
        new_image.save(path_to_image, format="png")
        image = ImageModel.objects.create(img=path_to_image, url=f'http://localhost:8000/uploads/{image_name}')
        return image


class ImageModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageModel
        exclude = []

