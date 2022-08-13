from django.http import JsonResponse
from rest_framework import viewsets
from rest_framework.views import APIView
from io import BytesIO

from PIL import Image
from django.http import JsonResponse
from rest_framework import serializers
import numpy as np
import base64


from photo_editor.models.image import ImageModel
from photo_editor.serializers.image import ImageModelSerializer, ImageCreateSerializer
from photo_editor.utils.maths import Transformations


class ImagePreviewView(APIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformations = {
            'GRADIENT': Transformations.grad,
            'HIST_NORMALIZE': Transformations.normalize,
            'BLUR': Transformations.gaussian_blur,
            'SHARPEN': Transformations.laplacian_sharpening,
            'NEGATIVE': Transformations.negative,
        }

    def post(self, request):
        img = request.data['img']
        format, imgstr = img.split(';base64,')
        image = np.asarray(Image.open(BytesIO(base64.b64decode(imgstr))))
        initial_shape = image.shape
        transformations = request.data.get('transformations', [])
        new_image = image
        for transformation in transformations:
            new_image = self.transformations[transformation](new_image)
        new_image = (((new_image - new_image.min()) / (new_image.max() - new_image.min())) * 255.9).astype(np.uint8)
        new_image = Image.fromarray(new_image)
        buffered = BytesIO()
        new_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return JsonResponse({
            "img": 'data:image/jpeg;base64,' + img_str
        })


class ImageViewSet(viewsets.ModelViewSet):
    queryset = ImageModel.objects.all()
    serializer_class = ImageModelSerializer
    lookup_field = 'sid'
    model = ImageModel

    def get_serializer_class(self):
        if self.action in {'create', 'partial_update', 'update'}:
            return ImageCreateSerializer
        else:
            return self.serializer_class
