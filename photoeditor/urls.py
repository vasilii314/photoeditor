"""photoeditor URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.contrib import admin
from django.urls import re_path
from django.urls import path
from django.views.static import serve

from photo_editor.views.image import ImagePreviewView, ImageViewSet

urlpatterns = [
    path('api/test/transformation/', ImagePreviewView.as_view(), name='test-transformation'),
    path('api/transformation/final/', ImageViewSet.as_view({'get': 'list', 'post': 'create'}), name='final-transformation'),
    re_path(r'^uploads/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
    path('admin/', admin.site.urls),
]
