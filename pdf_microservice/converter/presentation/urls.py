from django.urls import path
from .views import ConvertToPDFView

urlpatterns = [
    path('convert/', ConvertToPDFView.as_view(), name='convert_to_pdf'),
]
