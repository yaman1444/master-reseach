import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import FileResponse
from rest_framework.parsers import MultiPartParser, FormParser
import logging

from converter.adapters.libreoffice_converter import LibreOfficeConverterAdapter
from converter.application.use_cases import ConvertDocumentUseCase

logger = logging.getLogger(__name__)

class ConvertToPDFView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({"error": "Aucun fichier n'a été fourni sous la clé 'file'."}, status=status.HTTP_400_BAD_REQUEST)

        # Injection de dépendance manuelle (Clean Architecture)
        converter = LibreOfficeConverterAdapter()
        use_case = ConvertDocumentUseCase(converter)

        try:
            file_content = file_obj.read()
            pdf_path = use_case.execute(file_content, file_obj.name)
            
            # Renvoi du fichier binaire en stream
            response = FileResponse(open(pdf_path, 'rb'), content_type='application/pdf')
            
            # Nettoyage du nom pour le téléchargement
            original_base = os.path.splitext(file_obj.name)[0]
            download_name = f"{original_base}.pdf"
            response['Content-Disposition'] = f'attachment; filename="{download_name}"'
            
            return response

        except ValueError as ve:
            return Response({"error": str(ve)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Erreur lors de la conversion: {str(e)}")
            return Response({"error": "Une erreur interne est survenue lors de la conversion. Assurez-vous que LibreOffice est installé."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
