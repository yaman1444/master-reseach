from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from django.core.files.uploadedfile import SimpleUploadedFile
import io

class ConvertToPDFViewTest(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.url = reverse('convert_to_pdf')

    def test_upload_no_file(self):
        response = self.client.post(self.url, format='multipart')
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.data)
        self.assertEqual(response.data['error'], "Aucun fichier n'a été fourni sous la clé 'file'.")

    def test_upload_invalid_extension(self):
        # Création d'un faux fichier texte
        invalid_file = SimpleUploadedFile(
            "test_file.txt",
            b"Ceci est un test",
            content_type="text/plain"
        )
        response = self.client.post(self.url, {'file': invalid_file}, format='multipart')
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.data)
        self.assertIn("Seuls les fichiers .docx sont supportés", response.data['error'])
