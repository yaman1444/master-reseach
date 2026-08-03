import os
import uuid
import tempfile
from converter.domain.interfaces import DocumentConverterPort

class ConvertDocumentUseCase:
    def __init__(self, converter: DocumentConverterPort):
        self.converter = converter

    def execute(self, file_content: bytes, original_filename: str) -> str:
        """
        Reçoit le contenu binaire d'un fichier, le convertit, 
        et retourne le chemin absolu vers le PDF généré.
        """
        if not original_filename.lower().endswith(".docx"):
            raise ValueError("Seuls les fichiers .docx sont supportés par ce microservice.")

        # Création d'un dossier temporaire unique par requête
        temp_dir = tempfile.mkdtemp(prefix="pdf_converter_")
        
        # Nettoyage du nom de fichier pour éviter les failles
        safe_filename = f"{uuid.uuid4()}_{original_filename.replace(' ', '_')}"
        input_filepath = os.path.join(temp_dir, safe_filename)
        
        with open(input_filepath, 'wb') as f:
            f.write(file_content)
            
        # Exécution de l'adaptateur
        pdf_filepath = self.converter.convert_to_pdf(input_filepath, temp_dir)
        
        return pdf_filepath
