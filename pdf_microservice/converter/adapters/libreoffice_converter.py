import os
import subprocess
import logging
from converter.domain.interfaces import DocumentConverterPort

logger = logging.getLogger(__name__)

class LibreOfficeConverterAdapter(DocumentConverterPort):
    def convert_to_pdf(self, input_filepath: str, output_dir: str) -> str:
        if not os.path.exists(input_filepath):
            raise FileNotFoundError(f"Fichier introuvable: {input_filepath}")
        
        # Commande LibreOffice en mode headless
        command = [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            input_filepath
        ]
        
        try:
            logger.info(f"Lancement de la conversion LibreOffice: {' '.join(command)}")
            # timeout de 60s pour éviter qu'un process zombie ne bloque le serveur
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            logger.info(f"Conversion terminée: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur LibreOffice: {e.stderr}")
            raise Exception("Échec de la conversion du document par LibreOffice.")
        except subprocess.TimeoutExpired:
            logger.error("Timeout de la conversion LibreOffice.")
            raise Exception("La conversion a pris trop de temps (timeout).")
        
        base_name = os.path.basename(input_filepath)
        name_without_ext = os.path.splitext(base_name)[0]
        expected_pdf_path = os.path.join(output_dir, f"{name_without_ext}.pdf")
        
        if not os.path.exists(expected_pdf_path):
            raise Exception("Le fichier PDF attendu n'a pas été généré.")
            
        return expected_pdf_path
