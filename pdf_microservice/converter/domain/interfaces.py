from abc import ABC, abstractmethod

class DocumentConverterPort(ABC):
    @abstractmethod
    def convert_to_pdf(self, input_filepath: str, output_dir: str) -> str:
        """
        Convertit un fichier d'entrée en PDF et le sauvegarde dans output_dir.
        Retourne le chemin absolu du fichier PDF généré.
        """
        pass
