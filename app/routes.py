

from flask import Blueprint, request, jsonify
from app.services.image_processing import analyze_image
from app.services.s3_service import upload_to_s3
import os
from PIL import Image
import logging

logging.basicConfig(filename='./logs/app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

image_analysis = Blueprint('image_analysis', __name__)

@image_analysis.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier envoyé'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    if not file.mimetype.startswith('image/'):
        return jsonify({'error': 'Le fichier sélectionné n\'est pas une image'}), 400

    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    image_path = os.path.join(uploads_dir, file.filename)
    file.save(image_path)

    try:
        img = Image.open(image_path)
        img.verify() 
        logging.info(f"L'image {file.filename} est valide.")
    except (IOError, SyntaxError) as e:
        os.remove(image_path)
        return jsonify({'error': 'Le fichier sauvegardé n\'est pas une image valide'}), 400

    try:
        logging.info(f"Début de l'analyse de l'image {file.filename}.")
        result = analyze_image(image_path)
        logging.info(f"Displaying image_path: {image_path}")
        logging.info(f"Analyse terminée avec succès : {result}")

        logging.info(f"Upload de l'image {file.filename} sur S3.")
        s3_url = upload_to_s3(image_path)
        logging.info(f"Image uploadée avec succès : {s3_url}")
        os.remove(image_path)
        logging.info(f"Le fichier temporaire {file.filename} a été supprimé.")

        return jsonify({
            'diagnostic': result['class'],
            'confidence': result['confidence'],
            'image_url': s3_url
        })
    except Exception as e:
        logging.error(f"Erreur lors du traitement de l'image {file.filename} : {str(e)}")
        return jsonify({'error': str(e)}), 502
