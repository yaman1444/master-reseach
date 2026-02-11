from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import logging

MODEL_PATH = os.path.join(os.getcwd(), 'scripts', 'my_model.keras')
model = None

def load_model_once():
    global model
    if model is None:
        logging.info("Chargement du modèle AI.")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Le modèle {MODEL_PATH} n'existe pas.")
        model = load_model(MODEL_PATH)
        logging.info("Modèle AI chargé avec succès.")
    return model

def analyze_image(image_path):
    try:
        model = load_model_once()
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array /= 255.0

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        class_mapping = {0: 'début', 1: 'avancé', 2: 'normal'}
        predicted_label = class_mapping.get(predicted_class, 'Inconnu')

        logging.info(f"Image analysée : Classe prédite - {predicted_label}, Confiance - {confidence}")
        return {'class': predicted_label, 'confidence': confidence}
    except Exception as e:
        logging.error(f"Erreur lors de l'analyse de l'image : {str(e)}")
        raise e
