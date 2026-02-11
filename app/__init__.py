from flask import Flask
from app.routes import image_analysis

def create_app():
    app = Flask(__name__)

    # Enregistrement des blueprints pour les routes
    app.register_blueprint(image_analysis)

    return app
