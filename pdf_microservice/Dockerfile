FROM python:3.12-slim

# Éviter la génération de .pyc et forcer l'affichage immédiat des logs
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Installation de LibreOffice et des polices indispensables pour garantir un bon rendu PDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    libreoffice-writer \
    fonts-liberation \
    fonts-dejavu \
    fonts-freefont-ttf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installation des dépendances Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copie du projet Django
COPY . /app/

# Exposition du port
EXPOSE 8000

# Commande par défaut pour le développement (pour la prod, utiliser Gunicorn)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
