from dotenv import load_dotenv
import boto3
import os
import logging

s3_client = boto3.client('s3', 
                         aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), 
                         aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'), 
                         region_name=os.getenv('AWS_REGION')
                        )
S3_BUCKET = 'aws-s3-nodejs-uploader'

def upload_to_s3(file_path):
    try:
        file_name = os.path.basename(file_path)

        s3_key = f"uploads/{file_name}"
        logging.info(f"Nom du fichier : {file_path}")
        logging.info(f"S3_BUCKET : {S3_BUCKET}")
        logging.info(f"s3_key : {s3_key}")
        result = s3_client.upload_file(file_path, S3_BUCKET, s3_key)
        logging.info(f"upload_file : {result}")

        s3_url = f"https://{S3_BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"
        logging.info(f"Image uploadée sur S3 : {s3_url}")

        return s3_url
    except Exception as e:
        logging.error(f"Erreur lors de l'upload sur S3 : {str(e)}")
        raise e

