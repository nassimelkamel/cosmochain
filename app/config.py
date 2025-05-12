import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'default-key-for-dev'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-key'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    PROFILE_IMAGES_FOLDER = os.path.join(UPLOAD_FOLDER, 'profile_images')
    # Angular project assets path for profile images
    ANGULAR_IMG_FOLDER = r"C:\Users\GIGABYTE\Desktop\material-dashboard-angular\src\assets\img"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
