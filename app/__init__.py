# Flask application factory
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_cors import CORS

from app.config import Config

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    CORS(app)
    
    # Create upload directories if they don't exist
    os.makedirs(app.config['PROFILE_IMAGES_FOLDER'], exist_ok=True)
    
    # Import and register blueprints
    from app.routes.auth import auth_bp
    from app.routes.profile import profile_bp
    from app.routes.chatbot import chatbot_bp
    from app.routes.cosmochain import cosmo_bp
    
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(profile_bp, url_prefix='/api/profile')
    
    app.register_blueprint(chatbot_bp, url_prefix='/api/chatbot')
    app.register_blueprint(cosmo_bp)
    
    return app
