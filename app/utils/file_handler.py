import os
import uuid
from werkzeug.utils import secure_filename
from flask import current_app

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_profile_image(file):
    if not file or not allowed_file(file.filename):
        return None
        
    # Generate unique filename
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    
    # Save file
    file_path = os.path.join(current_app.config['PROFILE_IMAGES_FOLDER'], unique_filename)
    file.save(file_path)
    # Also save to Angular assets folder if configured
    angular_folder = current_app.config.get('ANGULAR_IMG_FOLDER')
    if angular_folder:
        angular_path = os.path.join(angular_folder, unique_filename)
        import shutil
        shutil.copy(file_path, angular_path)
    
    # Return relative path for storage in database
    if angular_folder:
        return f"/assets/img/{unique_filename}"
    return f"profile_images/{unique_filename}"
