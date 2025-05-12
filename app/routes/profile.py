from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.models.user import User
from app.utils.file_handler import save_profile_image

profile_bp = Blueprint('profile', __name__)

@profile_bp.route('', methods=['GET'])
@jwt_required()
def get_profile():
    user_id = get_jwt_identity()
    current_app.logger.info(f"Fetching profile for user_id: {user_id}")
    user = User.query.get(user_id)
    current_app.logger.info(f"User data: {user.to_dict() if user else 'User not found'}")
    
    if not user:
        return jsonify({'message': 'User not found'}), 404
        
    response_data = user.to_dict()
    current_app.logger.info(f"Response data: {response_data}")
    return jsonify(response_data), 200

@profile_bp.route('', methods=['PUT'])
@jwt_required()
def update_profile():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'message': 'User not found'}), 404
    
    # Handle form data for file upload
    data = request.form.to_dict() if request.form else request.get_json()
    
    # Update username if provided
    if data.get('username') and data.get('username') != user.username:
        # Check if username already exists
        if User.query.filter_by(username=data.get('username')).first():
            return jsonify({'message': 'Username already exists'}), 400
        user.username = data.get('username')
    
    # Update email if provided
    if data.get('email') and data.get('email') != user.email:
        # Check if email already exists
        if User.query.filter_by(email=data.get('email')).first():
            return jsonify({'message': 'Email already exists'}), 400
        user.email = data.get('email')
    
    # Update password if provided
    if data.get('password'):
        user.set_password(data.get('password'))
    
    # Handle profile image upload
    if 'profile_image' in request.files:
        file = request.files['profile_image']
        file_path = save_profile_image(file)
        if file_path:
            user.profile_image = file_path
    
    db.session.commit()
    
    return jsonify({
        'message': 'Profile updated successfully',
        'user': user.to_dict()
    }), 200

@profile_bp.route('/users/by-role/<role>', methods=['GET'])
@jwt_required()
def get_users_by_role(role):
    """
    Fetch users filtered by role
    Valid roles: production_manager, marketing_manager, logistics_manager
    """
    current_app.logger.info(f"Fetching users with role: {role}")
    
    # Validate role is one of the allowed values
    valid_roles = ['production_manager', 'marketing_manager', 'logistics_manager']
    if role not in valid_roles:
        return jsonify({
            'message': f'Invalid role. Must be one of: {", ".join(valid_roles)}'
        }), 400
    
    # Get the current user to check permissions
    current_user_id = get_jwt_identity()
    current_user = User.query.get(current_user_id)
    
    if not current_user:
        return jsonify({'message': 'Unauthorized access'}), 401
    
    # Query users with the specified role
    users = User.query.filter_by(role=role).all()
    
    # Convert user objects to dictionaries
    users_data = [user.to_dict() for user in users]
    
    current_app.logger.info(f"Found {len(users_data)} users with role '{role}'")
    
    return jsonify({
        'role': role,
        'count': len(users_data),
        'users': users_data
    }), 200