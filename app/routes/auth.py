from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from app import db
from app.models.user import User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['username', 'email', 'password', 'role']
    for field in required_fields:
        if not data.get(field):
            return jsonify({'message': f'{field} is required'}), 400
    
    # Validate role
    valid_roles = ['production_manager', 'marketing_manager', 'logistics_manager']
    if data.get('role') not in valid_roles:
        return jsonify({'message': f'Role must be one of: {", ".join(valid_roles)}'}), 400
    
    # Check if username or email already exists
    if User.query.filter_by(username=data.get('username')).first():
        return jsonify({'message': 'Username already exists'}), 400
        
    if User.query.filter_by(email=data.get('email')).first():
        return jsonify({'message': 'Email already exists'}), 400
    
    # Create new user
    user = User(
        username=data.get('username'),
        email=data.get('email'),
        role=data.get('role')
    )
    user.set_password(data.get('password'))
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({
        'message': 'User created successfully',
        'user': user.to_dict()
    }), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    # Validate required fields
    if not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Email and password are required'}), 400
    
    # Check if user exists
    user = User.query.filter_by(email=data.get('email')).first()
    if not user or not user.check_password(data.get('password')):
        return jsonify({'message': 'Invalid email or password'}), 401
    
    # Generate access token
    access_token = create_access_token(identity=str(user.id))
    
    return jsonify({
        'token': access_token,
        'user': user.to_dict()
    }), 200
