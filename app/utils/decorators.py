from functools import wraps
from flask import jsonify
from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request

def role_required(allowed_roles):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            verify_jwt_in_request()
            user_id = get_jwt_identity()
            from app.models.user import User
            user = User.query.get(user_id)
            
            if not user:
                return jsonify(message="User not found"), 404
                
            if user.role not in allowed_roles:
                return jsonify(message="Insufficient permissions"), 403
                
            return fn(*args, **kwargs)
        return wrapper
    return decorator
