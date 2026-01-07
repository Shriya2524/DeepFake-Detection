"""
History API routes for managing user's deepfake detection history.
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from .models import db, AnalysisHistory

history_bp = Blueprint('history', __name__, url_prefix='/history')


@history_bp.route('', methods=['GET'])
@jwt_required()
def get_history():
    """Get all analysis history for the current user."""
    user_id = get_jwt_identity()
    
    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    # Limit per_page to reasonable bounds
    per_page = min(per_page, 100)
    
    # Query user's history, ordered by most recent first
    query = AnalysisHistory.query.filter_by(user_id=user_id).order_by(AnalysisHistory.created_at.desc())
    
    # Paginate results
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'history': [record.to_dict() for record in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
        'per_page': per_page
    }), 200


@history_bp.route('', methods=['POST'])
@jwt_required()
def save_history():
    """Save a new analysis result to history."""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Validate required fields
    required_fields = ['filename', 'type', 'result', 'confidence']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Validate type
    valid_types = ['image', 'video', 'Image', 'Video', 'ai-check', 'AI-Check']
    if data['type'] not in valid_types:
        return jsonify({'error': 'Type must be "image", "video", or "ai-check"'}), 400
    
    # Validate result
    valid_results = ['Real', 'Fake', 'AI-Generated', 'Natural']
    if data['result'] not in valid_results:
        return jsonify({'error': 'Result must be "Real", "Fake", "AI-Generated", or "Natural"'}), 400
    
    # Create new history record
    record = AnalysisHistory(
        user_id=user_id,
        filename=data['filename'],
        type=data['type'].capitalize(),  # Normalize to 'Image' or 'Video'
        result=data['result'],
        confidence=float(data['confidence']),
        details=data.get('details')  # Optional additional details
    )
    
    db.session.add(record)
    db.session.commit()
    
    return jsonify({
        'message': 'History saved successfully',
        'record': record.to_dict()
    }), 201


@history_bp.route('/<int:record_id>', methods=['DELETE'])
@jwt_required()
def delete_history(record_id):
    """Delete a specific history record."""
    user_id = get_jwt_identity()
    
    # Find the record
    record = AnalysisHistory.query.filter_by(id=record_id, user_id=user_id).first()
    
    if not record:
        return jsonify({'error': 'Record not found'}), 404
    
    db.session.delete(record)
    db.session.commit()
    
    return jsonify({'message': 'Record deleted successfully'}), 200


@history_bp.route('/clear', methods=['DELETE'])
@jwt_required()
def clear_history():
    """Clear all history for the current user."""
    user_id = get_jwt_identity()
    
    # Delete all records for this user
    deleted_count = AnalysisHistory.query.filter_by(user_id=user_id).delete()
    db.session.commit()
    
    return jsonify({
        'message': 'History cleared successfully',
        'deleted_count': deleted_count
    }), 200

