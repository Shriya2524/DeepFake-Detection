"""
Database models for user authentication and analysis history.
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    """User model for authentication."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to analysis history
    analyses = db.relationship('AnalysisHistory', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class AnalysisHistory(db.Model):
    """Model to store user's deepfake detection history."""
    __tablename__ = 'analysis_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(10), nullable=False)  # 'image' or 'video'
    result = db.Column(db.String(10), nullable=False)  # 'Real' or 'Fake'
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    details = db.Column(db.JSON, nullable=True)  # Store additional detection details
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'filename': self.filename,
            'type': self.type,
            'result': self.result,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'date': self.created_at.strftime('%Y-%m-%d') if self.created_at else None,
            'time': self.created_at.strftime('%H:%M') if self.created_at else None,
            'details': self.details
        }

