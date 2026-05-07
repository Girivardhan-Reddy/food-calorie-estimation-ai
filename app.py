from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
import torch
from PIL import Image
import json
import plotly
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from functools import wraps

# Import the enhanced detector
from model_utils import FoodCalorieEstimator

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-super-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///food_calorie.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please login to access this page.'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# ---------------------------- Database Models ----------------------------
class User(UserMixin, db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    analyses = db.relationship('FoodAnalysis', backref='user', lazy=True, cascade='all, delete-orphan')
    daily_logs = db.relationship('DailyCalorieLog', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class FoodAnalysis(db.Model):
    """Food analysis history model"""
    __tablename__ = 'food_analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    food_name = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    estimated_weight = db.Column(db.Float, nullable=False)
    estimated_calories = db.Column(db.Float, nullable=False)
    portion_ratio = db.Column(db.Float, nullable=False)
    calories_per_100g = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(200))
    processed_image_path = db.Column(db.String(200))
    bbox_coordinates = db.Column(db.Text)  # Store as JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'food_name': self.food_name,
            'confidence': round(self.confidence, 2),
            'estimated_weight': round(self.estimated_weight, 1),
            'estimated_calories': round(self.estimated_calories, 1),
            'portion_ratio': round(self.portion_ratio, 1),
            'calories_per_100g': self.calories_per_100g,
            'image_path': self.image_path,
            'processed_image_path': self.processed_image_path,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

class DailyCalorieLog(db.Model):
    """Daily calorie tracking model"""
    __tablename__ = 'daily_calorie_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    date = db.Column(db.Date, default=datetime.utcnow().date)
    total_calories = db.Column(db.Float, default=0)
    goal_calories = db.Column(db.Float, default=2000)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------------------- Initialize Model ----------------------------
MODEL_PATH = "efficientnet_food101.pth"
DATA_DIR = "split_data"

try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    estimator = FoodCalorieEstimator(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        device=device
    )
    print("Model loaded successfully!")
    print(f"Number of classes: {len(estimator.class_names)}")
except Exception as e:
    print(f"Error loading model: {e}")
    estimator = None

# ---------------------------- Helper Functions ----------------------------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_image_with_bbox(image, filename):
    """Save image with bounding box"""
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    image.save(processed_path)
    return processed_path

def update_daily_calories(user_id, calories):
    """Update daily calorie log"""
    today = datetime.utcnow().date()
    daily_log = DailyCalorieLog.query.filter_by(user_id=user_id, date=today).first()
    
    if not daily_log:
        daily_log = DailyCalorieLog(user_id=user_id, date=today, total_calories=calories)
        db.session.add(daily_log)
    else:
        daily_log.total_calories += calories
    
    db.session.commit()
    return daily_log

def create_calorie_chart(user_id, days=7):
    """Create calorie intake chart"""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)
    
    logs = DailyCalorieLog.query.filter(
        DailyCalorieLog.user_id == user_id,
        DailyCalorieLog.date >= start_date,
        DailyCalorieLog.date <= end_date
    ).order_by(DailyCalorieLog.date).all()
    
    dates = []
    calories = []
    goals = []
    
    current_date = start_date
    while current_date <= end_date:
        log = next((l for l in logs if l.date == current_date), None)
        dates.append(current_date.strftime('%Y-%m-%d'))
        calories.append(log.total_calories if log else 0)
        goals.append(log.goal_calories if log else 2000)
        current_date += timedelta(days=1)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates,
        y=calories,
        name='Calories Consumed',
        marker_color='rgb(59, 130, 246)',
        text=[f'{c:.0f} kcal' for c in calories],
        textposition='auto',
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=goals,
        name='Daily Goal',
        mode='lines',
        line=dict(color='rgb(16, 185, 129)', width=3, dash='dash'),
    ))
    
    fig.update_layout(
        title=f'Calorie Intake - Last {days} Days',
        xaxis_title='Date',
        yaxis_title='Calories (kcal)',
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_food_category_chart(user_id):
    """Create food category distribution chart"""
    analyses = FoodAnalysis.query.filter_by(user_id=user_id).order_by(
        FoodAnalysis.created_at.desc()
    ).limit(50).all()
    
    categories = {}
    for analysis in analyses:
        category = analysis.food_name.split()[0] if analysis.food_name else 'Other'
        categories[category] = categories.get(category, 0) + analysis.estimated_calories
    
    fig = go.Figure(data=[go.Pie(
        labels=list(categories.keys()),
        values=list(categories.values()),
        hole=.3,
        marker_colors=px.colors.qualitative.Set3
    )])
    
    fig.update_layout(
        title='Calorie Distribution by Food Category',
        template='plotly_dark',
        showlegend=True
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_weight_trend_chart(user_id):
    """Create food weight trend chart"""
    analyses = FoodAnalysis.query.filter_by(user_id=user_id).order_by(
        FoodAnalysis.created_at.desc()
    ).limit(30).all()
    
    dates = [a.created_at.strftime('%Y-%m-%d %H:%M') for a in reversed(analyses)]
    weights = [a.estimated_weight for a in reversed(analyses)]
    foods = [a.food_name for a in reversed(analyses)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=weights,
        mode='lines+markers',
        name='Food Weight',
        line=dict(color='rgb(245, 158, 11)', width=2),
        marker=dict(size=8),
        text=foods,
        hovertemplate='<b>%{text}</b><br>Weight: %{y:.1f}g<br>Date: %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Food Weight Trends',
        xaxis_title='Date & Time',
        yaxis_title='Weight (g)',
        template='plotly_dark',
        hovermode='x'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# ---------------------------- Authentication Routes ----------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required', 'error')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    # Redirect if already logged in
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        
        user = User.query.filter_by(username=username).first()
        
        if not user or not user.check_password(password):
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
        
        login_user(user, remember=remember)
        
        # Create daily log if not exists
        today = datetime.utcnow().date()
        daily_log = DailyCalorieLog.query.filter_by(user_id=user.id, date=today).first()
        if not daily_log:
            daily_log = DailyCalorieLog(user_id=user.id, date=today)
            db.session.add(daily_log)
            db.session.commit()
        
        flash('Login successful!', 'success')
        return redirect(url_for('index'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# ---------------------------- Main Application Routes ----------------------------
@app.route('/')
@login_required
def index():
    """Main dashboard page"""
    # Get today's stats
    today = datetime.utcnow().date()
    daily_log = DailyCalorieLog.query.filter_by(user_id=current_user.id, date=today).first()
    
    # If no daily log exists, create one
    if not daily_log:
        daily_log = DailyCalorieLog(user_id=current_user.id, date=today)
        db.session.add(daily_log)
        db.session.commit()
    
    # Get recent analyses
    recent_analyses = FoodAnalysis.query.filter_by(
        user_id=current_user.id
    ).order_by(
        FoodAnalysis.created_at.desc()
    ).limit(10).all()
    
    # Get statistics
    total_analyses = FoodAnalysis.query.filter_by(user_id=current_user.id).count()
    total_calories = db.session.query(db.func.sum(FoodAnalysis.estimated_calories)).filter(
        FoodAnalysis.user_id == current_user.id
    ).scalar() or 0
    
    avg_confidence = db.session.query(db.func.avg(FoodAnalysis.confidence)).filter(
        FoodAnalysis.user_id == current_user.id
    ).scalar() or 0
    
    # Create charts
    calorie_chart = create_calorie_chart(current_user.id)
    category_chart = create_food_category_chart(current_user.id)
    weight_chart = create_weight_trend_chart(current_user.id)
    
    return render_template(
        'index.html',
        user=current_user,
        daily_log=daily_log,
        recent_analyses=[a.to_dict() for a in recent_analyses],
        total_analyses=total_analyses,
        total_calories=round(total_calories, 0),
        avg_confidence=round(avg_confidence, 1),
        calorie_chart=calorie_chart,
        category_chart=category_chart,
        weight_chart=weight_chart
    )

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Make prediction on uploaded image - supports multiple food items"""
    if estimator is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = secure_filename(file.filename)
            original_filename = f"{current_user.id}_{timestamp}_{filename}"
            original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            file.save(original_filepath)
            
            # Make prediction - now returns multiple results
            result = estimator.predict(original_filepath)
            
            if not result or not result.get('results'):
                return jsonify({'error': 'No food items detected in the image'}), 400
            
            # Save processed image
            processed_filename = f"processed_{current_user.id}_{timestamp}_{filename}"
            processed_filepath = save_image_with_bbox(
                result['image_with_bbox'],
                processed_filename
            )
            
            # Save each detected food item as a separate analysis in database
            for item in result['results']:
                analysis = FoodAnalysis(
                    user_id=current_user.id,
                    food_name=item['food_name'],
                    confidence=item['confidence'],
                    estimated_weight=item['estimated_weight'],
                    estimated_calories=item['estimated_calories'],
                    portion_ratio=item['portion_ratio'],
                    calories_per_100g=item['calories_per_100g'],
                    image_path=f'/static/uploads/{original_filename}',
                    processed_image_path=f'/static/processed/{processed_filename}',
                    bbox_coordinates=json.dumps(item['bbox'])
                )
                db.session.add(analysis)
            
            db.session.commit()
            
            # Update daily calories with total from all detected items
            total_calories = result['total_calories']
            update_daily_calories(current_user.id, total_calories)
            
            # Format predictions for JSON response
            predictions = []
            for item in result['results']:
                bbox_list = [
                    [int(item['bbox'][0][0]), int(item['bbox'][0][1])],
                    [int(item['bbox'][1][0]), int(item['bbox'][1][1])]
                ]
                predictions.append({
                    'food_name': item['food_name'],
                    'confidence': item['confidence'],
                    'estimated_weight': item['estimated_weight'],
                    'estimated_calories': item['estimated_calories'],
                    'portion_ratio': item['portion_ratio'],
                    'calories_per_100g': item['calories_per_100g'],
                    'bbox': bbox_list
                })
            
            return jsonify({
                'success': True,
                'original_image': f'/static/uploads/{original_filename}',
                'processed_image': f'/static/processed/{processed_filename}',
                'total_calories': total_calories,
                'items_detected': len(result['results']),
                'predictions': predictions
            })
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/history')
@login_required
def history():
    """View all analysis history"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    analyses = FoodAnalysis.query.filter_by(
        user_id=current_user.id
    ).order_by(
        FoodAnalysis.created_at.desc()
    ).paginate(page=page, per_page=per_page)
    
    # Group analyses by timestamp (same processed_image_path indicates same upload)
    grouped_analyses = {}
    for analysis in analyses.items:
        key = analysis.processed_image_path
        if key not in grouped_analyses:
            grouped_analyses[key] = {
                'timestamp': analysis.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'image_path': analysis.image_path,
                'processed_image_path': analysis.processed_image_path,
                'items': []
            }
        grouped_analyses[key]['items'].append(analysis.to_dict())
    
    return jsonify({
        'analyses': list(grouped_analyses.values()),
        'total_pages': analyses.pages,
        'current_page': analyses.page,
        'total_items': analyses.total
    })

@app.route('/delete_analysis/<int:analysis_id>', methods=['DELETE'])
@login_required
def delete_analysis(analysis_id):
    """Delete a specific analysis"""
    analysis = FoodAnalysis.query.filter_by(
        id=analysis_id, 
        user_id=current_user.id
    ).first_or_404()
    
    # Check if this is the only analysis using these images
    same_images = FoodAnalysis.query.filter_by(
        image_path=analysis.image_path,
        processed_image_path=analysis.processed_image_path
    ).count()
    
    # Delete image files only if this is the last analysis using them
    if same_images == 1:
        try:
            if analysis.image_path:
                # Remove leading slash and get full path
                img_path = analysis.image_path.lstrip('/')
                full_img_path = os.path.join(app.root_path, img_path)
                if os.path.exists(full_img_path):
                    os.remove(full_img_path)
            
            if analysis.processed_image_path:
                proc_path = analysis.processed_image_path.lstrip('/')
                full_proc_path = os.path.join(app.root_path, proc_path)
                if os.path.exists(full_proc_path):
                    os.remove(full_proc_path)
        except Exception as e:
            print(f"Error deleting files: {e}")
    
    db.session.delete(analysis)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Analysis deleted successfully'})

@app.route('/set_calorie_goal', methods=['POST'])
@login_required
def set_calorie_goal():
    """Set daily calorie goal"""
    goal = request.json.get('goal', 2000)
    
    today = datetime.utcnow().date()
    daily_log = DailyCalorieLog.query.filter_by(user_id=current_user.id, date=today).first()
    
    if not daily_log:
        daily_log = DailyCalorieLog(user_id=current_user.id, date=today, goal_calories=goal)
        db.session.add(daily_log)
    else:
        daily_log.goal_calories = goal
    
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Calorie goal updated successfully'})

@app.route('/get_charts')
@login_required
def get_charts():
    """Get updated charts data"""
    days = request.args.get('days', 7, type=int)
    
    return jsonify({
        'calorie_chart': create_calorie_chart(current_user.id, days),
        'category_chart': create_food_category_chart(current_user.id),
        'weight_chart': create_weight_trend_chart(current_user.id)
    })

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    total_analyses = FoodAnalysis.query.filter_by(user_id=current_user.id).count()
    total_calories = db.session.query(db.func.sum(FoodAnalysis.estimated_calories)).filter(
        FoodAnalysis.user_id == current_user.id
    ).scalar() or 0
    
    # Get most common food (accounting for multiple detections)
    favorite_food = db.session.query(
        FoodAnalysis.food_name, 
        db.func.count(FoodAnalysis.food_name).label('count')
    ).filter(
        FoodAnalysis.user_id == current_user.id
    ).group_by(
        FoodAnalysis.food_name
    ).order_by(
        db.desc('count')
    ).first()
    
    # Get average items per upload
    total_uploads = db.session.query(
        db.func.count(db.func.distinct(FoodAnalysis.processed_image_path))
    ).filter(
        FoodAnalysis.user_id == current_user.id
    ).scalar() or 1
    
    avg_items_per_upload = round(total_analyses / total_uploads, 1) if total_uploads > 0 else 0
    
    return jsonify({
        'username': current_user.username,
        'email': current_user.email,
        'member_since': current_user.created_at.strftime('%B %d, %Y'),
        'total_analyses': total_analyses,
        'total_calories': round(total_calories, 0),
        'favorite_food': favorite_food[0] if favorite_food else 'None',
        'avg_calories_per_meal': round(total_calories / total_analyses, 1) if total_analyses > 0 else 0,
        'avg_items_per_upload': avg_items_per_upload
    })

@app.route('/health')
def health_check():
    if estimator is not None:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

# Create database tables
with app.app_context():
    db.create_all()
    print("Database tables created successfully!")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)