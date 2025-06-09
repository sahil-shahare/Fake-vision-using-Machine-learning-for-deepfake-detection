from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import os
from torchvision import transforms
from PIL import Image
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import timm
from collections import OrderedDict
from datetime import datetime
import cv2
import random
import numpy as np

app = Flask(__name__)

# Generate a random secret key using os.urandom
app.secret_key = os.urandom(24).hex()
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configurations for SQLite
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'site.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Define the model matching the checkpoint structure
class CustomXceptionNet(nn.Module):
    def __init__(self):
        super(CustomXceptionNet, self).__init__()
        self.backbone = timm.create_model('xception', pretrained=False)
        num_ftrs = self.backbone.num_features
        self.backbone.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.backbone(x)

# Load the model
model = CustomXceptionNet()
checkpoint_path = "models/xception.pth"
state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

new_state_dict = OrderedDict()
for key, value in state_dict.items():
    if key.startswith("module.backbone."):
        new_key = key.replace("module.backbone.", "")
        new_state_dict[new_key] = value

model.backbone.load_state_dict(new_state_dict, strict=False)
model.eval()

# Preprocessing function
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

# Preprocessing function for individual video frames
def preprocess_frame(frame):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),  # Convert from NumPy array (frame) to PIL Image
        transforms.Resize((299, 299)),  # Resize to the input size of the model
        transforms.ToTensor(),  # Convert PIL Image to Tensor
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Normalize
    ])
    return preprocess(frame).unsqueeze(0)  # Add batch dimension

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# User model
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=False, unique=True)
    email = db.Column(db.String(255), nullable=False, unique=True)
    password = db.Column(db.String(500), nullable=False)

# Define the AnalysisResults model
class AnalysisResult(db.Model):
    __tablename__ = 'analysis_results'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    result = db.Column(db.String(10), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

# Index route
@app.route('/')
def index():
    return render_template('index.html')

# Singup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    message = None
    message_type = None
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            message = "Passwords do not match!"
            message_type = "danger"
            return render_template('signup.html', message=message, message_type=message_type)

        # Check if user already exists
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            message = "User with the same username or email already exists."
            message_type = "danger"
            return render_template('signup.html', message=message, message_type=message_type)

        # Hash the password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Add new user to the database
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        message = "Account created successfully! Please log in."
        message_type = "success"
        return render_template('signup.html', message=message, message_type=message_type)

    return render_template('signup.html', message=message, message_type=message_type)

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    message = None
    message_type = 'danger'
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            # Store user info in session
            session['user_id'] = user.id
            session['user_email'] = user.email
            session['username'] = user.username
            return redirect(url_for('dashboard'))  # Redirect to dashboard
        else:
            message = "Invalid email or password. Please try again."
    return render_template('login.html', message=message, message_type=message_type)

# Dashboard route
@app.route('/dashboard')
def dashboard():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session.get('username'))

# Upload Image route
@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('upload_image.html', error='No file selected.')

        file = request.files['image']
        if file.filename == '':
            return render_template('upload_image.html', error='No file selected.')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Debug: Ensure file is saved
            print(f"File saved at: {filepath}")

            # Run the detection model
            try:
                input_tensor = preprocess_image(filepath)
                print(f"Input tensor shape: {input_tensor.shape}")  # Debug: Log tensor shape

                with torch.no_grad():
                    # Run multiple times for stable predictions
                    outputs = [model(input_tensor) for _ in range(5)]  # Run 5 times
                    prob = torch.sigmoid(torch.stack(outputs)).mean().item()  # Average predictions
                    print(f"Confidence score: {prob:.2f}")  # Debug: Log confidence score

                prediction = 'FAKE' if prob > 0.5 else 'REAL'
                confidence = round(prob, 2)

                # Save the result to the database
                try:
                    user_id = session['user_id']
                    new_result = AnalysisResult(
                        user_id=user_id,
                        file_path=f'static/uploads/{filename}',
                        result=prediction,
                        confidence_score=confidence
                    )
                    db.session.add(new_result)
                    db.session.commit()
                    print("Result saved to database.")
                except Exception as e:
                    print(f"Database error: {e}")
                    return render_template('upload_image.html', error='Failed to save result to database.')

                return render_template(
                    'upload_image.html',
                    success=f'Prediction: {prediction}, Confidence: {confidence}',
                    image_url=f'uploads/{filename}'  # Corrected URL for static path
                )
            except Exception as e:
                # Debug: Log any errors during detection
                print(f"Error during detection: {e}")
                return render_template('upload_image.html', error='Error processing the image.')

        return render_template('upload_image.html', error='Unsupported file format.')

    return render_template('upload_image.html')

    
    
@app.route('/upload-video', methods=['GET', 'POST'])
def upload_video():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template('upload_video.html', error='No file selected.')

        file = request.files['video']
        if file.filename == '':
            return render_template('upload_video.html', error='No file selected.')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Prepare to save frames
            frame_dir = os.path.join(app.config['UPLOAD_FOLDER'], filename + "_frames")
            os.makedirs(frame_dir, exist_ok=True)

            try:
                cap = cv2.VideoCapture(filepath)
                frames = []
                confidences = []
                frame_paths = []
                frame_count = 0
                success = True

                while success:
                    success, frame = cap.read()
                    if success:
                        frame_count += 1
                        if frame_count % 30 == 0:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            input_tensor = preprocess_frame(rgb_frame)

                            with torch.no_grad():
                                output = model(input_tensor)
                                prob = torch.sigmoid(output).item()
                                confidences.append(round(prob, 2))

                            # Save the frame
                            frame_filename = f"frame_{frame_count}.jpg"
                            frame_save_path = os.path.join(frame_dir, frame_filename)
                            cv2.imwrite(frame_save_path, frame)
                            frame_paths.append(f"uploads/{filename}_frames/{frame_filename}")


                cap.release()

                if not confidences:
                    raise Exception("No frames captured from video.")

                avg_confidence = sum(confidences) / len(confidences)
                prediction = 'FAKE' if avg_confidence < 0.5 else 'REAL'
                confidence = round(avg_confidence, 2)

                # Save the result to the database
                user_id = session['user_id']
                new_result = AnalysisResult(
                    user_id=user_id,
                    file_path=f'static/uploads/{filename}',
                    result=prediction,
                    confidence_score=confidence
                )
                db.session.add(new_result)
                db.session.commit()

                # Frame + confidence for template
                frame_data = [{"path": p, "confidence": c} for p, c in zip(frame_paths, confidences)]

                return render_template(
                    'upload_video.html',
                    success=f'Prediction: {prediction}, Confidence: {confidence}',
                    video_url=f'uploads/{filename}',
                    frame_data=frame_data
                )
            except Exception as e:
                print(f"Error processing video: {e}")
                return render_template('upload_video.html', error='Error processing the video.')

        return render_template('upload_video.html', error='Unsupported file format.')

    return render_template('upload_video.html')


# History route
@app.route('/history')
def history():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Fetch analysis results for the logged-in user
    user_id = session['user_id']
    results = AnalysisResult.query.filter_by(user_id=user_id).order_by(AnalysisResult.uploaded_at.desc()).all()

    return render_template('history.html', results=results)

# Logout route
@app.route('/logout')
def logout():
    session.clear()  # Clear all session data
    return redirect(url_for('login'))

if __name__ == '__main__':
    # Run app
    app.run(debug=True)
