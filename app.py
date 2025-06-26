from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
import os, sys, signal, atexit, shutil, random, torch, timm, cv2, requests
from datetime import datetime, timedelta
from torchvision import transforms
from PIL import Image
import numpy as np
import platform

# --- App Config ---
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default-fallback-key")
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'site.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# --- Device & Seed ---
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(500), nullable=False)

class AnalysisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    result = db.Column(db.String(10), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    return transform(img).unsqueeze(0).to(device)

# --- Model Loader ---
def download_model_if_needed():
    model_url = "https://drive.google.com/uc?id=1K67r5mpaul5TWlZ2EtXABNArUFSteY4x"
    model_path = "models/efficientnet_deepfake.pth"
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(model_path):
        print("Downloading model...")
        session = requests.Session()
        response = session.get(model_url, stream=True)
        token = next((v for k, v in response.cookies.items() if k.startswith("download_warning")), None)
        if token:
            response = session.get(model_url, params={'confirm': token}, stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)
        print("Model downloaded to", model_path)

def load_model():
    download_model_if_needed()
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1).to(device)
    weights = torch.load("models/efficientnet_deepfake.pth", map_location=device)
    model.load_state_dict(weights)
    model.eval()
    return model

model = load_model()

# --- Utils ---
def save_result(user_id, path, result, confidence):
    db.session.add(AnalysisResult(user_id=user_id, file_path=path, result=result, confidence_score=round(confidence, 2)))
    db.session.commit()

@app.context_processor
def inject_user():
    return dict(logged_in='user_id' in session, username=session.get('username'), user_email=session.get('user_email'))

# --- Routes ---
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and bcrypt.check_password_hash(user.password, request.form['password']):
            session.update({'user_id': user.id, 'username': user.username, 'user_email': user.email})
            return redirect(url_for('dashboard'))
        flash('Invalid credentials.', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirm_password']
        if password != confirm:
            flash("Passwords do not match", "warning")
        elif User.query.filter((User.username == username) | (User.email == email)).first():
            flash("Username or Email already exists", "warning")
        else:
            hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
            db.session.add(User(username=username, email=email, password=hashed_pw))
            db.session.commit()
            flash("Signup successful!", "success")
            return redirect(url_for('login'))
    return render_template("signup.html")

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("dashboard.html")

@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(filepath)
            try:
                input_tensor = preprocess_image(filepath)
                with torch.no_grad():
                    confidence = torch.sigmoid(model(input_tensor)).item()
                result = 'FAKE' if confidence > 0.5 else 'REAL'
                save_result(session['user_id'], f'static/uploads/{unique_name}', result, confidence)
                flash(f"Prediction: {result} (Confidence: {confidence:.2%})", "info")
                return render_template("upload_image.html", image_url=f"uploads/{unique_name}")
            except Exception as e:
                flash("Failed to process image.", "danger")
    return render_template("upload_image.html")

@app.route('/upload-video', methods=['GET', 'POST'])
def upload_video():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST' and 'video' in request.files:
        file = request.files['video']
        if file:
            filename = secure_filename(file.filename)
            unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{filename}"
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(video_path)
            cap = cv2.VideoCapture(video_path)
            frame_data = []
            confidences, frame_num = [], 0
            frame_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_name}_frames")
            os.makedirs(frame_folder, exist_ok=True)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame_num += 1
                if frame_num % 30 == 0:
                    input_tensor = preprocess_frame(frame)
                    with torch.no_grad():
                        conf = torch.sigmoid(model(input_tensor)).item()
                    frame_name = f"frame_{frame_num}.jpg"
                    full_path = os.path.join(frame_folder, frame_name)
                    cv2.imwrite(full_path, frame)
                    frame_data.append({'path': f'uploads/{unique_name}_frames/{frame_name}', 'confidence': round(conf, 2)})
                    confidences.append(conf)
            cap.release()
            if confidences:
                avg_conf = sum(confidences) / len(confidences)
                result = 'FAKE' if avg_conf > 0.5 else 'REAL'
                save_result(session['user_id'], f'static/uploads/{unique_name}', result, avg_conf)
                flash(f"Prediction: {result} (Confidence: {avg_conf:.2%})", "info")
            else:
                flash("No frames analyzed.", "danger")
            return render_template("upload_video.html", video_url=f"uploads/{unique_name}", frame_data=frame_data)
    return render_template("upload_video.html")

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    results = AnalysisResult.query.filter_by(user_id=session['user_id']).order_by(AnalysisResult.uploaded_at.desc()).all()
    return render_template("history.html", results=results)

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for('login'))

# --- Cleanup ---
def cleanup_temp_uploads():
    print("🧹 Cleaning up temporary uploads...")
    uploads_folder = app.config['UPLOAD_FOLDER']
    now = datetime.utcnow()
    for item in os.listdir(uploads_folder):
        path = os.path.join(uploads_folder, item)
        try:
            if os.path.isdir(path) and item.endswith("_frames"):
                shutil.rmtree(path)
                print(f"✅ Deleted folder: {path}")
            elif os.path.isfile(path):
                file_time = datetime.utcfromtimestamp(os.path.getmtime(path))
                if now - file_time > timedelta(hours=24):
                    os.remove(path)
                    print(f"🗑️ Deleted old file: {path}")
        except Exception as e:
            print(f"❌ Error deleting {path}: {e}")

def graceful_shutdown(signum=None, frame=None):
    print("⚠️ Graceful shutdown initiated.")
    cleanup_temp_uploads()
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)
atexit.register(cleanup_temp_uploads)

# --- Main ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.first():
            default_pw = bcrypt.generate_password_hash("admin123").decode('utf-8')
            db.session.add(User(username="admin", email="admin@example.com", password=default_pw))
            db.session.commit()

    is_windows = platform.system() == "Windows"
    port = int(os.environ.get("PORT", 5000))

    if is_windows:
        from waitress import serve
        print("🚀 Running with Waitress (Windows)")
        serve(app, host="0.0.0.0", port=port)
    else:
        print("🟢 Run using: gunicorn app:app --bind 0.0.0.0:$PORT")
        app.run(host="0.0.0.0", port=port)
