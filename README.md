# Fake Vision

**Deepfake Detection using Machine Learning**

A Python and web-based application to detect deepfakes using machine learning techniques, featuring a Flask app, a trained model, and a UI for testing.

---

##  Table of Contents

- [Project Overview]  
- [Features]  
- [Repository Structure]  
- [Installation]  
- [Usage]  
- [Model Training & Database]  
- [Deployment] 
- [Contributing] 
- [License]

---

##  Project Overview

This project aims to detect manipulated media (deepfakes), using a machine‑learning model served via a Flask app to classify images/videos as real or fake. The application includes a user interface and a backend model, enabling quick testing and deployment.

---

##  Features

-  **Flask-based web interface** to upload and analyze media.  
-  **Machine learning model** to determine authenticity.  
-  **SQLite database** for logging and result tracking.  
-  **Automated deployment** via Render (or similar services).

---

##  Repository Structure

```
.
├── app.py                  # Main Flask application
├── deepfake.sql            # Schema or sample database for deepfake results
├── models/                 # Trained model files
├── templates/              # HTML templates for UI
├── static/                 # CSS, JS, images
├── Procfile                # Deployment configuration (e.g., Render or Heroku)
├── render.yaml             # Configuration for Render deployment
├── requirements.txt        # Python dependencies
├── runtime.txt             # Runtime environment specification
├── site.db                 # Local SQLite database
├── uninstall.bat           # Cleanup script for Windows
└── Direct Run.txt          # Instructions for running locally
```

---

##  Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/sahil-shahare/Fake-vision-using-Machine-learning-for-deepfake-detection.git
   cd Fake-vision-using-Machine-learning-for-deepfake-detection
   ```

2. **Set up a virtual environment (optional but recommended)**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # or `venv\Scripts\activate` on Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

##  Usage

1. **Run the Flask app**  
   ```bash
   python app.py
   ```

2. **Access the application**  
   Open your browser and go to `http://localhost:5000`

3. **Upload media**  
   Use the UI to upload images or videos for deepfake detection.

4. **View results**  
   The interface should display the classification result, and logs are saved in the `site.db`.

---

##  Model Training & Database

- **models/** folder contains pre-trained model files.  
- **deepfake.sql** defines the database schema (e.g., results table).  
- **site.db** stores detection logs such as file name, timestamp, prediction, confidence, etc.

---

##  Deployment

Configured for deployment using Render (via `render.yaml`) or similar services. Ensure your dependencies, runtime, and startup commands are properly defined.

---

##  Contributing

Contributions are welcome! You might consider:

- Improving detection accuracy or adding model explainability.  
- Supporting batch uploads (e.g., process all frames of a video).  
- Enabling deployment on non-CUDA systems or containerizing (Docker).  
- Enhancing UI/UX or logging capabilities.

---

##  License



---

##  Acknowledgements

- Built upon concepts in **deepfake detection** research using CNNs and other ML techniques.  
- Thanks to contributors who work on improving media authenticity and detection.
