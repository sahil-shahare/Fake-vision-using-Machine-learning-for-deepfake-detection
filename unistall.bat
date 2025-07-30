@echo off
echo Uninstalling Python packages...

pip uninstall -y Flask Flask_SQLAlchemy Flask-Bcrypt Werkzeug waitress python-dotenv
pip uninstall -y torch torchvision torchaudio
pip uninstall -y timm opencv-python Pillow Jinja2

echo All listed packages have been uninstalled.
pause
