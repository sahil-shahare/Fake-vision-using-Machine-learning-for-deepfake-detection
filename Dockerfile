# Use a stable Python base image compatible with torch
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System dependencies (OpenCV, etc.)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Create virtual environment and activate it
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Flask runs on (optional, for local use)
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
