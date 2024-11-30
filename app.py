import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from dehaze import dehaze  # Import the dehaze function from dehaze.py

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'  # Folder for storing processed images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Utility: Check if file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Serve processed files
@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Upload and process file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error="No file selected.")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Apply dehazing on the uploaded image
        image = cv2.imread(filepath)
        dehazed_image = dehaze(image)  # Apply dehazing using your dehaze function

        # Save the dehazed image
        processed_filename = "dehazed_" + filename
        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        cv2.imwrite(processed_filepath, dehazed_image)

        # Pass the processed filename to result page
        return redirect(url_for('result', filename=processed_filename))

    return render_template('index.html', error="Invalid file type. Please upload a valid image (png, jpg, jpeg).")

# Display result with the dehazed image
@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)  # Ensure processed folder exists
    app.run(debug=True)
