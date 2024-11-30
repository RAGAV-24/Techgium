import os
import cv2
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import torch
from dehaze import dehaze  # Assuming you have a custom dehaze module
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

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
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Uploaded file saved at {filepath}")

        try:
            # Apply dehazing
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("Invalid image file.")

            logger.info("Applying dehazing...")
            dehazed_image = dehaze(image)

            # Save the dehazed image
            processed_filename = "dehazed_" + filename
            processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, dehazed_image)
            logger.info(f"Dehazed image saved at {processed_filepath}")

            # YOLO Object Detection
            detection_results = detect_objects(processed_filepath)

            return render_template(
                'result.html',
                filename=processed_filename,
                detections=detection_results
            )
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            return render_template('index.html', error=f"Processing error: {e}")

    return render_template('index.html', error="Invalid file type. Please upload a valid image (png, jpg, jpeg).")

# YOLO detection function
def detect_objects(image_path):
    logger.info("Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Ensure internet access for the first load

    # Perform inference
    logger.info(f"Running YOLOv5 on {image_path}...")
    results = model(image_path)

    # Save rendered image with detections
    detection_filename = 'detected_' + os.path.basename(image_path)
    detection_filepath = os.path.join(app.config['PROCESSED_FOLDER'], detection_filename)
    results.save(save_dir=app.config['PROCESSED_FOLDER'])

    # Extract detection details
    detections = results.pandas().xyxy[0].to_dict(orient='records')
    logger.info(f"Detections: {detections}")

    return detections

# Display result
@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)