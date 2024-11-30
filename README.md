
# Vehicle Detection Web Application

This web application allows users to upload an image and perform vehicle detection using the YOLO (You Only Look Once) model. The processed image with bounding boxes around detected vehicles is saved in a folder named `identified`. The user can then view the result through a link provided on the page.

## Features
- Upload an image for vehicle detection.
- YOLO-based vehicle detection.
- Processed images are stored in the `identified` folder.
- Display processed images with bounding boxes around detected vehicles.
- Easy-to-use interface with error and success messages.

## Prerequisites
Before running the application, ensure you have the following installed:

- Python 3.x
- Flask (Web framework)
- OpenCV (For image processing and YOLO model)
- NumPy (For numerical operations)

### Dependencies

You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

You can create a `requirements.txt` file with the following content:

```
Flask==2.0.2
opencv-python==4.5.3.56
numpy==1.21.2
```

## Project Structure

```
/vehicle-detection-app
    /identified           <-- Folder for saved processed images
    /uploads             <-- Folder for uploaded images
    app.py               <-- Flask backend application
    /static              <-- Static files for web app (CSS, JS)
    /templates           <-- HTML templates (e.g., index.html)
    requirements.txt     <-- Python dependencies
    README.md            <-- Project documentation
```

## Installation & Setup

1. **Clone the repository:**

   You can clone this project using the following command:

   ```bash
   git clone https://github.com/your-username/vehicle-detection-app.git
   cd vehicle-detection-app
   ```

2. **Install dependencies:**

   Install the required dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO Model Files:**

   You'll need to download the YOLOv3 model files (`yolov3.weights` and `yolov3.cfg`) and place them in the root directory of the project. You can download them from:

   - [YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights)
   - [YOLOv3 Config](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)

4. **Run the Flask Application:**

   Once everything is set up, you can run the application using the following command:

   ```bash
   python app.py
   ```

   The application will start on `http://127.0.0.1:5000/` by default.

5. **Upload an Image for Detection:**

   Open the application in your browser (`http://127.0.0.1:5000/`) and upload an image. The app will process the image, perform vehicle detection using YOLO, and save the output in the `identified` folder.

6. **View the Result:**

   After processing, you will be provided with a link to view the processed image with bounding boxes around the detected vehicles.

## Usage

1. Visit the web page and upload an image of a vehicle.
2. The backend will process the image and detect vehicles using the YOLO model.
3. The result will be saved to the `identified` folder.
4. A success message with a link to the processed image will be displayed on the page.

## License

This project is open-source and available under the MIT License.

---

Feel free to modify the code and customize it to your needs. If you encounter any issues, feel free to open an issue on the GitHub repository.
