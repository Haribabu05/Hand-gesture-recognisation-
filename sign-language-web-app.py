# app.py
from flask import Flask, render_template, Response
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sign_language_detection import SignLanguageDetector

app = Flask(__name__)

# Initialize the detector
model_path = "C:/Users/Handrec/Sign-Language-detection/Model/keras_model.h5"
labels_path = "C:/Users/Handrec/Sign-Language-detection/Model/labels.txt"
detector = None

def init_detector():
    global detector
    if detector is None:
        detector = SignLanguageDetector(model_path, labels_path)

def generate_frames():
    init_detector()
    
    while True:
        success, img = detector.cap.read()
        if not success:
            break
            
        imgOutput = img.copy()
        hands, img = detector.detector.findHands(img)
        
        if hands:
            imgCrop, imgWhite, results = detector.process_hand(img, hands[0])
            if all(v is not None for v in (imgCrop, imgWhite, results)):
                imgOutput = detector.draw_results(imgOutput, results)
        
        # Encode the frame for web streaming
        ret, buffer = cv2.imencode('.jpg', imgOutput)
        if not ret:
            continue
            
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('services.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start_detection():
    init_detector()
    return "Detection started"

@app.route('/stop')
def stop_detection():
    global detector
    if detector:
        detector.cleanup()
        detector = None
    return "Detection stopped"

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        if detector:
            detector.cleanup()
