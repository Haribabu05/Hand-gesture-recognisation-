import os
import cv2
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import layers
from cvzone.HandTrackingModule import HandDetector
import h5py
import json

def fix_model_loading():
    # Monkey patch _deserialize_keras_object to remove 'groups' from DepthwiseConv2D
    from tensorflow.keras.layers import DepthwiseConv2D
    original_from_config = DepthwiseConv2D.from_config
    
    def patched_from_config(cls, config):
        if 'groups' in config:
            del config['groups']
        return original_from_config(config)
    
    DepthwiseConv2D.from_config = classmethod(patched_from_config)

class SignLanguageDetector:
    def __init__(self, model_path, labels_path):
        # Fix model loading before initializing
        fix_model_loading()
        
        # Validate paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found at: {labels_path}")

        # Initialize parameters
        self.offset = 20
        self.imgSize = 300
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        # Initialize detector
        self.detector = HandDetector(maxHands=1)
        
        # Load the model directly using Keras
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

    def get_prediction(self, img):
        # Preprocess image
        processed_img = cv2.resize(img, (224, 224))
        processed_img = processed_img.astype(np.float32) / 255.0
        processed_img = np.expand_dims(processed_img, axis=0)

        # Get prediction
        prediction = self.model.predict(processed_img, verbose=0)
        index = np.argmax(prediction[0])
        return prediction[0], index

    def process_hand(self, img, hand):
        x, y, w, h = hand['bbox']
        
        # Create white background image
        imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
        
        try:
            # Ensure coordinates are within image bounds
            y_start = max(y - self.offset, 0)
            y_end = min(y + h + self.offset, img.shape[0])
            x_start = max(x - self.offset, 0)
            x_end = min(x + w + self.offset, img.shape[1])
            
            imgCrop = img[y_start:y_end, x_start:x_end]
            
            if imgCrop.size == 0:
                return None, None, None
                
            aspectRatio = h / w
            
            if aspectRatio > 1:
                k = self.imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                wGap = math.ceil((self.imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = self.imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                hGap = math.ceil((self.imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                
            prediction, index = self.get_prediction(imgWhite)
            return imgCrop, imgWhite, (prediction, index, (x, y, w, h))
            
        except Exception as e:
            print(f"Error processing hand: {e}")
            return None, None, None

    def draw_results(self, img, results):
        if results is None:
            return img
        
        prediction, index, (x, y, w, h) = results
        
        # Draw rectangle for text background
        cv2.rectangle(img, 
                     (x-self.offset, y-self.offset-70),
                     (x-self.offset+400, y-self.offset+60-50),
                     (0, 255, 0),
                     cv2.FILLED)
        
        # Draw prediction text
        cv2.putText(img,
                   self.labels[index],
                   (x, y-30),
                   cv2.FONT_HERSHEY_COMPLEX,
                   2,
                   (0, 0, 0),
                   2)
        
        # Draw bounding box
        cv2.rectangle(img,
                     (x-self.offset, y-self.offset),
                     (x+w+self.offset, y+h+self.offset),
                     (0, 255, 0),
                     4)
        return img

    def run(self):
        try:
            while True:
                success, img = self.cap.read()
                if not success:
                    print("Failed to read from camera")
                    break
                    
                imgOutput = img.copy()
                hands, img = self.detector.findHands(img)
                
                if hands:
                    imgCrop, imgWhite, results = self.process_hand(img, hands[0])
                    if all(v is not None for v in (imgCrop, imgWhite, results)):
                        imgOutput = self.draw_results(imgOutput, results)
                        cv2.imshow('ImageCrop', imgCrop)
                        cv2.imshow('ImageWhite', imgWhite)
                
                cv2.imshow('Image', imgOutput)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        # Suppress TensorFlow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
        
        model_path = "/Users/Handrec/Sign-Language-detection/Model/keras_model.h5"
        labels_path = "/Users/Handrec/Sign-Language-detection/Model/labels.txt"
        
        detector = SignLanguageDetector(model_path, labels_path)
        detector.run()
        
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()
