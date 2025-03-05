import cv2
import numpy as np
import time
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class ObjectDetector:
    def __init__(self, 
                 weights_path='./models/yolov3.weights', 
                 config_path='./models/yolov3.cfg', 
                 classes_path='./models/coco.names'):
        """
        Initialize object detection with comprehensive error checking
        """
        # Validate model files
        self._validate_model_files(weights_path, config_path, classes_path)
        
        # Disable OpenCL
        cv2.ocl.setUseOpenCL(False)
        
        # Load class names
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Generate color palette
        self.colors = self._generate_color_palette()
        
        # Load neural network
        try:
            self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as e:
            print(f"Error loading neural network: {e}")
            print("Detailed file paths:")
            print(f"Weights path: {os.path.abspath(weights_path)}")
            print(f"Config path: {os.path.abspath(config_path)}")
            raise
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

    def _validate_model_files(self, weights_path, config_path, classes_path):
        """
        Validate existence and basic properties of model files
        """
        # Check file existence
        files_to_check = [
            ('Weights', weights_path),
            ('Config', config_path),
            ('Classes', classes_path)
        ]
        
        for file_type, file_path in files_to_check:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_type} file not found: {file_path}")
            
            # Check file size (basic validation)
            file_size = os.path.getsize(file_path)
            print(f"{file_type} file size: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError(f"{file_type} file is empty")

    def _generate_color_palette(self):
        """
        Generate consistent color palette for object classes
        """
        colors = {}
        np.random.seed(42)
        for cls in self.classes:
            colors[cls] = tuple(np.random.randint(100, 230, 3).tolist())
        return colors

    def detect_objects(self, frame):
        """
        Detect objects in the frame with detailed error handling
        """
        try:
            height, width, _ = frame.shape
            
            # Prepare blob for network input
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), 
                                         swapRB=True, crop=False)
            self.net.setInput(blob)
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Forward pass
            outs = self.net.forward(output_layers)
            
            # Process detections
            detections = self._process_network_output(frame, outs, width, height)
            
            # Draw detections
            annotated_frame = self._draw_detections(frame, detections)
            
            return annotated_frame, detections
        
        except Exception as e:
            print(f"Error in object detection: {e}")
            return frame, []

    def _process_network_output(self, frame, outs, width, height):
        """
        Process raw network output with detailed logging
        """
        class_ids = []
        confidences = []
        boxes = []
        
        # Debugging: print number of output layers
        print(f"Number of output layers: {len(outs)}")
        
        for out in outs:
            # Debugging: print shape of each output
            print(f"Output layer shape: {out.shape}")
            
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Scale bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-maximum suppression
        if len(boxes) > 0:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 
                                       self.confidence_threshold, 
                                       self.nms_threshold)
        else:
            indexes = []
        
        # Prepare detected objects
        detections = []
        for i in range(len(boxes)):
            if len(indexes) == 0 or i in indexes:
                detections.append({
                    'label': self.classes[class_ids[i]],
                    'confidence': confidences[i],
                    'bbox': boxes[i]
                })
        
        # Log detection results
        print(f"Total detections before NMS: {len(boxes)}")
        print(f"Detections after NMS: {len(detections)}")
        
        return detections

    def _draw_detections(self, frame, detections):
        """
        Draw detected objects on the frame
        """
        for obj in detections:
            x, y, w, h = obj['bbox']
            label = obj['label']
            confidence = obj['confidence']
            
            # Get consistent color for this object's class
            color = self.colors[label]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label text
            label_text = f'{label}: {confidence:.2f}'
            
            # Text background
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x, y - text_height - 10), 
                          (x + text_width, y), color, -1)
            
            # Draw label
            cv2.putText(frame, label_text, (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (255, 255, 255), 1)
        
        return frame

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Check webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize detector
    try:
        detector = ObjectDetector()
    except Exception as e:
        print(f"Detector initialization failed: {e}")
        return
    
    # Pre-define text parameters
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 0.7
    text_color = (0, 255, 0)
    text_thickness = 1
    text_line_type = cv2.LINE_AA
    
    # FPS tracking with smoothing
    prev_time = time.time()
    frame_count = 0
    fps = 0
    fps_update_interval = 0.5  # Update FPS every half second
    last_fps_update = prev_time
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Calculate FPS with smoothing
        curr_time = time.time()
        frame_count += 1
        
        # Update FPS only at intervals
        if curr_time - last_fps_update >= fps_update_interval:
            fps = frame_count / (curr_time - last_fps_update)
            frame_count = 0
            last_fps_update = curr_time
        
        # Detect objects
        processed_frame, detected_objects = detector.detect_objects(frame)
        
        # Combine text rendering for efficiency
        info_text = f'FPS: {fps:.1f} | Objects: {len(detected_objects)}'
        
        # Optional: Create a semi-transparent background for text
        overlay = processed_frame[5:40, 5:350].copy()
        cv2.rectangle(processed_frame, (5, 5), (350, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.2, processed_frame[5:40, 5:350], 0.8, 0, processed_frame[5:40, 5:350])
        
        # Single text render instead of multiple
        cv2.putText(processed_frame, 
                    info_text, 
                    (10, 30), 
                    text_font, 
                    text_scale, 
                    text_color, 
                    text_thickness,
                    text_line_type)
        
        # Show frame
        cv2.imshow('Object Detection', processed_frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()