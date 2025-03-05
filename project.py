import cv2
import numpy as np
import time
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class ObjectDetector:
    def __init__(self, 
                 weights_path='./models/yolov3.weights', 
                 config_path='./models/yolov3.cfg', 
                 classes_path='./models/coco.names'):

        self._validate_model_files(weights_path, config_path, classes_path)
        

        cv2.ocl.setUseOpenCL(False)
        

        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        

        self.colors = self._generate_color_palette()
        

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
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

    def _validate_model_files(self, weights_path, config_path, classes_path):
        files_to_check = [
            ('Weights', weights_path),
            ('Config', config_path),
            ('Classes', classes_path)
        ]
        
        for file_type, file_path in files_to_check:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_type} file not found: {file_path}")
            
   
            file_size = os.path.getsize(file_path)
            print(f"{file_type} file size: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError(f"{file_type} file is empty")

    def _generate_color_palette(self):

        colors = {}
        np.random.seed(42)
        for cls in self.classes:
            colors[cls] = tuple(np.random.randint(100, 230, 3).tolist())
        return colors

    def detect_objects(self, frame):
        try:
            height, width, _ = frame.shape
            
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), 
                                         swapRB=True, crop=False)
            self.net.setInput(blob)
            

            layer_names = self.net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            

            outs = self.net.forward(output_layers)
 
            detections = self._process_network_output(frame, outs, width, height)
            
            annotated_frame = self._draw_detections(frame, detections)
            
            return annotated_frame, detections
        
        except Exception as e:
            print(f"Error in object detection: {e}")
            return frame, []

    def _process_network_output(self, frame, outs, width, height):

        class_ids = []
        confidences = []
        boxes = []
        
 
        print(f"Number of output layers: {len(outs)}")
        
        for out in outs:
 
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
                    
          
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        

        if len(boxes) > 0:
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 
                                       self.confidence_threshold, 
                                       self.nms_threshold)
        else:
            indexes = []
        detections = []
        for i in range(len(boxes)):
            if len(indexes) == 0 or i in indexes:
                detections.append({
                    'label': self.classes[class_ids[i]],
                    'confidence': confidences[i],
                    'bbox': boxes[i]
                })
        
        print(f"Total detections before NMS: {len(boxes)}")
        print(f"Detections after NMS: {len(detections)}")
        
        return detections

    def _draw_detections(self, frame, detections):

        for obj in detections:
            x, y, w, h = obj['bbox']
            label = obj['label']
            confidence = obj['confidence']
            
       
            color = self.colors[label]
            

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            label_text = f'{label}: {confidence:.2f}'
            

            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x, y - text_height - 10), 
                          (x + text_width, y), color, -1)
            
            cv2.putText(frame, label_text, (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (255, 255, 255), 1)
        
        return frame

def main():

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    try:
        detector = ObjectDetector()
    except Exception as e:
        print(f"Detector initialization failed: {e}")
        return
    
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 0.7
    text_color = (0, 255, 0)
    text_thickness = 1
    text_line_type = cv2.LINE_AA
    
    prev_time = time.time()
    frame_count = 0
    fps = 0
    fps_update_interval = 0.5 
    last_fps_update = prev_time
    
    while True:

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
    
        curr_time = time.time()
        frame_count += 1
        
        if curr_time - last_fps_update >= fps_update_interval:
            fps = frame_count / (curr_time - last_fps_update)
            frame_count = 0
            last_fps_update = curr_time
        
        processed_frame, detected_objects = detector.detect_objects(frame)
        
        info_text = f'FPS: {fps:.1f} | Objects: {len(detected_objects)}'
        
        overlay = processed_frame[5:40, 5:350].copy()
        cv2.rectangle(processed_frame, (5, 5), (350, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.2, processed_frame[5:40, 5:350], 0.8, 0, processed_frame[5:40, 5:350])
        
        cv2.putText(processed_frame, 
                    info_text, 
                    (10, 30), 
                    text_font, 
                    text_scale, 
                    text_color, 
                    text_thickness,
                    text_line_type)

        cv2.imshow('Object Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
