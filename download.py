import os
import urllib.request

def ensure_models_directory():
    """Create models directory if it doesn't exist"""
    os.makedirs('models', exist_ok=True)

def download_file(url, filename):
    """
    Download a file from given URL
    
    Args:
        url (str): URL of the file to download
        filename (str): Local filename to save
    """
    full_path = os.path.join('models', filename)
    
    # Check if file already exists
    if os.path.exists(full_path):
        print(f"{filename} already exists. Skipping download.")
        return full_path
    
    print(f"Downloading {filename}...")
    
    try:
        # Download the file
        urllib.request.urlretrieve(url, full_path)
        
        print(f"{filename} downloaded successfully!")
        return full_path
    
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None

def download_yolo_models():
    """Download YOLO model files"""
    ensure_models_directory()
    
    # YOLO model file URLs
    models = [
        ('https://pjreddie.com/media/files/yolov3.weights', 'yolov3.weights'),
        ('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg', 'yolov3.cfg'),
    ]
    
    # Download model files
    for url, filename in models:
        download_file(url, filename)
    
    # Create COCO names file
    coco_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop_sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    # Write COCO classes to file
    with open('models/coco.names', 'w') as f:
        for cls in coco_classes:
            f.write(f"{cls}\n")
    
    print("COCO classes file created.")

if __name__ == '__main__':
    download_yolo_models()