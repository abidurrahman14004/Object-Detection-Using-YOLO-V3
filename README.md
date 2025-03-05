# Object Detection using YOLOv3

This project implements object detection using **YOLOv3 (You Only Look Once v3)** with OpenCV and deep learning frameworks.

## Features
- Real-time object detection
- Supports image and video input
- Uses **YOLOv3** pre-trained weights for accurate detection
- Implemented in Python with OpenCV

---

## Installation
### **Step 1: Install Dependencies**
Ensure you have Python and `pip` installed, then install the required libraries:

```sh
pip install opencv-python numpy torch torchvision ultralytics
```

If you are using **GPU (CUDA)**, install PyTorch with CUDA support from [PyTorch's official website](https://pytorch.org/get-started/locally/).

---

### **Step 2: Download YOLOv3 Weights and Configuration Files**
1. **Clone the YOLOv3 repository:**
   ```sh
   git clone https://github.com/pjreddie/darknet.git
   cd darknet
   ```

2. **Download YOLOv3 pre-trained weights:**
   ```sh
   wget https://pjreddie.com/media/files/yolov3.weights
   ```

3. **Download YOLOv3 configuration files:**
   ```sh
   wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
   wget https://github.com/pjreddie/darknet/blob/master/data/coco.names
   ```

---


## **Project Structure**
```
Object-Detection-YOLOv3/
│── yolov3.cfg
│── yolov3.weights
│── coco.names
│── yolo_detect.py
│── download.py
│── project.py
│── README.md
```

---

## **Credits**
- YOLOv3 Paper: [Joseph Redmon](https://pjreddie.com/darknet/yolo/)
- OpenCV DNN module
- Ultralytics YOLOv3

### 🚀 Happy Coding! 🎯

