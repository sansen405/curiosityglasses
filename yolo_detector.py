import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import os
from collections import defaultdict

class ObjectTracker:
    def __init__(self):
        self.object_counts = defaultdict(int)
        self.average_confidences = defaultdict(float)
        self.image_ids = []
        self.target_object = None  # FOR HEAP COMPARISON
    
    def set_target_object(self, object_type):
        """Set the object type to use for comparisons"""
        self.target_object = object_type
    
    def __lt__(self, other):
        """Compare based on average confidence of target object"""
        if self.target_object is None:
            raise ValueError("Target object type not set for comparison")
        
        self_conf = self.average_confidences.get(self.target_object, 0)
        other_conf = other.average_confidences.get(self.target_object, 0)
        
        # HANDLE EQUAL CONFIDENCES
        if self_conf == other_conf:
            # EMPTY LISTS GO LAST
            if not self.image_ids:
                return True
            if not other.image_ids:
                return False
            # USE EARLIEST FRAME AS TIEBREAKER
            return self.image_ids[0] < other.image_ids[0]
            
        return self_conf > other_conf  # MAINTAIN MAX HEAP
    
    def update(self, object_class, confidence):
        current_count = self.object_counts[object_class]
        current_avg = self.average_confidences[object_class]
        
        self.object_counts[object_class] += 1
        new_count = self.object_counts[object_class]
        
        # UPDATE RUNNING AVERAGE: NEW_AVG = (N-1)/N * OLD_AVG + 1/N * NEW_VALUE
        self.average_confidences[object_class] = (
            (current_count / new_count) * current_avg +
            (1 / new_count) * confidence
        )
    
    def add_image_id(self, image_id):
        self.image_ids.append(image_id)
    
    def __str__(self):
        output = "\n=== Object Detection Summary ===\n"
        if not self.object_counts:
            return output + "No objects detected"
        
        max_name = max(len(name) for name in self.object_counts.keys())
        output += f"\n{'Object':<{max_name}} | Count | Avg Confidence\n"
        output += "-" * (max_name + 20) + "\n"
        
        for obj in sorted(self.object_counts.keys()):
            count = self.object_counts[obj]
            avg_conf = self.average_confidences[obj]
            output += f"{obj:<{max_name}} | {count:^5} | {avg_conf:.3f}\n"
        
        output += "\nImage IDs:\n" + "\n".join(self.image_ids) 
        return output

def download_yolo_files():
    # YOLO FILE URLS AND NAMES
    weights_url = "https://github.com/patrick013/Object-Detection---Yolov3/raw/master/model/yolov3.weights"
    weights_file = "yolov3.weights"
    config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
    config_file = "yolov3.cfg"
    names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    names_file = "coco.names"
    
    print("Downloading YOLOv3 files...")
    if not os.path.exists(weights_file):
        print("Downloading weights (this might take a while)...")
        urllib.request.urlretrieve(weights_url, weights_file)
    if not os.path.exists(config_file):
        print("Downloading config...")
        urllib.request.urlretrieve(config_url, config_file)
    if not os.path.exists(names_file):
        print("Downloading class names...")
        urllib.request.urlretrieve(names_url, names_file)
    print("All files ready!")

def load_yolo():
    # LOAD MODEL AND CLASSES
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def process_image(frame, net, classes, colors, output_layers, conf_threshold=0.5, nms_threshold=0.4):
    tracker = ObjectTracker()
    
    # PREPARE IMAGE
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    # RUN DETECTION
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # STORE RESULTS
    class_ids = []
    confidences = []
    boxes = []
    
    # GET DETECTIONS
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                # GET BOX COORDS
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # APPLY NMS
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # DRAW BOXES AND UPDATE TRACKER
    for i in range(len(boxes)):
        if i in indexes:
            class_name = classes[class_ids[i]]
            confidence = confidences[i]
            tracker.update(class_name, confidence)
            
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)  
            cv2.putText(frame, f"{label} {confidence:.3f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)  # INCREASED FONT SCALE FROM 1 TO 1.0 AND THICKNESS FROM 2 TO 3
    
    return frame, boxes, class_ids, confidences, tracker

def display_image(image):
    cv2.imshow('Frame', image) #DISPLAY ANNOTATED FRAMES
    cv2.waitKey(1)  