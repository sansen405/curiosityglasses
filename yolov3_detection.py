import cv2
import numpy as np
import urllib.request
from IPython.display import Image, display
import matplotlib.pyplot as plt
import os
from collections import defaultdict

class ObjectTracker:
    def __init__(self):
        self.object_counts = defaultdict(int)
        self.average_confidences = defaultdict(float)
    
    def update(self, object_class, confidence):
        """Update counts and average confidence for an object"""
        current_count = self.object_counts[object_class]
        current_avg = self.average_confidences[object_class]
        
        self.object_counts[object_class] += 1
        new_count = self.object_counts[object_class]
        
        #NEW_AVERAGE = (n-1)/n * old_avg + 1/n * new_value
        self.average_confidences[object_class] = (
            (current_count / new_count) * current_avg +
            (1 / new_count) * confidence
        )
    
    #PRINT OBJECT TRACKER STATS
    def __str__(self):
        """Pretty print the tracker's state"""
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
        
        return output
    
    def get_stats(self):
        """Get current statistics"""
        return {
            "counts": dict(self.object_counts),
            "averages": {
                k: round(v, 3) 
                for k, v in self.average_confidences.items()
            }
        }

def download_yolo_files():
    """Download YOLOv3 weights, config, and class names"""
    # YOLO CONFIGURATIONS
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
    """Load YOLOv3 model and classes"""
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    #RANDOM COLORS FOR BOUNDING BOXES
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    return net, classes, colors, output_layers

def process_image(image_path, net, classes, colors, output_layers, conf_threshold=0.5, nms_threshold=0.4):
    """Process image and draw bounding boxes"""
    tracker = ObjectTracker()
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    height, width = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    #DETECTIONS
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    #NMS
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    #OBJECT TRACKER UPDATE
    for i in range(len(boxes)):
        if i in indexes:
            class_name = classes[class_ids[i]]
            confidence = confidences[i]
            tracker.update(class_name, confidence)
            
            #DRAW BOUNDING BOXES
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {confidence:.3f}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    
    return image, boxes, class_ids, confidences, tracker

def display_image(image):
    """Display image using matplotlib"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)  #RENDER TIME
    plt.show()

if __name__ == "__main__":
    download_yolo_files()
    
    #YOLO MODEL
    net, classes, colors, output_layers = load_yolo()
    
    image_path = "/Users/sanjaysenthil/Downloads/dog.jpg" 
    processed_image, boxes, class_ids, confidences, tracker = process_image(
        image_path, net, classes, colors, output_layers
    )
    
    print(tracker)
    print("\nDisplaying image")
    display_image(processed_image) 