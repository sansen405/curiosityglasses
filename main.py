import cv2
import os
from yolo_detector import download_yolo_files, load_yolo, process_image, display_image
from s3_uploader import S3Uploader

# REPLACE THESE WITH YOUR AWS SETTINGS
BUCKET_NAME = "annotated-frames"
AWS_REGION = "us-east-2" 

if __name__ == "__main__":
    # INITIALIZE YOLO
    download_yolo_files()
    net, classes, colors, output_layers = load_yolo()
    
    # INITIALIZE S3 UPLOADER
    s3_uploader = S3Uploader(BUCKET_NAME, region_name=AWS_REGION)
    
    # CLEAR S3 BUCKET
    print("Clearing S3 bucket...") 
    response = s3_uploader.s3_client.list_objects_v2(Bucket=BUCKET_NAME)
    if 'Contents' in response:
        for item in response['Contents']:
            s3_uploader.s3_client.delete_object(Bucket=BUCKET_NAME, Key=item['Key'])
    
    # PROCESS VIDEO
    video_path = "tesla.mp4" 
    cap = cv2.VideoCapture(video_path)
    fps = 10
    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            processed_image, boxes, class_ids, confidences, tracker = process_image(
                frame, net, classes, colors, output_layers
            )
            
            # PRINT DETECTIONS
            print("\nDetected Objects:")
            print(tracker)
            
            # UPLOAD TO S3
            frame_id = s3_uploader.upload_frame(processed_image)
            if frame_id:
                print(f"\nUploaded frame: {frame_id}")
            
            # DISPLAY IMAGE
            display_image(processed_image)
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows() 