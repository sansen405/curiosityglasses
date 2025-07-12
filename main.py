import cv2
import os
import math
import heapq
from concurrent.futures import ThreadPoolExecutor
from yolo_detector import download_yolo_files, load_yolo, process_image, display_image
from s3_uploader import S3Uploader

# REPLACE THESE WITH YOUR AWS SETTINGS
BUCKET_NAME = "annotated-frames"
AWS_REGION = "us-east-1"

def upload_frame_task(frame_data):
    """
    Task function for uploading a frame to S3
    """
    processed_image, tracker, s3_uploader = frame_data
    frame_id = s3_uploader.upload_frame(processed_image)
    if frame_id:
        print(f"\nUploaded frame: {frame_id}")
        tracker.add_image_id(frame_id)

if __name__ == "__main__":
    # Get target object type from user
    target_object = input("Enter object type to track (e.g., car, person): ").strip().lower()
    
    # YOLO SETUP
    download_yolo_files()
    net, classes, colors, output_layers = load_yolo()
    
    # TRACKING LISTS AND SETS
    trackers = []
    detected_objects = set()
    
    # S3 SETUP
    s3_uploader = S3Uploader(BUCKET_NAME, region_name=AWS_REGION)
    
    # CLEAR OLD S3 DATA
    print("Clearing S3 bucket...")
    response = s3_uploader.s3_client.list_objects_v2(Bucket=BUCKET_NAME)
    if 'Contents' in response:
        for item in response['Contents']:
            s3_uploader.s3_client.delete_object(Bucket=BUCKET_NAME, Key=item['Key'])
    
    # START THREAD POOL FOR UPLOADS
    with ThreadPoolExecutor(max_workers=3) as executor:
        # VIDEO SETUP
        video_path = "tesla.mp4"
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / original_fps
        target_fps = 10
        frame_interval = int(original_fps / target_fps)
        expected_frames = math.ceil(total_frames / frame_interval)
        
        # SHOW VIDEO INFO
        print(f"\nVideo Stats:")
        print(f"Duration: {video_duration:.2f} seconds")
        print(f"Original FPS: {original_fps}")
        print(f"Total Frames: {total_frames}")
        print(f"Target FPS: {target_fps}")
        print(f"Processing every {frame_interval}th frame")
        print(f"Expected processed frames: {expected_frames}")
        
        frame_count = 0
        processed_count = 0
        futures = []
        
        print("\nProcessing video frames...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # PROCESS EVERY Nth FRAME
            if frame_count % frame_interval == 0:
                processed_count += 1
                # DETECT OBJECTS
                processed_image, boxes, class_ids, confidences, tracker = process_image(
                    frame, net, classes, colors, output_layers
                )
                
                # SET TARGET AND SAVE TRACKER
                tracker.set_target_object(target_object)
                if target_object in tracker.object_counts:
                    trackers.append(tracker)
                
                # SAVE DETECTED OBJECTS
                detected_objects.update(tracker.object_counts.keys())
                
                # SHOW FRAME INFO
                print(f"\nFrame {processed_count} - Detected Objects:")
                print(tracker)
                
                # UPLOAD TO S3
                future = executor.submit(
                    upload_frame_task, 
                    (processed_image, tracker, s3_uploader)
                )
                futures.append(future)
            
            frame_count += 1
        
        # WAIT FOR UPLOADS
        print("\nWaiting for remaining uploads to complete...")
        for future in futures:
            future.result()
        
        # SAVE TRACKERS BEFORE HEAP OPS
        all_trackers = trackers.copy()
        
        # MAKE AND SHOW HEAP
        print("\nCreating max heap based on confidence scores...")
        heapq.heapify(trackers)
        
        # DISPLAY INITIAL RESULTS
        print(f"\nFrames ordered by {target_object} confidence (highest to lowest):")
        temp_trackers = trackers.copy()
        while temp_trackers:
            top_tracker = heapq.heappop(temp_trackers)
            conf = top_tracker.average_confidences[target_object]
            frame_ids = ", ".join(top_tracker.image_ids)
            print(f"Confidence: {conf:.3f} - Frame IDs: {frame_ids}")
        
        print("\nVideo processing complete!")
        
        # INTERACTIVE LOOP
        while True:
            print("\nEnter a new object type to reorder frames, or 'quit' to exit")
            print("\nDetected object types:", ", ".join(sorted(detected_objects)))
            new_target = input("Object type (e.g., car, person): ").strip().lower()
            
            if new_target == 'quit':
                break
                
            if new_target not in detected_objects:
                print(f"\nNo frames containing '{new_target}' were detected in the video.")
                continue
                
            # RESET AND REORDER
            trackers = all_trackers.copy()
            for tracker in trackers:
                tracker.set_target_object(new_target)
            heapq.heapify(trackers)
            
            # SHOW NEW ORDER
            print(f"\nFrames ordered by {new_target} confidence (highest to lowest):")
            temp_trackers = trackers.copy()
            while temp_trackers:
                top_tracker = heapq.heappop(temp_trackers)
                conf = top_tracker.average_confidences[new_target]
                frame_ids = ", ".join(top_tracker.image_ids)
                print(f"Confidence: {conf:.3f} - Frame IDs: {frame_ids}")
        
        cap.release()
        cv2.destroyAllWindows() 