import cv2
import os
import math
import heapq
from concurrent.futures import ThreadPoolExecutor
from yolo_detector import download_yolo_files, load_yolo, process_image, display_image
from s3_uploader import S3Uploader
from s3_accessor import S3Accessor
import numpy as np

# REPLACE THESE WITH YOUR AWS SETTINGS
BUCKET_NAME = "annotated-frames"
AWS_REGION = "us-east-1"

def display_top_frames(frame_ids, s3_accessor):
    """DISPLAY MULTIPLE FRAMES SIDE BY SIDE"""
    frames = []
    max_height = 0
    total_width = 0
    
    # LOAD ALL FRAMES
    for frame_id in frame_ids:
        frame = s3_accessor.get_frame(frame_id)
        if frame is not None:
            frames.append(frame)
            max_height = max(max_height, frame.shape[0])
            total_width += frame.shape[1]
    
    if not frames:
        print("No frames to display")
        return
    
    # CREATE COMBINED IMAGE
    combined = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    current_x = 0
    
    # COMBINE FRAMES
    for frame in frames:
        h, w = frame.shape[:2]
        # CENTER VERTICALLY IF HEIGHTS DIFFER
        y_offset = (max_height - h) // 2
        combined[y_offset:y_offset+h, current_x:current_x+w] = frame
        current_x += w
    
    # DISPLAY COMBINED IMAGE
    cv2.imshow('Top Frames', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def upload_frame_task(frame_data):
    """TASK FUNCTION FOR UPLOADING A FRAME TO S3"""
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
    try:
        # LIST ALL OBJECTS
        response = s3_uploader.s3_client.list_objects_v2(Bucket=BUCKET_NAME)
        if 'Contents' in response:
            # DELETE ALL OBJECTS
            objects_to_delete = [{'Key': item['Key']} for item in response['Contents']]
            s3_uploader.s3_client.delete_objects(
                Bucket=BUCKET_NAME,
                Delete={'Objects': objects_to_delete}
            )
        print("S3 bucket cleared successfully")
    except Exception as e:
        print(f"Error clearing bucket: {str(e)}")
        print("Continuing anyway...")
    
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
        
        # GET TOP 3 FRAMES
        top_frames = []
        for _ in range(min(3, len(temp_trackers))):
            if temp_trackers:
                top_tracker = heapq.heappop(temp_trackers)
                top_frames.extend(top_tracker.image_ids)
                conf = top_tracker.average_confidences[target_object]
                frame_ids = ", ".join(top_tracker.image_ids)
                print(f"Confidence: {conf:.3f} - Frame IDs: {frame_ids}")
        
        # DISPLAY TOP 3 FRAMES
        print("\nDisplaying top 3 frames...")
        s3_accessor = S3Accessor(BUCKET_NAME, region_name=AWS_REGION)
        display_top_frames(top_frames[:3], s3_accessor)
        
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
            
            # SHOW NEW ORDER AND TOP 3
            print(f"\nFrames ordered by {new_target} confidence (highest to lowest):")
            temp_trackers = trackers.copy()
            
            # GET AND DISPLAY TOP 3
            top_frames = []
            while temp_trackers and len(top_frames) < 3:
                top_tracker = heapq.heappop(temp_trackers)
                top_frames.extend(top_tracker.image_ids)
                conf = top_tracker.average_confidences[new_target]
                frame_ids = ", ".join(top_tracker.image_ids)
                print(f"Confidence: {conf:.3f} - Frame IDs: {frame_ids}")
            
            print("\nDisplaying top 3 frames...")
            display_top_frames(top_frames[:3], s3_accessor)
            
            # DISPLAY REMAINING RESULTS
            while temp_trackers:
                top_tracker = heapq.heappop(temp_trackers)
                conf = top_tracker.average_confidences[new_target]
                frame_ids = ", ".join(top_tracker.image_ids)
                print(f"Confidence: {conf:.3f} - Frame IDs: {frame_ids}")
        
        cap.release()
        cv2.destroyAllWindows() 