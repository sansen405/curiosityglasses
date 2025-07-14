import threading
import queue
import cv2
import os
import math
import heapq
from concurrent.futures import ThreadPoolExecutor
from yolo_detector import download_yolo_files, load_yolo, process_image, display_image, ObjectTracker
from s3_uploader import S3Uploader
from s3_accessor import S3Accessor
import numpy as np
from prompt_handler import GPTHandler, get_initial_prompt, get_collective_frames_prompt
from pathlib import Path
import json

# AWS SETTINGS
BUCKET_NAME = "annotated-frames"
AWS_REGION = "us-east-2"

class VideoPipeline:
    def __init__(self):
        self.gpt = GPTHandler()
        self.question_result = None
        self.user_question = None  # STORE ORIGINAL QUESTION
        self.question_queue = queue.Queue()
        self.video_queue = queue.Queue()
        self.trackers = []
        self.detected_objects = set()
        self.upload_futures = []
        
        # YOLO SETUP
        print("SETTING UP YOLO...")
        download_yolo_files()
        self.net, self.classes, self.colors, self.output_layers = load_yolo()
        
        # S3 SETUP
        self.s3_uploader = S3Uploader(BUCKET_NAME, region_name=AWS_REGION)
        self.s3_accessor = S3Accessor(BUCKET_NAME, region_name=AWS_REGION)

    def upload_frame_task(self, data):
        """UPLOAD A SINGLE FRAME TO S3"""
        processed_frame, tracker = data
        frame_id = self.s3_uploader.upload_frame(processed_frame)
        if frame_id:
            tracker.add_image_id(frame_id)
        return tracker
        
    def process_question(self):
        """GET USER INPUT AND RUN GPT"""
        print("\nEnter your question about the video:")
        question = input().strip()
        self.user_question = question  # STORE ORIGINAL QUESTION
        print("PROCESSING QUESTION WITH GPT...")
        
        prompt = get_initial_prompt(question)
        response = self.gpt.get_json_completion(prompt)
        
        if response:
            self.question_result = response
            self.question_queue.put(response)
            print("\nQUESTION ANALYSIS COMPLETE")
        else:
            self.question_queue.put(None)
            print("\nQUESTION ANALYSIS FAILED")

    def process_video(self, video_path):
        """ANALYZE VIDEO FRAMES"""
        try:
            # SETUP VIDEO
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("ERROR: VIDEO FILE ACCESS FAILED")

            # VIDEO SETTINGS
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target_fps = 10
            frame_interval = int(original_fps / target_fps)
            
            frame_count = 0
            processed_count = 0
            
            # CREATE THREAD POOL FOR S3 UPLOADS
            with ThreadPoolExecutor(max_workers=3) as executor:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # PROCESS EVERY Nth FRAME
                    if frame_count % frame_interval == 0:
                        processed_count += 1
                        
                        # DETECT OBJECTS
                        processed_frame, boxes, class_ids, confidences, tracker = process_image(
                            frame, self.net, self.classes, self.colors, self.output_layers
                        )
                        
                        # SAVE TRACKER AND DETECTED OBJECTS
                        self.trackers.append(tracker)
                        self.detected_objects.update(tracker.object_counts.keys())
                        
                        # SUBMIT UPLOAD TASK
                        future = executor.submit(self.upload_frame_task, (processed_frame, tracker))
                        self.upload_futures.append(future)
                    
                    frame_count += 1
                
                print("\nFRAME PROCESSING COMPLETE - WAITING FOR UPLOADS...")
                # WAIT FOR ALL UPLOADS TO COMPLETE
                for future in self.upload_futures:
                    future.result()
            
            cap.release()
            print("ALL UPLOADS COMPLETE")
            print("VIDEO THREAD FINISHED")
            self.video_queue.put(True)
            
        except Exception as e:
            print(f"ERROR IN VIDEO PROCESSING: {str(e)}")
            self.video_queue.put(False)

    def get_top_frames(self, target_object, n=3):
        """GET BEST N FRAMES BY CONFIDENCE"""
        if not self.trackers:
            return []
        
        # SET TARGET OBJECT FOR ALL TRACKERS
        temp_trackers = self.trackers.copy()
        for tracker in temp_trackers:
            tracker.set_target_object(target_object)
        
        # CREATE HEAP
        heapq.heapify(temp_trackers)
        
        # GET TOP N FRAMES
        top_frames = []
        while temp_trackers and len(top_frames) < n:
            top_tracker = heapq.heappop(temp_trackers)
            if target_object in top_tracker.object_counts:
                top_frames.extend(top_tracker.image_ids)
        
        return top_frames[:n]

    def display_frames(self, frame_ids):
        """DISPLAY FRAMES SIDE BY SIDE"""
        frames = []
        max_height = 0
        total_width = 0
        
        # LOAD FRAMES
        for frame_id in frame_ids:
            frame = self.s3_accessor.get_frame(frame_id)
            if frame is not None:
                frames.append(frame)
                max_height = max(max_height, frame.shape[0])
                total_width += frame.shape[1]
        
        if not frames:
            return
        
        # CREATE COMBINED IMAGE
        combined = np.zeros((max_height, total_width, 3), dtype=np.uint8)
        current_x = 0
        
        # COMBINE FRAMES
        for frame in frames:
            h, w = frame.shape[:2]
            y_offset = (max_height - h) // 2
            combined[y_offset:y_offset+h, current_x:current_x+w] = frame
            current_x += w
        
        # DISPLAY
        cv2.imshow('Top Frames', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def describe_objects_in_frames(self, frame_ids, user_question, target_object=None):
        """DESCRIBE OBJECTS IN FRAMES USING GPT VISION - COLLECTIVE ANALYSIS"""
        if not frame_ids:
            return "No frames available for analysis."
        
        print(f"\nANALYZING {len(frame_ids)} FRAMES COLLECTIVELY...")
        
        # GET ALL IMAGES FROM S3
        images = []
        successful_frames = []
        for frame_id in frame_ids:
            image = self.s3_accessor.get_frame(frame_id)
            if image is not None:
                images.append(image)
                successful_frames.append(frame_id)
        
        if not images:
            return "Could not retrieve any images for analysis."
        
        print(f"Successfully loaded {len(images)} images: {successful_frames}")
        
        # GET COLLECTIVE ANALYSIS PROMPT WITH USER QUESTION
        prompt = get_collective_frames_prompt(user_question, target_object)
        
        # ANALYZE ALL IMAGES TOGETHER
        description = self.gpt.describe_multiple_images_collectively(images, prompt)
        
        return description if description else "Failed to analyze images collectively."

    def run(self, video_path):
        """MAIN PIPELINE EXECUTION"""
        print("\nSTARTING PIPELINE...")
        print("STARTING VIDEO PROCESSING THREAD...")
        # START VIDEO THREAD
        video_thread = threading.Thread(target=self.process_video, args=(video_path,))
        video_thread.start()
        
        print("STARTING QUESTION THREAD...")
        # START QUESTION THREAD
        question_thread = threading.Thread(target=self.process_question)
        question_thread.start()
        
        # WAIT FOR COMPLETION
        question_result = self.question_queue.get()
        video_result = self.video_queue.get()
        
        question_thread.join()
        video_thread.join()
        
        print("\nALL THREADS COMPLETE")
        
        # SHOW RESULTS
        if question_result and video_result:
            print("\nAnalysis Results:")
            print("-----------------")
            print(f"Needs Video: {question_result['needs_video']}")
            
            if question_result['needs_video']:
                relevant_object = question_result['relevant_object']
                print(f"Relevant Object: {relevant_object}")
                
                if relevant_object != "no relevant object found":
                    print("\nGetting top 3 frames...")
                    top_frames = self.get_top_frames(relevant_object, 3)
                    if top_frames:
                        print(f"Found {len(top_frames)} top frames: {top_frames}")
                        
                        print("\nDESCRIBING OBJECTS IN FRAMES...")
                        descriptions = self.describe_objects_in_frames(top_frames, self.user_question, relevant_object)
                        print("\nCOLLECTIVE ANALYSIS:")
                        print("=" * 60)
                        print(descriptions)
                        print("=" * 60)
                    else:
                        print(f"No frames found containing {relevant_object}")
            
        else:
            print("ERROR: PIPELINE FAILED")

if __name__ == "__main__":
    # SET YOUR VIDEO PATH HERE
    VIDEO_PATH = "tesla.mp4"  # REPLACE WITH YOUR VIDEO PATH
    
    pipeline = VideoPipeline()
    
    if not Path(VIDEO_PATH).exists():
        print("ERROR: VIDEO FILE NOT FOUND")
    else:
        pipeline.run(VIDEO_PATH) 