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
from prompt_handler import GPTHandler, get_initial_prompt, get_collective_frames_prompt, get_direct_answer_prompt
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

    def get_frames_for_objects(self, relevant_objects, max_frames=3):
        """GET ONE FRAME FOR EACH OBJECT, OR RANDOM FRAMES IF OBJECT NOT FOUND"""
        if not self.trackers:
            return []
        
        selected_frames = []
        used_frame_ids = set()  # AVOID DUPLICATES
        
        for obj in relevant_objects[:max_frames]:  # LIMIT TO MAX_FRAMES
            if obj == "no relevant object found":
                # GET RANDOM FRAME THAT HASN'T BEEN USED
                available_trackers = [t for t in self.trackers if t.image_ids and not any(img_id in used_frame_ids for img_id in t.image_ids)]
                if available_trackers:
                    import random
                    random_tracker = random.choice(available_trackers)
                    if random_tracker.image_ids:
                        frame_id = random_tracker.image_ids[0]
                        selected_frames.append(frame_id)
                        used_frame_ids.add(frame_id)
                        print(f"Random frame for unknown object: {frame_id}")
            else:
                # GET BEST FRAME FOR THIS SPECIFIC OBJECT
                best_tracker = None
                best_confidence = 0
                
                for tracker in self.trackers:
                    if obj in tracker.object_counts and tracker.image_ids:
                        # SKIP IF ALL FRAMES FROM THIS TRACKER ALREADY USED
                        if all(img_id in used_frame_ids for img_id in tracker.image_ids):
                            continue
                        
                        confidence = tracker.average_confidences.get(obj, 0)
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_tracker = tracker
                
                if best_tracker and best_tracker.image_ids:
                    # GET FIRST UNUSED FRAME FROM BEST TRACKER
                    for frame_id in best_tracker.image_ids:
                        if frame_id not in used_frame_ids:
                            selected_frames.append(frame_id)
                            used_frame_ids.add(frame_id)
                            print(f"Best frame for {obj}: {frame_id} (confidence: {best_confidence:.3f})")
                            break
                else:
                    # OBJECT NOT FOUND - GET RANDOM FRAME
                    available_trackers = [t for t in self.trackers if t.image_ids and not any(img_id in used_frame_ids for img_id in t.image_ids)]
                    if available_trackers:
                        import random
                        random_tracker = random.choice(available_trackers)
                        if random_tracker.image_ids:
                            frame_id = random_tracker.image_ids[0]
                            selected_frames.append(frame_id)
                            used_frame_ids.add(frame_id)
                            print(f"Random frame for missing {obj}: {frame_id}")
        
        return selected_frames

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

    def describe_objects_in_frames(self, frame_ids, user_question, relevant_objects=None):
        """DESCRIBE OBJECTS IN FRAMES USING GPT VISION - COLLECTIVE ANALYSIS"""
        if not frame_ids:
            return "No frames available for analysis."
        
        print(f"\nANALYZING {len(frame_ids)} FRAMES FOR MULTIPLE OBJECTS...")
        
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
        prompt = get_collective_frames_prompt(user_question, relevant_objects)
        
        # ANALYZE ALL IMAGES TOGETHER
        description = self.gpt.describe_multiple_images_collectively(images, prompt)
        
        return description if description else "Failed to analyze images collectively."

    def answer_question_directly(self, question):
        """PROVIDE DIRECT FACTUAL ANSWER FOR QUESTIONS THAT DON'T NEED VIDEO"""
        prompt = get_direct_answer_prompt(question)
        system_role = "You are a knowledgeable assistant that provides concise, factual answers to questions."
        
        answer = self.gpt.get_completion(prompt, system_role)
        return answer if answer else "Unable to provide an answer."

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
                relevant_objects = question_result['relevant_objects']
                print(f"Relevant Objects: {relevant_objects}")
                
                if relevant_objects and relevant_objects != ["no relevant object found"]:
                    print("\nGetting frames for relevant objects...")
                    selected_frames = self.get_frames_for_objects(relevant_objects, 3)
                    if selected_frames:
                        print(f"Found {len(selected_frames)} frames: {selected_frames}")
                        
                        print("\nDESCRIBING OBJECTS IN FRAMES...")
                        descriptions = self.describe_objects_in_frames(selected_frames, self.user_question, relevant_objects)
                        print("\nCOLLECTIVE ANALYSIS:")
                        print("=" * 60)
                        print(descriptions)
                        print("=" * 60)
                    else:
                        print(f"Could not find frames for objects: {relevant_objects}")
            else:
                print("\nProviding direct answer...")
                answer = self.answer_question_directly(self.user_question)
                print("\nANSWER:")
                print("=" * 40)
                print(answer)
                print("=" * 40)
            
        elif question_result and not video_result:
            print("\nVideo processing failed, but can still answer question...")
            if not question_result['needs_video']:
                answer = self.answer_question_directly(self.user_question)
                print("\nANSWER:")
                print("=" * 40)
                print(answer)
                print("=" * 40)
            else:
                print("ERROR: Video analysis required but video processing failed")
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