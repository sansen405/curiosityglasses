import boto3
import cv2
import numpy as np
from io import BytesIO
import time
import uuid

class S3Uploader:
    def __init__(self, bucket_name, region_name="us-east-2"): 
        """INITIALIZE S3 CLIENT"""
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.frame_counter = 0  # TRACK FRAME NUMBERS
    
    def upload_frame(self, frame):
        """UPLOAD FRAME TO S3 AND RETURN FRAME ID"""
        try:
            # GENERATE UNIQUE FRAME ID
            unique_id = uuid.uuid4()
            timestamp = int(time.time())
            frame_id = f"frame_{timestamp}_{unique_id}"
            
            # ENCODE FRAME
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = BytesIO(buffer.tobytes())
            
            # UPLOAD TO S3
            self.s3_client.upload_fileobj(
                image_bytes,
                self.bucket_name,
                f"{frame_id}.jpg"
            )
            
            return frame_id
            
        except Exception as e:
            print(f"Error uploading frame: {str(e)}")
            return None 