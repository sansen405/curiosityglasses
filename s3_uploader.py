import boto3
import cv2
from datetime import datetime
import uuid

class S3Uploader:
    def __init__(self, bucket_name, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
        """
        Initialize S3 uploader with credentials
        If credentials are None, boto3 will use default credentials from ~/.aws/credentials
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

    def upload_frame(self, image, prefix='frames'):
        """
        Upload a frame to S3 and return its unique ID
        """
        # GENERATE UNIQUE FRAME ID: timestamp_uuid
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        frame_id = f"{timestamp}_{unique_id}"
        
        # CREATE S3 KEY
        s3_key = f"{prefix}/{frame_id}.jpg"
        
        try:
            # CONVERT OPENCV IMAGE TO JPG BYTES
            _, img_encoded = cv2.imencode('.jpg', image)
            
            # UPLOAD TO S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=img_encoded.tobytes()
            )
            return frame_id
        except Exception as e:
            print(f"ERROR UPLOADING TO S3: {e}")
            return None 