import boto3
import cv2
import numpy as np
import os
from io import BytesIO

class S3Accessor:
    def __init__(self, bucket_name, region_name="us-east-2"):
        """INITIALIZE S3 CLIENT"""
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=region_name)
    
    def get_frame(self, frame_id):
        """GET FRAME FROM S3 BY ID AND RETURN AS CV2 IMAGE"""
        try:
            # GET OBJECT FROM S3
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=f"{frame_id}.jpg"
            )
            
            # READ IMAGE DATA
            image_data = response['Body'].read()
            
            # CONVERT TO CV2 FORMAT
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
                
            return image
            
        except self.s3_client.exceptions.NoSuchKey:
            print(f"Frame {frame_id} not found in S3")
            return None
        except Exception as e:
            print(f"Error retrieving frame {frame_id}: {str(e)}")
            return None
    
    def save_frame_locally(self, frame_id, output_dir="downloaded_frames"):
        """SAVE FRAME TO LOCAL DIRECTORY"""
        # CREATE OUTPUT DIR IF NEEDED
        os.makedirs(output_dir, exist_ok=True)
        
        # GET FRAME
        image = self.get_frame(frame_id)
        if image is None:
            return False
        
        # SAVE TO FILE
        output_path = os.path.join(output_dir, f"{frame_id}.jpg")
        cv2.imwrite(output_path, image)
        print(f"Saved frame to {output_path}")
        return True
    
    def display_frame(self, frame_id):
        """DISPLAY FRAME IN WINDOW"""
        image = self.get_frame(frame_id)
        if image is not None:
            cv2.imshow(f"Frame {frame_id}", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return True
        return False

if __name__ == "__main__":
    # EXAMPLE USAGE
    BUCKET_NAME = "annotated-frames"
    accessor = S3Accessor(BUCKET_NAME)
    
    # GET FRAME ID FROM USER
    frame_id = input("Enter frame ID to retrieve: ").strip()
    
    # SHOW OPTIONS
    print("\nChoose action:")
    print("1. Display frame")
    print("2. Save frame locally")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        accessor.display_frame(frame_id)
    elif choice == "2":
        accessor.save_frame_locally(frame_id)
    else:
        print("Invalid choice") 