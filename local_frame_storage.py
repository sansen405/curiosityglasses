import cv2
import os
import uuid
from pathlib import Path
import shutil

class LocalFrameStorage:
    def __init__(self, base_dir="frames"):
        """INIT FRAME STORAGE"""
        self.base_dir = Path(base_dir)
        
        # CLEAN OLD FRAMES
        if self.base_dir.exists():
            shutil.rmtree(str(self.base_dir))
            print(f"Cleaned up existing frames in {self.base_dir}")
            
        # INIT NEW DIR
        self.base_dir.mkdir(exist_ok=True)
        print(f"Created new frames directory at {self.base_dir}")
        
    def save_frame(self, frame):
        """SAVE FRAME AND RETURN ID"""
        frame_id = str(uuid.uuid4())[:8]
        frame_path = self.base_dir / f"{frame_id}.jpg"
        
        # SAVE TO DISK
        cv2.imwrite(str(frame_path), frame)
        return frame_id
        
    def get_frame(self, frame_id):
        """GET FRAME BY ID"""
        frame_path = self.base_dir / f"{frame_id}.jpg"
        if not frame_path.exists():
            print(f"Frame {frame_id} not found")
            return None
            
        # READ FROM DISK
        frame = cv2.imread(str(frame_path))
        return frame
    
    def cleanup(self):
        """REMOVE ALL FRAMES"""
        for frame_file in self.base_dir.glob("*.jpg"):
            frame_file.unlink()
            
    def __del__(self):
        """AUTO CLEANUP"""
        try:
            self.cleanup()
        except:
            pass 