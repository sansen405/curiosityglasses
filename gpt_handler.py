import os
import json
from openai import OpenAI
import configparser
from pathlib import Path
import base64
import cv2

class GPTHandler:
    def __init__(self, API_KEY=None, PROFILE="default"):
        """INITIALIZE GPT HANDLER WITH API KEY"""
        # FIRST TRY DIRECT API KEY
        self.API_KEY = API_KEY
        
        # THEN TRY ENV VAR
        if not self.API_KEY:
            self.API_KEY = os.getenv('OPENAI_API_KEY')
        
        # FINALLY TRY AWS CREDENTIALS
        if not self.API_KEY:
            try:
                CONFIG = configparser.ConfigParser()
                CONFIG.read(str(Path.home() / ".aws" / "credentials"))
                self.API_KEY = CONFIG[PROFILE]["OPENAI_API_KEY"]
            except:
                raise ValueError(
                    "OPENAI API KEY MUST BE PROVIDED, SET IN OPENAI_API_KEY ENVIRONMENT VARIABLE, "
                    "OR ADDED TO ~/.aws/credentials AS 'OPENAI_API_KEY'"
                )
        
        self.CLIENT = OpenAI(api_key=self.API_KEY)
        
        # DEFAULT SETTINGS
        self.MODEL = "gpt-3.5-turbo"        # MOST COST-EFFECTIVE MODEL
        self.VISION_MODEL = "gpt-4o"        # VISION MODEL FOR IMAGE ANALYSIS
        self.TEMPERATURE = 0.7              # CONTROLS RANDOMNESS
        self.MAX_TOKENS = 150               # LIMITS RESPONSE LENGTH
        self.VISION_MAX_TOKENS = 300        # LONGER FOR IMAGE DESCRIPTIONS
    
    def encode_image(self, image):
        """ENCODE CV2 IMAGE TO BASE64 STRING"""
        try:
            # ENCODE IMAGE TO JPG
            _, buffer = cv2.imencode('.jpg', image)
            # CONVERT TO BASE64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
        except Exception as e:
            print(f"ERROR ENCODING IMAGE: {str(e)}")
            return None

    def get_completion(self, PROMPT, ROLE="You are a helpful AI assistant."):
        """GET COMPLETION FROM GPT"""
        try:
            RESPONSE = self.CLIENT.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": ROLE},
                    {"role": "user", "content": PROMPT}
                ],
                temperature=self.TEMPERATURE,
                max_tokens=self.MAX_TOKENS
            )
            
            return RESPONSE.choices[0].message.content
            
        except Exception as e:
            print(f"ERROR GETTING GPT COMPLETION: {str(e)}")
            return None
    
    def describe_image_objects(self, image, custom_prompt=None):
        """DESCRIBE OBJECTS IN A SINGLE IMAGE"""
        try:
            # PREPARE MESSAGE CONTENT
            if custom_prompt:
                text_prompt = custom_prompt
            else:
                text_prompt = "Describe the main objects you see in this image. Focus on identifying what each object is and any notable details about them."
            
            content = [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encode_image(image)}"
                    }
                }
            ]
            
            RESPONSE = self.CLIENT.chat.completions.create(
                model=self.VISION_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying and describing objects in images. Be specific and detailed in your descriptions."},
                    {"role": "user", "content": content}
                ],
                temperature=self.TEMPERATURE,
                max_tokens=self.VISION_MAX_TOKENS
            )
            
            return RESPONSE.choices[0].message.content
            
        except Exception as e:
            print(f"ERROR DESCRIBING IMAGE OBJECTS: {str(e)}")
            return None

    def describe_multiple_images_collectively(self, images, custom_prompt):
        """ANALYZE MULTIPLE IMAGES TOGETHER AND PROVIDE ONE UNIFIED DESCRIPTION"""
        try:
            if not images:
                return "No images provided for analysis."
            
            # PREPARE MESSAGE CONTENT WITH MULTIPLE IMAGES
            content = [{"type": "text", "text": custom_prompt}]
            
            # ADD ALL IMAGES TO CONTENT
            for i, image in enumerate(images):
                image_base64 = self.encode_image(image)
                if image_base64:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    })
            
            RESPONSE = self.CLIENT.chat.completions.create(
                model=self.VISION_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing multiple images together to provide comprehensive descriptions. Look across all images to understand the complete context."},
                    {"role": "user", "content": content}
                ],
                temperature=self.TEMPERATURE,
                max_tokens=self.VISION_MAX_TOKENS
            )
            
            return RESPONSE.choices[0].message.content
            
        except Exception as e:
            print(f"ERROR ANALYZING MULTIPLE IMAGES: {str(e)}")
            return None

    def get_json_completion(self, PROMPT, ROLE="You are a helpful AI assistant."):
        """GET COMPLETION AND PARSE AS JSON"""
        try:
            RESPONSE = self.get_completion(PROMPT, ROLE)
            if RESPONSE:
                return json.loads(RESPONSE)
            return None
        except json.JSONDecodeError:
            print("ERROR: GPT RESPONSE WAS NOT VALID JSON")
            return None 