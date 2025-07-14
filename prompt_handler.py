import os
from gpt_handler import GPTHandler

# COCO CLASSES (80 OBJECTS)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_initial_prompt(USER_QUESTION):
    """DETERMINE IF QUESTION REQUIRES VIDEO ANALYSIS FROM SMARTGLASSES"""
    return f"""Analyze whether this question requires video analysis based on these STRICT rules:

1. VIDEO ANALYSIS REQUIRED WHEN (ANY):
   - Demonstrative pronouns ("this/that/these/those")
   - Definite articles ("the") with physical objects
   - Possessive context ("my/your/our") with objects
   - Location words ("here/there/current")
   - Visual commands ("identify/read/describe")

2. GENERAL KNOWLEDGE WHEN:
   - No reference to physically present objects
   - Abstract/historical questions

OBJECT HANDLING RULES:
1. MUST ONLY use these detectable COCO classes: {', '.join(COCO_CLASSES)}
2. If object mentioned isn't in COCO classes: 
   - Still return needs_video:true if visual context exists
   - Set relevant_object to "no relevant object found"
3. Only return null for relevant_object when needs_video is false

USER QUESTION: "{USER_QUESTION}"

Return ONLY this JSON (NO explanations):
{{
    "needs_video": boolean,
    "relevant_object": "exact-coco-class"|"no relevant object found"|null
}}"""

def get_collective_frames_prompt(target_object=None):
    """GENERATE PROMPT FOR ANALYZING MULTIPLE FRAMES COLLECTIVELY"""
    if target_object and target_object != "no relevant object found":
        return f"""Analyze these images as a sequence and provide a comprehensive description focusing on the {target_object}(s) shown.

Look across all the images and describe:
1. The {target_object}(s) - appearance, color, condition, brand/model if identifiable
2. How the {target_object} appears across the different frames
3. The setting and environment where the {target_object} is located
4. Any changes or different angles/perspectives shown
5. Other notable objects or context that help understand the scene

Provide one unified description that captures what you can determine about the {target_object} from viewing all these images together."""
    else:
        return """Analyze these images as a sequence and provide a comprehensive description of what you see.

Look across all the images and describe:
1. The main objects, people, or subjects visible
2. The setting and environment
3. Any actions, movements, or changes between the images
4. The overall context or story being told
5. Key details about colors, conditions, brands, or identifying features

Provide one unified description that captures the complete scene from viewing all these images together."""

if __name__ == "__main__":
    # INITIALIZE GPT
    GPT = GPTHandler()
    
    # SET SYSTEM ROLE
    SYSTEM_ROLE = """You are an AI assistant analyzing questions about video content.
Your task is to determine if user questions require further video context or not.
Be concise and precise in your responses."""
    
    while True:
        print("\nEnter your question (or 'quit' to exit):")
        QUESTION = input().strip()
        
        if QUESTION.lower() == 'quit':
            break
        
        # GET AND PRINT PROMPT
        PROMPT = get_initial_prompt(QUESTION)
        print("\nSending to GPT...")
        
        # GET GPT RESPONSE
        RESPONSE = GPT.get_json_completion(PROMPT, ROLE=SYSTEM_ROLE)
        
        if RESPONSE:
            print("\nGPT Response:")
            print("-------------")
            if RESPONSE["needs_video"]:
                print(f"This question needs video analysis.")
                print(f"Relevant object: {RESPONSE['relevant_object']}")
            else:
                print("This question doesn't need video analysis.")
                print(f"Answer: {RESPONSE['answer']}")
            print("-------------")
