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

def get_collective_frames_prompt(user_question, target_object=None):
    """GENERATE PROMPT FOR ANALYZING MULTIPLE FRAMES TO ANSWER USER'S QUESTION"""
    if target_object and target_object != "no relevant object found":
        return f"""The user asked: "{user_question}"

Look at these images and identify the specific make and model of the {target_object}. Once you've identified it, provide general information about that model in a conversational way.

For example, if you see a Tesla Model X, don't describe this specific car in the images, but instead tell me about the Tesla Model X as a vehicle model in general. Keep your response conversational and around 3 sentences. Focus on interesting facts, features, or characteristics of that model."""
    else:
        return f"""The user asked: "{user_question}"

Look at these images and identify the main subject they're asking about. Once you've identified it, provide general information about that subject in a conversational way.

Keep your response conversational and around 3 sentences. Focus on interesting facts, features, or characteristics rather than describing what's specifically shown in these images."""

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
