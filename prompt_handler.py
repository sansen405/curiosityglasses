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
2. Identify up to 3 relevant objects maximum
3. Convert plurals to singular (cars -> car, people -> person)
4. If object mentioned isn't in COCO classes but visual context exists:
   - Still return needs_video:true
   - Add "no relevant object found" to the list
5. Return empty list for relevant_objects when needs_video is false

USER QUESTION: "{USER_QUESTION}"

Return ONLY this JSON (NO explanations):
{{
    "needs_video": boolean,
    "relevant_objects": ["exact-coco-class", "exact-coco-class"] or ["no relevant object found"] or []
}}"""

def get_collective_frames_prompt(user_question, relevant_objects=None):
    """GENERATE PROMPT FOR ANALYZING MULTIPLE FRAMES TO ANSWER USER'S QUESTION"""
    if relevant_objects and relevant_objects != ["no relevant object found"]:
        objects_text = ", ".join(relevant_objects)
        return f"""The user asked: "{user_question}"

Look at these images and identify the specific type, breed, or model of the {objects_text} shown. Each image may focus on different objects from the user's question. Once you've identified them, provide general information about those specific types in a conversational way.

Include specific details like the exact type or breed (Golden Retriever, iPhone 14, Tesla Model S, etc.), notable features, and interesting characteristics. For example: "This is a Golden Retriever! Golden Retrievers are friendly, intelligent dogs originally bred in Scotland for retrieving waterfowl." Keep your response conversational and around 3-4 sentences."""
    else:
        return f"""The user asked: "{user_question}"

Look at these images and identify the specific types, breeds, or models of what you see. Once you've identified them, provide general information about those specific items in a conversational way.

Include specific details like exact types, breeds, or model names, notable features, and interesting characteristics. Keep your response conversational and around 3-4 sentences."""

def get_direct_answer_prompt(question):
    """GENERATE PROMPT FOR DIRECT FACTUAL ANSWERS"""
    return f"""Answer this question directly with factual information in simple, concise terms. Keep your response to 1-2 sentences and focus on key facts.

Question: {question}

Provide a clear, factual answer:"""

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
                print(f"Relevant objects: {RESPONSE['relevant_objects']}")
            else:
                print("This question doesn't need video analysis.")
                print(f"Answer: {RESPONSE['answer']}")
            print("-------------")
