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
    # EXTRACT OBJECT COUNTS FROM USER QUESTION
    import re
    
    # SAFETY CHECKS FOR PLURALS AND COUNTS
    def get_expected_counts(question):
        """Extract expected counts of objects from question"""
        counts = {}
        words = question.lower().split()
        for i, word in enumerate(words):
            if word in ['these', 'those', 'the']:
                if i + 1 < len(words) and words[i + 1].endswith('s'):
                    base_word = words[i + 1].rstrip('s')
                    counts[base_word] = 'multiple'
        return counts
    
    expected_counts = get_expected_counts(user_question)
    
    if relevant_objects and relevant_objects != ["no relevant object found"]:
        objects_text = ", ".join(relevant_objects)
        
        # BUILD CONTEXT-AWARE PROMPT
        prompt = f"""The user asked: "{user_question}"

IMPORTANT CONTEXT CHECKS:
1. The user's question implies they expect to see: {', '.join(f'{obj} ({"multiple" if obj in expected_counts else "single"})' for obj in relevant_objects)}
2. In the provided frames, carefully count each object type.
3. If there's a mismatch between expected and actual counts, acknowledge this first.
   Example: "I notice you asked about multiple cars, but I can only see one car in the frame. The one car I see is a Tesla Model S. The Tesla Model S is a luxury electric car known for its sleek design and advanced technology."

ANALYSIS INSTRUCTIONS:
1. Start by addressing any count mismatches between user expectations and what you see.
2. For each object present, identify its specific type, model, or characteristics.
3. Provide interesting details about the specific type you've identified.
4. Keep your response conversational and around 3-4 sentences.

Look at these images and identify the specific type, breed, or model of the {objects_text} shown. Each image may focus on different objects from the user's question."""
    else:
        prompt = f"""The user asked: "{user_question}"

ANALYSIS INSTRUCTIONS:
1. First, count and acknowledge the number of each type of object you see.
2. For each object, identify its specific type, model, or characteristics.
3. Provide interesting details about the specific types you've identified.
4. Keep your response conversational and around 3-4 sentences.

Look at these images and identify the specific types, breeds, or models of what you see."""
    
    return prompt

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
