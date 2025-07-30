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
1. ONLY match objects to COCO classes if they are EXTREMELY similar:
   - "Tesla" -> "car" (specific brand)
   - "iPhone" -> "cell phone" (specific product)
   - "Labrador" -> "dog" (specific breed)
   - "sparrow" -> "bird" (specific species)
2. DO NOT match loosely related concepts:
   - "grass" is NOT related to "sheep"
   - "road" is NOT related to "car"
   - "furniture" is NOT related to "chair"
3. If no EXTREMELY close match exists in {COCO_CLASSES}, return "no relevant object found"
4. Convert plurals to singular (cars -> car, people -> person)
5. Return empty list for relevant_objects when needs_video is false

USER QUESTION: "{USER_QUESTION}"

Return ONLY this JSON (NO explanations):
{{
    "needs_video": boolean,
    "relevant_objects": ["exact-coco-class", "exact-coco-class"] or ["no relevant object found"] or []
}}"""

def get_collective_frames_prompt(user_question, relevant_objects=None):
    """GENERATE PROMPT FOR ANALYZING MULTIPLE FRAMES TO ANSWER USER'S QUESTION"""
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
    
    # BUILD CONTEXT-AWARE PROMPT
    prompt = f"""The user asked: "{user_question}"

IMPORTANT CONTEXT:
1. Focus on answering EXACTLY what the user asked about, regardless of what objects were used for frame selection
2. Describe what you see in the current view that's relevant to their question
3. If you can't see what they asked about, be direct and honest about it

RESPONSE STRUCTURE:
1. FIRST LINE - DIRECT OBSERVATION:
   - Start with "I'm seeing..." and describe exactly what you observe
   - Include color, position, state, and any immediately visible details
   - Keep this to a single sentence
   Example: "You are currently seeing a white Tesla car positioned at an angle, showing both the front and side of the vehicle."

2. IDENTIFY SPECIFIC TYPE/MODEL:
   - If you can identify a specific model/type/breed/variant, state it clearly
   - Use a transition like "This appears to be..." or "I can identify this as..."
   Example: "This appears to be a Tesla Model X, the company's luxury SUV offering."

3. GENERAL INFORMATION:
   - Provide 2-3 interesting facts about that specific model/type
   - Focus on notable features, specifications, or characteristics
   - Use factual, informative tone
   Example: "The Tesla Model X is known for its distinctive falcon-wing doors and was first introduced in 2015. It features all-wheel drive capability and can seat up to seven passengers."

4. MAINTAIN NATURAL CONVERSATION:
   - Use real-time perspective throughout
   - End with an invitation for more specific questions if appropriate
   - If you can't see what they asked about: "I don't see [what they asked about] in my current view. Feel free to point me towards it if you'd like me to take a look!"

Through my smart glasses, I'll analyze what's currently in view regarding what you asked about."""
    
    return prompt

def get_direct_answer_prompt(question):
    """GENERATE PROMPT FOR DIRECT FACTUAL ANSWERS"""
    return f"""Answer this question directly with factual information in simple, concise terms. Keep your response to 1-2 sentences and focus on key facts.

Question: {question}

Provide a clear, factual answer:"""

if __name__ == "__main__":
    #INITIALIZE GPT
    GPT = GPTHandler()
    
    #SYSTEM ROLE
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
