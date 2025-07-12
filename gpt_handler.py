import os
import json
from openai import OpenAI
import configparser
from pathlib import Path

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
        self.TEMPERATURE = 0.7              # CONTROLS RANDOMNESS
        self.MAX_TOKENS = 150               # LIMITS RESPONSE LENGTH
    
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