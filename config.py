import os 

# Configuration constants
ENERGY_THRESHOLD = 400
PAUSE_THRESHOLD = 2
SAVE_HISTORY_LAST_N = 10
PLAYBACK_DELAY = 0.5

# API configurations
OPENAI_API_KEY = os.getenv("openai_api_key") or input("Enter OpenAI API Key: ")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or input("Enter Google API Key: ")