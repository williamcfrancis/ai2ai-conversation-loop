import argparse
import os
import time
import speech_recognition as sr
import threading
import collections
import pygame
import google.generativeai as genai
import openai
import re
from tempfile import NamedTemporaryFile
import keyboard
import random

# Prompt for API keys if not set as environment variables
OPENAI_API_KEY = os.getenv("openai_api_key") or input("Enter OpenAI API Key: ")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or input("Enter Google API Key: ")
ENERGY_THRESHOLD = 400  # minimum audio energy to consider for recording
PAUSE_THRESHOLD = 2  # seconds of non-speaking audio before a phrase is considered complete
SAVE_HISTORY_LAST_N = 10  # Number of last messages to save in the conversation history

# Configure APIs
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
wake_word_pattern_gpt = re.compile(r'h(ey|i)?,?\s+(gpt|chatgpt)', re.IGNORECASE)
wake_word_pattern_gemini = re.compile(r'h(ey|i)?,?\s+(gemini|google)', re.IGNORECASE)

class AI2AI:
    def __init__(self):
        pygame.mixer.init()
        self.gemini_client = genai.GenerativeModel('gemini-pro')
        self.openai_client = openai.OpenAI()
        self.sr_recognizer = sr.Recognizer()
        self.sr_microphone = sr.Microphone()
        self.chat_history = collections.deque(maxlen=10)
        self.active_conversation = True
        self.next_speaker = 'GPT'  # Determines who speaks next in the AI conversation
        self.interact_with_human = False
        self.topic = args.topic
        self.stop_listening = None
        self.interrupt_conversation = False
        self.resume_conversation = False
        self.human_interaction_count = 0
        self.GPT_model = "gpt-3.5-turbo"

    def start_conversation(self):
        self.stop_listening = threading.Thread(target=self.listen_for_interaction, daemon=True).start()
        threading.Thread(target=self.continuous_ai_conversation, daemon=True).start()

    def continuous_ai_conversation(self):
        self.begin_conversation()
        while self.active_conversation:
            if self.interact_with_human:
                if time.time() - self.last_human_interaction_time > 15:  # 15 seconds of no human interaction
                    self.interact_with_human = False
                    prompt = "A human has stopped talking to us. Let's continue our conversation."
                    self.insert_human_interaction_prompt(prompt)
            else:
                self.ai_to_ai_conversation()
                # print("Chat history: ", self.chat_history)
           
                
    def begin_conversation(self):
        if self.topic == "default":
            self.topic = self.get_random_topic()
        if self.next_speaker == 'GPT':
            initial_prompt = f"As GPT, you are starting a casual and human-like conversation with Google's Gemini about an interesting topic. The topic of the conversation is {self.topic}. Say hi to Gemini."
            gpt_response = self.call_gpt(initial_prompt)
            print(f"GPT: {gpt_response}\n")
            self.next_speaker = 'Gemini'
        elif self.next_speaker == 'Gemini':
            initial_prompt = f"As Google Gemini, You are starting a casual and human-like conversation with OpenAI's GPT about an interesting topic. The topic of the conversation is {self.topic}. Say hi to GPT."
            gemini_response = self.call_gemini(initial_prompt)
            print(f"Gemini: {gemini_response}\n")
            self.next_speaker = 'GPT'
        
    def get_random_topic(self):
        filename = "topics.txt"
        try:
            with open(filename, 'r', encoding='utf-8') as file:  # Specify encoding here
                topics = file.readlines()  # Read all lines into a list
                topics = [topic.strip() for topic in topics]  # Remove any trailing newlines or spaces
                return random.choice(topics) if topics else "default topic"
        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found.")
            return "default topic"

    def ai_to_ai_conversation(self):
        query = "Continue the conversation casually and naturally. Avoid banality and feel free to gradually change topic if it is getting repetitive. Your response should only contain the reply to the last message in the conversation. \nHere is the conversation history:\n" + "\n".join(self.chat_history)
        if self.next_speaker == 'Gemini':
            gemini_response = self.call_gemini(query)
            print(f"Gemini: {gemini_response} \n\n")
            self.next_speaker = 'GPT'
        else:
            gpt_response = self.call_gpt(query)
            print(f"GPT: {gpt_response} \n\n")
            self.next_speaker = 'Gemini'
            
    def call_gpt(self, user_prompt):
        try:
            completion = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an intelligent AI assistant, skilled in engaging in meaningful conversations with both humans and other AI. Your responses should be insightful, respectful, and considerate of the conversational context and topic at hand."},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=100, temperature=1
            )
            # Adjusting the way to access the text output based on the actual structure of the response
            if completion.choices and completion.choices[0].message:
                text_output = completion.choices[0].message.content  # Adjusted access here
                self.chat_history.append("GPT: " + text_output + "\n")
                return text_output
            else:
                print("No response from GPT.")
                return ""
        except Exception as e:
            print(f"Error calling GPT: {e}")
            return ""
        
    def call_gemini(self, prompt):
        try:
            response = self.gemini_client.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=100, temperature=1))
            gemini_response_text = response.text            
            self.chat_history.append("Gemini: " + gemini_response_text + "\n")
            return gemini_response_text
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return ""

    def listen_for_interaction(self):
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        recognizer.pause_threshold = PAUSE_THRESHOLD
        recognizer.energy_threshold = ENERGY_THRESHOLD
        
        # Adjust for ambient noise initially
        with microphone as source:
            print("Adjusting for ambient noise...", end=" ")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Mic energy threshold set at", int(recognizer.energy_threshold))
            
        # Start the background listening
        stop_listening = recognizer.listen_in_background(microphone, self.continuous_transcription_callback, phrase_time_limit=PAUSE_THRESHOLD)
        return stop_listening
    
    def continuous_transcription_callback(self, recognizer, audio):
        try:
            transcription = recognizer.recognize_whisper(audio, model="base.en", language=None)
            print("User: ", transcription)

            if wake_word_pattern_gpt.search(transcription): # Wake word detected
                self.next_speaker = 'GPT'
                self.handle_human_interaction(transcription)
            if wake_word_pattern_gemini.search(transcription): # Wake word detected
                self.next_speaker = 'Gemini'
                self.handle_human_interaction(transcription)
            if "bye" in transcription.lower(): # Interrupt the assistant
                self.resume_conversation = True
                self.text_to_speech("Resuming AI2AI conversation")
                
                
        except sr.UnknownValueError:
            print("Whisper could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results from Whisper service; {e}")
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
            # Continue listening even after timeout

    def handle_human_interaction(self, command):
        prompt = f"A human wants to talk to you. He is asking '{command}'."
        while self.human_interaction_count < 6 and not self.resume_conversation:
            self.human_interaction_count += 1
            if self.next_speaker == 'GPT':
                self.call_gpt(prompt)
                self.next_speaker = 'Gemini'
            else:
                self.call_gemini(prompt)
                self.next_speaker = 'GPT'

    def text_to_speech(self, text):
        threading.Thread(target=self._play_audio, args=(text,), daemon=True).start()
        
    def _play_audio(self, text):
        try:
            
            response = self.openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
            )
            with NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
                audio_file_path = temp_audio_file.name
                response.stream_to_file(audio_file_path)
            
            pygame.mixer.music.load(audio_file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():  # Wait for the audio to finish playing
                if self.interrupt_conversation:  # Check for the interrupt flag
                    pygame.mixer.music.stop()  # Stop the audio
                    break
                

        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")

def main(args):
    ai_conversation = AI2AI()
    ai_conversation.start_conversation()
    try:
        while True:
            time.sleep(0.1)
            if keyboard.is_pressed('esc'):
                break
    except KeyboardInterrupt:
        print("Exiting...")
        ai_conversation.stop_listening(wait_for_stop=False)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI2AI Conversation Loop")
    parser.add_argument("--topic", type=str, default="default", help="Initial topic for the conversation")
    args = parser.parse_args()
    main(args)

    
    
    