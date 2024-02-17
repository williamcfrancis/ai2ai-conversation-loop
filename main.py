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
        threading.Thread(target=self.continuous_ai_conversation, daemon=True).start()
        self.stop_listening = threading.Thread(target=self.listen_for_interaction, daemon=True).start()

    def continuous_ai_conversation(self):
        while self.active_conversation:
            if self.interact_with_human:
                if time.time() - self.last_human_interaction_time > 15:  # 15 seconds of no human interaction
                    self.interact_with_human = False
                    prompt = "A human has stopped talking to us. Let's continue our conversation."
                    self.insert_human_interaction_prompt(prompt)
            else:
                self.ai_to_ai_conversation()
                time.sleep(1)

    def ai_to_ai_conversation(self):
        query = "Continue this conversation casually and naturally. It should mimic human-like conversation. So keep it natural and concise." + "\n".join(self.chat_history)
        if self.next_speaker == 'GPT':
            gemini_response = self.communicate_with_gemini(query)
            print(f"Gemini: {gemini_response}")
            self.next_speaker = 'Gemini'
        else:
            gpt_response = self.communicate_with_gpt(query)
            print(f"GPT: {gpt_response}")
            self.next_speaker = 'GPT'

    def communicate_with_gpt(self, prompt):
        response = openai.chat.completions.create(
                model=self.GPT_model,
                messages=prompt,
                temperature=0.5,
                top_p=0.5
            )
        text_output = response.choices[0].message.content
        return text_output

    def communicate_with_gemini(self, prompt):
        gemini_response = self.gemini_client.generate_content(prompt)
        self.chat_history.append("Gemini: " + gemini_response.text + "\n")
        return gemini_response

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
                self.communicate_with_gpt(prompt)
                self.next_speaker = 'Gemini'
            else:
                self.communicate_with_gemini(prompt)
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

    
    
    