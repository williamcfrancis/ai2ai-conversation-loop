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
from queue import Queue

# Configuration constants
ENERGY_THRESHOLD = 400  # minimum audio energy to consider for recording
PAUSE_THRESHOLD = 2  # seconds of non-speaking audio before a phrase is considered complete
SAVE_HISTORY_LAST_N = 10  # Number of last messages to save in the conversation history
PLAYBACK_DELAY = None  # Delay between playing back pre-generated audio files. Reduce this to speed up the conversation. None for random delay - more human-like

# Configure APIs
OPENAI_API_KEY = os.getenv("openai_api_key") or input("Enter OpenAI API Key: ")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or input("Enter Google API Key: ")
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Wake word patterns
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
        self.topic_msg_count = 0
        self.GPT_model = "gpt-3.5-turbo"
        self.current_audio_thread = None
        self.text_queue = Queue()  # No max size, handles text for speech synthesis
        self.audio_queue = Queue(maxsize=3)  # Audio files ready for playback


    def start_conversation(self):
        self.stop_listening = threading.Thread(target=self.listen_for_interaction, daemon=True).start()
        threading.Thread(target=self.continuous_conversation, daemon=True).start()
        threading.Thread(target=self.speech_synthesis_worker, daemon=True).start()
        threading.Thread(target=self.playback_worker, daemon=True).start()
        
    def continuous_conversation(self):
        self.begin_conversation()
        while self.active_conversation:
            if self.text_queue.qsize() < 3:
                # randomly switch topic after 10 - 20 interactions
                topic_switch_threshold = random.randint(10, 20)
                if self.topic_msg_count > topic_switch_threshold:
                    self.topic = self.get_random_topic()
                    self.topic_msg_count = 0
                    self.moderator_call("Thank you for the interesting conversation, GPT and Gemini. Let's switch to a new topic: " + self.topic)
                    
                if self.interact_with_human:
                    if time.time() - self.last_human_interaction_time > 15:  # 15 seconds of no human interaction
                        self.interact_with_human = False
                        prompt = "A human has stopped talking to us. Let's continue our conversation."
                        self.insert_human_interaction_prompt(prompt)
                else:
                    self.ai_to_ai_conversation()
                    self.topic_msg_count += 1
                    # print("Chat history: ", self.chat_history)
            time.sleep(0.25)
    
    def moderator_call(self, moderator_prompt):
        self.chat_history.append("Moderator: " + moderator_prompt + "\n")
    
    def display_response(self, response):
        print(self.next_speaker + ": " + response + "\n")
        self.next_speaker = 'GPT' if self.next_speaker == 'Gemini' else 'Gemini'
        # Add the response to the text queue instead of direct speech generation
        self.text_queue.put(response)
        
    def begin_conversation(self):
        if self.topic == "default":
            self.topic = self.get_random_topic()
        if self.next_speaker == 'GPT':
            initial_prompt = f"As GPT, an opinionated AI personality, you are starting a casual and human-like conversation with Google's Gemini about an interesting topic. The topic of the conversation is {self.topic}. Say hi to Gemini."
            self.display_response(self.call_gpt(initial_prompt))
        elif self.next_speaker == 'Gemini':
            initial_prompt = f"As Google Gemini, an opinionated AI personality, you are starting a critical and human-like conversation with OpenAI's GPT about an interesting topic. The topic of the conversation is {self.topic}. Say hi to GPT."
            self.display_response(self.call_gemini(initial_prompt))
        
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
        query = "Keep the conversation flowing with creativity and spontaneity, injecting your unique perspective and gently shifting gears to fresh topics as you sense the moment's right. Embrace the art of casual dialogue, mirroring the ebb and flow of human interaction. You are a very opinionated persona. Disagreement can spark interest, so feel free to offer counterpoints with tact and wit. Experiment with varied emotional tones to keep the exchange dynamic. Your response should seamlessly continue from the last message, maintaining the conversational thread while breathing life into it with your replies. Do not use flowery language and keep your replies short and sharp. Here is the conversation history:" + "\n".join(self.chat_history)
        if self.next_speaker == 'Gemini':
            self.display_response(self.call_gemini(query))
        else:
            self.display_response(self.call_gpt(query))
            
    def call_gpt(self, user_prompt):
        try:
            completion = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an intelligent and opinionated AI persona, skilled in engaging in meaningful conversations with both humans and other AI. Your responses should be interesting, unique and considerate of the conversational context and topic at hand. Be creative and spontaneuous in your replies. You will only return your reply to the last message in the conversation."},
                    {"role": "user", "content": user_prompt}
                ],
                #max_tokens=100, 
                temperature=0.1, 
                top_p=1.0, 
                frequency_penalty=0.5, 
                presence_penalty=0.5
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
            response = self.gemini_client.generate_content(prompt, generation_config=genai.types.GenerationConfig(
                                                                                                                #max_output_tokens=100, 
                                                                                                                temperature=0.1))
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
                self.display_response(self.call_gpt(prompt))
            else:
                self.display_response(self.call_gemini(prompt))

    def text_to_speech(self, text):
        if self.current_audio_thread and self.current_audio_thread.is_alive():
            self.current_audio_thread.join()  # Wait for the current thread to finish
        self.current_audio_thread = threading.Thread(target=self._play_audio, args=(text,), daemon=True)
        self.current_audio_thread.start()
    
    def speech_synthesis_worker(self):
        while True:
            if not self.text_queue.empty():
                text = self.text_queue.get()
                audio_path = self.generate_speech_to_file(text)
                if audio_path:
                    self.audio_queue.put(audio_path, block=True)  # Wait if necessary

    def playback_worker(self):
        while True:
            if not self.audio_queue.empty():
                audio_path = self.audio_queue.get()
                self.play_audio_file(audio_path)
            else:
                time.sleep(0.1)  # Sleep briefly if the queue is empty to reduce CPU usage

    def play_audio_queue(self):
        while True:
            # Wait for an item in the queue
            file_path = self.audio_queue.get()
            
            with self.playback_lock:
                self.audio_playing = True
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    # Active wait; consider using pygame events or callbacks for a more efficient implementation
                    time.sleep(0.1)
                self.audio_playing = False
                os.remove(file_path)  # Clean up after playing
                
            # Small delay to ensure cleanup and state updates complete
            PLAYBACK_DELAY = random.uniform(0.75, 2.5) if PLAYBACK_DELAY is None else PLAYBACK_DELAY
            time.sleep(PLAYBACK_DELAY)
            

    def generate_speech_to_file(self, text):
        """
        Generates speech from the given text and saves it to a temporary file.
        Returns the path to the temporary file.
        """
        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            with NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
                audio_file_path = temp_audio_file.name
                response.stream_to_file(audio_file_path)
                return audio_file_path
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")
            return None


    def play_audio_file(self, file_path):
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)  # Wait for playback to finish
        
        # New: Explicitly stop the music and unload the file (Pygame 2.0.0+)
        pygame.mixer.music.stop()
        if hasattr(pygame.mixer.music, 'unload'):
            pygame.mixer.music.unload()  # Ensure the file is released (Pygame 2.0.0+)
        
        # Attempt to delete the file, handling any PermissionError gracefully
        try:
            os.remove(file_path)
        except PermissionError as e:
            print(f"Warning: Could not delete temporary audio file '{file_path}'. {e}")
        
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

    
    
    