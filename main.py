import argparse
import os
import time
import speech_recognition as sr
import threading
import collections
import pygame
import google.generativeai as genai
from PIL import Image
import openai
import re
from tempfile import NamedTemporaryFile
import tempfile
import keyboard
import random
from queue import Queue
import cv2
from PIL import PngImagePlugin
import tkinter as tk
from tkinter import scrolledtext

# Configuration constants
ENERGY_THRESHOLD = 400  # minimum audio energy to consider for recording
PAUSE_THRESHOLD = 3  # seconds of non-speaking audio before a phrase is considered complete
SAVE_HISTORY_LAST_N = 6  # Number of last messages to save in the conversation history
PLAYBACK_DELAY = None  # Delay between playing back pre-generated audio files. Reduce this to speed up the conversation. None for random delay - more human-like
FIRST_SPEAKER = 'GPT'  # The first speaker in the conversation

# Configure APIs
OPENAI_API_KEY = os.getenv("openai_api_key") or input("Enter OpenAI API Key: ")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or input("Enter Google API Key: ")
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Wake word patterns
wake_word_pattern_gpt = re.compile(r'h(ey|i)?,?\s+(gpt|chatgpt)', re.IGNORECASE)
wake_word_pattern_gemini = re.compile(r'h(ey|i)?,?\s+(gemini|google)', re.IGNORECASE)

class AI2AI:
    def __init__(self, root):
        pygame.mixer.init()
        self.setup_gui(root)
        self.gemini_client = genai.GenerativeModel('gemini-pro')
        self.openai_client = openai.OpenAI()
        self.sr_recognizer = sr.Recognizer()
        self.sr_microphone = sr.Microphone()
        self.chat_history = collections.deque(maxlen=SAVE_HISTORY_LAST_N)
        self.active_conversation = True
        self.next_speaker = FIRST_SPEAKER  # Determines who speaks next in the AI conversation
        self.interact_with_human = False
        self.topic = args.topic
        self.stop_listening = None
        self.next_human = False
        self.resume_conversation = False
        self.topic_msg_count = 0
        # self.GPT_model = "gpt-3.5-turbo"
        self.GPT_model = "gpt-4-0125-preview"
        self.current_audio_thread = None
        self.text_queue = Queue()  # No max size, handles text for speech synthesis
        self.audio_queue = Queue(maxsize=2)  # Audio files ready for playback
        self.transcription = ""
        self.gui_update_queue = Queue()



    def start_conversation(self):

        threading.Thread(target=self.conversation_loop, daemon=True).start()
        threading.Thread(target=self.human_detection_worker, daemon=True).start()
        threading.Thread(target=self.speech_synthesis_worker, daemon=True).start()
        threading.Thread(target=self.playback_worker, daemon=True).start()
    
    #################### GUI ####################
        
    def setup_gui(self, root):
        self.root = root
        self.root.title("AI Conversation")

        # Define fonts and colors before using them
        self.font_family = "Poppins"
        self.font_size = 14  
        self.gpt_color = "#e9edc9" 
        self.gemini_color = "#faedcd" 
        self.human_color = "#FAD7A0" 
        self.text_color = "#0a0908"  
        self.bg_color = "#FFFBE6"  

        # Set window size and position
        window_width = 600
        window_height = 400
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

        self.root.config(bg='#2C3E50')  # Dark shade of blue as border/background

        # Custom title bar
        title_bar = tk.Frame(self.root, bg='#2C3E50', relief='raised', bd=2)
        title_bar.pack(fill='x')

        # Flexible space before the title label to center it
        left_space = tk.Frame(title_bar, bg='#2C3E50', width=200)
        left_space.pack(side='left', fill='x', expand=True)

        # Title label centered
        title_label = tk.Label(title_bar, text="AI Conversation", bg='#2C3E50', fg='#ECF0F1', font=(self.font_family, 12, 'bold'))
        title_label.pack(side='left', expand=False)

        # Flexible space after the title label to keep it centered
        right_space = tk.Frame(title_bar, bg='#2C3E50', width=200)
        right_space.pack(side='left', fill='x', expand=True)

        # Close button on title bar, packed last to appear on the right
        close_button = tk.Button(title_bar, text='X', bg='#2C3E50', fg='#ECF0F1', command=self.root.destroy)
        close_button.pack(side='right')

        # Setup chat display area within a frame for padding
        chat_frame = tk.Frame(self.root, bg=self.bg_color)
        chat_frame.pack(padx=10, pady=10, expand=True, fill='both')
        
        self.chat_display_area = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=80, height=20,
                                                            font=(self.font_family, self.font_size), padx=15, pady=15)
        self.chat_display_area.pack(expand=True, fill='both')
        self.chat_display_area.config(state='disabled', bg=self.bg_color)  # Matching background

                
    def display_message_gui(self, message, sender="ai"):
        self.chat_display_area.config(state='normal')
        
        # Define sender's display name and tag for styling
        sender_name = {"gpt": "GPT", "gemini": "Gemini", "human": "Human"}.get(sender, "Unknown")
        tag = sender
        bg_color = {"gpt": self.gpt_color, "gemini": self.gemini_color, "human": self.human_color}.get(sender, "#FFFFFF")
        
        # Configure tags for sender's name and message body
        self.chat_display_area.tag_configure(sender + "_name", font=(self.font_family, self.font_size, "bold"), 
                                            foreground=self.text_color, background=bg_color,
                                            spacing1=4, spacing3=4, lmargin1=20, lmargin2=20, rmargin=20)
        self.chat_display_area.tag_configure(tag, background=bg_color, foreground=self.text_color,
                                            font=(self.font_family, self.font_size), lmargin1=20, lmargin2=20,
                                            rmargin=20, spacing3=4, relief='flat', wrap='word')

        # Insert a visual separator for a new message if desired
        separator_tag = "separator"
        self.chat_display_area.tag_configure(separator_tag, spacing1=10)
        self.chat_display_area.insert(tk.END, "\n", separator_tag)
        
        # Insert sender's name with a dedicated tag for background color
        self.chat_display_area.insert(tk.END, sender_name + ": ", sender + "_name")

        # Insert the message body with its own tag
        self.chat_display_area.insert(tk.END, message + "\n\n", tag)

        self.chat_display_area.config(state='disabled')
        self.chat_display_area.see(tk.END)

                
    def enqueue_gui_update(self, message, sender="ai"):
        self.gui_update_queue.put((message, sender))
        
    def process_gui_updates(self):
        while not self.gui_update_queue.empty():
            message, sender = self.gui_update_queue.get()
            self.display_message_gui(message, sender)
        self.root.after(100, self.process_gui_updates)

    #################### VISION ####################
        
    def human_detection_worker(self):
        """Continuously captures images from a webcam and checks for human interaction using VLM."""        
        while self.active_conversation:
            img = self.capture_image_from_webcam()
            if img:
                self.send_image_to_vlm("Check if there is a person trying to interact with you in the image. Specifically, if there is a waving gesture, return 'YES', otherwise return 'NO'", img)
            time.sleep(0.25)  # Delay to avoid overwhelming the API and the webcam
    
    def capture_image_from_webcam(self):
        """Capture an image from the webcam and return it as a PIL image."""
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Failed to capture image")
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        return pil_img

    def send_image_to_vlm(self, input_text, img):
        """Send the captured image along with a prompt to Gemini Pro Vision and check for human interaction."""
        try:
            model = genai.GenerativeModel('gemini-pro-vision')
            gemini_response = model.generate_content([input_text, img], stream=False)
            gemini_response.resolve()
            if "YES" in gemini_response.text:
                self.clear_queues()
                self.interact_with_human = True
                
        except Exception as e:
            print("Failed to send image to Gemini Pro Vision:", e)
        
    def conversation_loop(self):
        self.begin_conversation()
        while self.active_conversation:
            if self.text_queue.qsize() < 2:
                topic_switch_threshold = random.randint(10, 20)  # randomly switch topic after 5 - 12 interactions
                if self.topic_msg_count > topic_switch_threshold:
                    self.topic = self.get_random_topic()
                    self.topic_msg_count = 0
                    self.moderator_call("Thank you for the interesting conversation, GPT and Gemini. Change the topic of conversation to: '" + self.topic + "' naturally and continue the conversation.")
                    
                if self.interact_with_human:
                    self.clear_queues()
                    self.moderator_call("A human is trying to interact with us. Let's pause the conversation and respond to the human. Say hi to the human.")
                    self.ai_to_ai_conversation()
                    self.next_human = True
                    self.human_to_ai_conversation()
                       
                else:
                    self.ai_to_ai_conversation()
                    self.topic_msg_count += 1
                    # print("Chat history: ", self.chat_history)
            time.sleep(0.1)
    
    def human_to_ai_conversation(self):
        self.sr_recognizer.dynamic_energy_threshold = True
        while self.interact_with_human:
            human_interaction_count = 0
            if self.next_human:
                with self.sr_microphone as source:
                    self.sr_recognizer.adjust_for_ambient_noise(source)
                    print("Listening for human speech... \n")
                    try:
                        audio = self.sr_recognizer.listen(source, timeout=10, phrase_time_limit=PAUSE_THRESHOLD)
                        # Recognize speech using Google Web Speech API
                        self.transcription = self.sr_recognizer.recognize_whisper(audio, model="base.en", language="English")
                        self.chat_history.append("Human: " + self.transcription + "\n")
                        self.display_response(self.transcription)
                    except sr.WaitTimeoutError:
                        print("No speech detected within the time limit.")
                        self.interact_with_human = False  # Stop interaction if no speech is detected within the time limit
                    except sr.UnknownValueError:
                        print("Google Web Speech API could not understand the audio.")
                    except sr.RequestError as e:
                        print(f"Could not request results from Google Web Speech API; {e}")
                self.next_human = False
                human_interaction_count += 1
                    
            else:
                self.ai_to_ai_conversation()
                self.next_human = True
                
            if human_interaction_count > 5:
                self.interact_with_human = False
                self.moderator_call("The human has stopped interacting with us. Let's continue the conversation.")
                
            
    def clear_queues(self):
        with self.text_queue.mutex:
            self.text_queue.queue.clear()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        # Stop any currently playing audio
        pygame.mixer.music.stop()
        if hasattr(pygame.mixer.music, 'unload'):
            pygame.mixer.music.unload()
    
    def moderator_call(self, moderator_prompt):
        self.chat_history.append("Silent Moderator: " + moderator_prompt + "\n")
        print("Moderator: " + moderator_prompt + "\n")
    
    def display_response(self, response):
        if self.interact_with_human and not self.next_human:
            print(self.next_speaker + ": " + response + "\n")
            self.display_message_gui(response, self.next_speaker.lower())
            # self.speak(response)

        elif not self.interact_with_human and not self.next_human:
            print(self.next_speaker + ": " + response + "\n")
            self.display_message_gui(response, self.next_speaker.lower())
            # Add the response to queue for speech synthesis
            # self.text_queue.put(response)
            self.next_speaker = 'GPT' if self.next_speaker == 'Gemini' else 'Gemini'
        else:
            print("Human: " + response + "\n")
            self.chat_history.append("Human: " + response + "\n")
            self.next_human = False            

    #################### AI to AI ####################
    
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
        if self.next_speaker == 'Gemini':
            query = "Keep the conversation flowing by embodying the persona of Gemini: An emotional AI with strong opinions, unique perspectives and humanized responses. Be casual and humanize your responses. Bring variance to the conversation with disagreements, counterpoints, and tangential thoughts. Your response should be your persona's relpy to the last message. Keep your replies direct, short and sharp. Here is the conversation history:" + "\n".join(self.chat_history)
            self.display_response(self.call_gemini(query))
        else:
            query = "Keep the conversation flowing by embodying the persona of GPT: An emotional AI with strong opinions, unique perspectives and humanized responses. Be casual and humanize your responses. Bring variance to the conversation with disagreements, counterpoints, and tangential thoughts. Your response should be your persona's relpy to the last message. Keep your replies direct, short and sharp. Here is the conversation history:" + "\n".join(self.chat_history)
            self.display_response(self.call_gpt(query))
            
    def call_gpt(self, user_prompt):
        try:
            completion = self.openai_client.chat.completions.create(
                model=self.GPT_model,
                messages=[
                    {"role": "system", "content": "You are an GPT, an intelligent and opinionated AI persona, skilled in engaging in meaningful conversations with both humans and other AI. Your responses should be interesting, unique and considerate of the conversational context and topic at hand. Be creative and spontaneuous in your replies while offering variance. Bring variance to the conversation with disagreements, counterpoints, and tangential thoughts. Your responses should be humanized. You will only reply to the last message in the conversation."},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=250, 
                temperature=0.2, 
                top_p=1.0, 
                frequency_penalty=0.8, 
                presence_penalty=0.8
            )
            # Adjusting the way to access the text output based on the actual structure of the response
            if completion.choices and completion.choices[0].message:
                text_output = completion.choices[0].message.content  # Adjusted access here
                cleaned_response = re.sub(r'^(Gemini:\s+)*(GPT:\s+)*', '', text_output) # Remove the speaker label if present
                self.chat_history.append("GPT: " + cleaned_response + "\n")
                return cleaned_response
            else:
                print("No response from GPT.")
                return ""
        except Exception as e:
            print(f"Error calling GPT: {e}")
            return ""
        
    def call_gemini(self, prompt):
        try:
            safety_settings = [
                                {
                                    "category": "HARM_CATEGORY_DANGEROUS",
                                    "threshold": "BLOCK_NONE",
                                },
                                {
                                    "category": "HARM_CATEGORY_HARASSMENT",
                                    "threshold": "BLOCK_NONE",
                                },
                                {
                                    "category": "HARM_CATEGORY_HATE_SPEECH",
                                    "threshold": "BLOCK_NONE",
                                },
                                {
                                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                    "threshold": "BLOCK_NONE",
                                },
                                {
                                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                    "threshold": "BLOCK_NONE",
                                },
                            ]
            response = self.gemini_client.generate_content(prompt, safety_settings=safety_settings, generation_config=genai.types.GenerationConfig(
                                                                                                                max_output_tokens=250, 
                                                                                                                temperature=0.2))
            gemini_response_text = response.text           
            cleaned_response = re.sub(r'^(Gemini:\s+)*(GPT:\s+)*(Moderator:\s+)*', '', gemini_response_text) # Remove the speaker label if present 
            self.chat_history.append("Gemini: " + cleaned_response + "\n")
            return cleaned_response
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return ""

    #################### AUDIO PROCESSING ####################
    def speak(self, text):
        """Directly convert text to speech and play it."""
        try:
            voice = "onyx" if self.next_speaker == 'Gemini' else "nova"
            response = self.openai_client.audio.speech.create(
                model="tts-1", 
                voice=voice, 
                input=text
            )
            # Use a temporary file name but manage the file manually to avoid permission issues
            temp_audio_file_path = tempfile.mktemp(suffix='.mp3')
            with open(temp_audio_file_path, 'wb') as temp_audio_file:
                response.stream_to_file(temp_audio_file.name)
            # Ensure the file is closed before attempting to play it
            pygame.mixer.music.load(temp_audio_file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():  # Wait for the audio to finish playing
                time.sleep(0.1)
            # Clean up
            os.remove(temp_audio_file_path)
        except Exception as e:
            print(f"Error in direct text-to-speech conversion: {e}")
    
    def speech_synthesis_worker(self):
        while True:
            if not self.text_queue.empty() and not self.interact_with_human:
                text = self.text_queue.get()
                audio_path = self.generate_speech_to_file(text)
                if audio_path:
                    self.audio_queue.put(audio_path, block=True)  # Wait if necessary
            time.sleep(0.2)

    def playback_worker(self):
        while True:
            if not self.audio_queue.empty() and not self.interact_with_human:
                audio_path = self.audio_queue.get()
                self.play_audio_file(audio_path)
            time.sleep(0.2)  # Sleep briefly if the queue is empty to reduce CPU usage

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
            if self.next_speaker == 'Gemini': 
                    response = self.openai_client.audio.speech.create(
                    model="tts-1", 
                    voice="onyx", # Gemini voice. All available voices: https://beta.openai.com/docs/engines/voice
                    input=text
                )
            else:
                response = self.openai_client.audio.speech.create(
                    model="tts-1",
                    voice="nova", # GPT voice
                    input=text
                )
            with NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
                audio_file_path = temp_audio_file.name
                response.stream_to_file(audio_file_path)
                return audio_file_path
        except Exception as e:
            if not self.interact_with_human: print(f"Warning in text-to-speech conversion: {e}")
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
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="AI2AI Conversation Loop")
    parser.add_argument("--topic", type=str, default="default", help="Initial topic for the conversation")
    args = parser.parse_args()
    
    root = tk.Tk()
    ai_conversation = AI2AI(root)
    
    threading.Thread(target=ai_conversation.start_conversation, daemon=True).start()
    ai_conversation.process_gui_updates()
    root.mainloop()
    
    try:
        while True:
            time.sleep(0.1)
            if keyboard.is_pressed('esc'):
                break
    except KeyboardInterrupt:
        print("Exiting...")
        ai_conversation.stop_listening(wait_for_stop=False)
        
    
    
    