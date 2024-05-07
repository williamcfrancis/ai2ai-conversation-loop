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
import random
from queue import Queue
import cv2
from PIL import PngImagePlugin
import tkinter as tk
from tkinter import scrolledtext
import signal
import sys
from playsound import playsound
from tkVideoPlayer import TkinterVideo
from elevenlabs.client import ElevenLabs
from elevenlabs import stream, save
import requests
import io
import base64


# Configuration constants
ENERGY_THRESHOLD = 400  # minimum audio energy to consider for recording
PAUSE_THRESHOLD = 1.5  # seconds of non-speaking audio before a phrase is considered complete
SAVE_HISTORY_LAST_N = 6  # Number of last messages to save in the conversation history
PLAYBACK_DELAY = random.uniform(0.2, 1)  # Delay between playing back pre-generated audio files. Reduce this to speed up the conversation. None for random delay - more human-like
FIRST_SPEAKER = 'GPT'  # The first speaker in the conversation
HUMAN_INTERACTION_LIMIT =  random.uniform(4, 5) # Number of interactions with a human before resuming the AI conversation
TOPIC_SWIITCH_THRESHOLD = random.uniform(10, 15)  # Number of messages before switching the topic of conversation
MAX_AUDIO_QUEUE_SIZE = 2  # Maximum number of audio files to keep in the queue for playback
MAX_RESPONSE_QUEUE_SIZE = 2  # Maximum number of responses to keep in the queue for speech synthesis
CAMERA_PORT = 0  # Port number for the webcam
VLM_MODEL = 'gpt-vision' # 'gemini-pro-vision'  # The model to use for vision-based interactions
GPT_VOICE = "T5cu6IU92Krx4mh43osx"  # Voice for GPT
GEMINI_VOICE = "fAWgiycvQBMqB5LDGr5G"  # Voice for Gemini


# Configure APIs
OPENAI_API_KEY = os.getenv("openai_api_key") or input("Enter OpenAI API Key: ")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or input("Enter Google API Key: ")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") or input("Enter Eleven Labs API Key: ")
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Wake word patterns
wake_word_pattern_gpt = re.compile(r'h(ey|i)?,?\s+(gpt|chatgpt)', re.IGNORECASE)
wake_word_pattern_gemini = re.compile(r'h(ey|i)?,?\s+(gemini|google)', re.IGNORECASE)

class AI2AI:
    def __init__(self, root):
        pygame.mixer.init()
        self.setup_gui(root)
        self.create_speak_popup()
        self.audio_stop_event = threading.Event()
        self.gemini_client = genai.GenerativeModel('gemini-pro')
        self.openai_client = openai.OpenAI()
        self.sr_recognizer = sr.Recognizer()
        self.sr_recognizer.dynamic_energy_threshold = True
        self.sr_recognizer.pause_threshold = PAUSE_THRESHOLD
        self.sr_microphone = sr.Microphone()
        self.chat_history = collections.deque(maxlen=SAVE_HISTORY_LAST_N)
        self.active_conversation = True
        self.next_speaker = FIRST_SPEAKER  # Determines who speaks next in the AI conversation
        self.interact_with_human = False
        self.topic = args.topic
        self.next_human = False
        self.resume_conversation = False
        self.detected_wave = False
        self.stop_video_thread = False
        self.msg_paused = False
        self.topic_msg_count = 0
        self.GPT_model = "gpt-3.5-turbo"
        # self.GPT_model = "gpt-4-0125-preview"
        self.current_audio_thread = None
        self.text_queue = Queue()  # No max size, handles text for speech synthesis
        self.audio_queue = Queue(maxsize=4)  # Audio files ready for playback
        self.transcription = ""
        self.gui_update_queue = Queue()
        self.speech_synthesis_complete = True
        self.vlm_model = genai.GenerativeModel('gemini-pro-vision')
        self.human_appearance = ""
        


    def start_conversation(self):

        threading.Thread(target=self.conversation_loop, daemon=True).start()
        self.human_detection_thread = threading.Thread(target=self.human_detection_worker, daemon=True)
        self.human_detection_thread.start()
        self.speech_synthesis_thread = threading.Thread(target=self.speech_synthesis_worker, daemon=True)
        self.speech_synthesis_thread.start()
        self.playback_thread = threading.Thread(target=self.playback_worker, daemon=True)
        self.playback_thread.start()
        
        threading.Thread(target=self.monitor_threads, daemon=True).start()

    def monitor_threads(self):
        while True:
            if not self.human_detection_thread.is_alive() and not self.interact_with_human and not self.detected_wave:
                self.human_detection_thread = threading.Thread(target=self.human_detection_worker, daemon=True)
                self.human_detection_thread.start()
                
            if not self.speech_synthesis_thread.is_alive() and not self.interact_with_human and not self.detected_wave:
                self.speech_synthesis_thread = threading.Thread(target=self.speech_synthesis_worker, daemon=True)
                self.speech_synthesis_thread.start()
                
            if not self.playback_thread.is_alive() and not self.interact_with_human and not self.detected_wave:
                self.playback_thread = threading.Thread(target=self.playback_worker, daemon=True)
                self.playback_thread.start()
            time.sleep(0.1)

    #################### GUI ####################
        
    def setup_gui(self, root):
        self.root = root
        self.root.title("AI Conversation")
        self.root.attributes('-fullscreen', True)
        # Define fonts and colors before using them
        self.font_family = "Poppins"
        self.font_size = 16  
        self.gpt_color = "#b7e1fc" # Light blue
        self.gemini_color = "#dde5b6" # Light green
        self.human_color = "#FAD7A0" # Light orange
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

        self.root.config(bg='#003747')  # Dark shade of blue as border/background

        # Custom title bar
        title_bar = tk.Frame(self.root, bg='#2C3E50', relief='raised', bd=2)
        title_bar.pack(fill='x')

        # Flexible space before the title label to center it
        left_space = tk.Frame(title_bar, bg='#121736', width=200)
        left_space.pack(side='left', fill='x', expand=True)

        # Title label centered
        title_label = tk.Label(title_bar, text="AI Conversation", bg='#2C3E50', fg='#ECF0F1', font=(self.font_family, 12, 'bold'))
        title_label.pack(side='left', expand=False)

        # Flexible space after the title label to keep it centered
        right_space = tk.Frame(title_bar, bg='#121736', width=200) 
        right_space.pack(side='left', fill='x', expand=True)

        # Close button on title bar, packed last to appear on the right
        close_button = tk.Button(title_bar, text='X', bg='#2C3E50', fg='#ECF0F1', command=self.root.destroy)
        close_button.pack(side='right')

        # Setup chat display area within a frame for padding
        chat_frame = tk.Frame(self.root, bg=self.bg_color)
        chat_frame.pack(padx=10, pady=10, expand=True, fill='both')
        
        # Status frame setup
        status_frame = tk.Frame(self.root, bg='#121736')  # Main container for the status bar
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)  # Apply padding to match the overall UI design

        # Left status box with darker background color
        self.status_left_frame = tk.Frame(status_frame, borderwidth=2, relief='groove', bg='#333940')  # Darker grey
        self.status_left_frame.pack(side='left', fill=tk.BOTH, expand=True)

        self.status_left = tk.Label(self.status_left_frame, text="The AIs are conversing..", bg='#333940', fg='#FFFFFF',
                                    anchor='center', font=('Helvetica', 12, 'bold'))
        self.status_left.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)  # Padding to ensure text is centered

        # Right status box with darker background color
        self.status_right_frame = tk.Frame(status_frame, borderwidth=2, relief='groove', bg='#2B303B')  # Even darker grey
        self.status_right_frame.pack(side='left', fill=tk.BOTH, expand=True)

        self.status_right = tk.Label(self.status_right_frame, text="Waiting for status...", bg='#2B303B', fg='#FFFFFF',
                                    anchor='center', font=('Helvetica', 12, 'bold'))
        self.status_right.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)  # Padding to ensure text is centered

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

        if sender == "human":
            self.chat_display_area.insert(tk.END, message, tag)
            self.chat_display_area.config(state='normal')
            self.chat_display_area.insert(tk.END, "\n")
            self.chat_display_area.config(state='disabled')
        else: 
            # Loop to insert message character by character
            self.root.after(0, self.insert_message_character_by_character, message, tag)
        
        self.chat_display_area.config(state='disabled')
        self.chat_display_area.see(tk.END)
        
    def insert_message_character_by_character(self, message, tag, index=0):
        if (self.interact_with_human or self.detected_wave) and not self.msg_paused:
            # self.chat_display_area.config(state='disabled')
            self.msg_paused = True
            # insert a newline even if the message is not fully displayed
            self.chat_display_area.config(state='normal')
            self.chat_display_area.insert(tk.END, "\n")
            self.chat_display_area.config(state='disabled')
            return
        if index < len(message):
            self.chat_display_area.config(state='normal')
            self.chat_display_area.insert(tk.END, message[index], tag)
            self.chat_display_area.config(state='disabled')
            self.chat_display_area.see(tk.END)
            # Schedule the next character to be inserted after a short delay
            self.root.after(50, self.insert_message_character_by_character, message, tag, index + 1)
        else:
            # Insert a newline after the message is fully displayed, without specifying a tag
            self.chat_display_area.config(state='normal')
            self.chat_display_area.insert(tk.END, "\n")  # Insert two newlines for spacing without using the tag
            self.chat_display_area.config(state='disabled')
            self.chat_display_area.see(tk.END)

    def update_left_status(self, message, bg_color='#121736'):
        self.status_left.config(text=message, bg=bg_color)

    def update_right_status(self, message, bg_color='#121736'):
        self.status_right.config(text=message, bg=bg_color)
        
    def enqueue_gui_update(self, message, sender="ai"):
        self.gui_update_queue.put((message, sender))
        
    def process_gui_updates(self):
        while not self.gui_update_queue.empty():
            message, sender = self.gui_update_queue.get()
            self.display_message_gui(message, sender)
        self.root.after(100, self.process_gui_updates)
        
    def create_speak_popup(self):
        # Create a borderless Toplevel window for the video
        self.speak_popup = tk.Toplevel(self.root, bg='black')
        self.speak_popup.overrideredirect(True)  # Makes the window borderless

        # Set initial size (this might be adjusted based on the video aspect ratio)
        video_width, video_height = 640, 640  # Example size, adjust as needed
        self.speak_popup.geometry(f"{video_width}x{video_height}")

        # Initialize the video player in the popup window
        self.videoplayer = TkinterVideo(master=self.speak_popup, scaled=True)

        # Load the video file
        self.videoplayer.load(r"./mic_video.mov")
        self.videoplayer.pack(expand=True, fill="both")

        # Initially, hide the popup
        self.speak_popup.withdraw()

        
    def show_speak_popup(self, show=True):
        if show:
            # Calculate the position to center the popup over the root window
            root_x = self.root.winfo_x()
            root_y = self.root.winfo_y()
            root_width = self.root.winfo_width()
            root_height = self.root.winfo_height()

            popup_width = self.speak_popup.winfo_width()
            popup_height = self.speak_popup.winfo_height()

            centered_x = root_x + (root_width - popup_width) // 2
            centered_y = root_y + (root_height - popup_height) // 2

            # Update the popup's geometry to center it
            self.speak_popup.geometry(f"+{centered_x}+{centered_y}")

            # Show the popup and play the video
            self.speak_popup.deiconify()  # Show the window
            self.videoplayer.play()  # Start playing the video
        else:
            # Stop the video and hide the popup
            self.videoplayer.pause()  # Pause or stop the video
            self.speak_popup.withdraw()  # Hide the window


    #################### VISION ####################
        
    def human_detection_worker(self):
        """Continuously captures images from a webcam and checks for human interaction using VLM."""        
        while self.active_conversation:
            if self.interact_with_human or self.detected_wave:
                break
            img = self.capture_image_from_webcam()
            if img:
                if VLM_MODEL == 'gpt-vision':
                    self.send_image_to_gpt("If there is a person waving at you with their hand up, return 'YES', otherwise return 'NO'. If you return 'YES', also include a short description of the person (other than the fact that they are waving) within curly braces. Return 'YES' only if the person if waving their hand at you.", img)
                else:
                    self.send_image_to_gemini("If there is a person waving at you with their hand up, return 'YES', otherwise return 'NO'. If you return 'YES', also include a short description of the person (other than the fact that they are waving) within curly braces. Return 'YES' only if the person if waving their hand at you.", img)
                # time.sleep(0.25)  # Delay to avoid overwhelming the API and the webcam
    
    def capture_image_from_webcam(self):
        """Capture an image from the webcam and return it as a PIL image."""
        cap = cv2.VideoCapture(CAMERA_PORT)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Failed to capture image")
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        return pil_img
    
    def encode_image_to_base64(self, pil_img):
        """Encode PIL image to base64 string."""
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')  # Save PIL image to byte array
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode('utf-8')  # Encode as base64

    def send_image_to_openai(self, base64_image, input_text):
        """Send the base64 encoded image to OpenAI API."""
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": input_text
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail":"low"
                }
                }
            ]
            }
        ],
        "max_tokens": 100
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()

    def send_image_to_gpt(self, input_text, img):
        """Send the captured image along with a prompt to Gemini Pro Vision and check for human interaction"""
        try:
            base64_image = self.encode_image_to_base64(img)
            response = self.send_image_to_openai(base64_image, input_text)
            gpt_response = response['choices'][0]['message']['content']
            #check for curly braces
            if "{" in gpt_response:
                self.human_appearance = re.search(r'\{(.*?)\}', gpt_response).group(1)
            if "YES" in gpt_response:
                self.clear_queues()
                self.interact_with_human = True
                self.detected_wave = True
                self.update_left_status("A human is detected waving. Initiating interaction...")
        except Exception as e:
            print("Failed to send image to GPT Vision:", e)
                
    def send_image_to_gemini(self, input_text, img):
        """Send the captured image along with a prompt to Gemini Pro Vision and check for human interaction."""
        try:
            gemini_response = self.vlm_model.generate_content([input_text, img], stream=False)
            #check for curly braces
            if "{" in gemini_response.text:
                self.human_appearance = re.search(r'\{(.*?)\}', gemini_response.text).group(1)
            gemini_response.resolve()
            if "YES" in gemini_response.text:
                self.clear_queues()
                self.interact_with_human = True
                self.detected_wave = True
                self.update_left_status("A human is detected waving. Initiating interaction...")
        except Exception as e:
            print("Failed to send image to Gemini Pro Vision:", e)
        
    def conversation_loop(self):
        self.begin_conversation()
        while self.active_conversation:
            if self.text_queue.qsize() < MAX_RESPONSE_QUEUE_SIZE:
                if self.topic_msg_count > TOPIC_SWIITCH_THRESHOLD:
                    self.topic = self.get_random_topic()
                    self.topic_msg_count = 0
                    self.moderator_call("Change the topic of conversation to: '" + self.topic + "' naturally and continue the conversation. Your next response must flow into the new topic casually.")
                    self.update_right_status("The moderator has called for a topic change.")
                    
                if self.interact_with_human:
                    self.clear_queues()
                    print("Human appearance: ", self.human_appearance, "\n")
                    self.moderator_call("A human is trying to interact with us. Pause the conversation and respond to the human. Say hi to the human. If appropriate, compliment the person's appearance or pick up on anything peculiar about them for a friendly start:"+ self.human_appearance)
                    self.detected_wave = False
                    self.ai_call()
                    self.next_human = True
                    self.human_to_ai_conversation()
                    self.interact_with_human = False
                    self.msg_paused = False # Reset the message pause flag
                       
                elif not self.interact_with_human:
                    self.update_left_status("The AIs are conversing...")
                    self.ai_call()
                    self.topic_msg_count += 1
            time.sleep(0.01)
    
    def human_to_ai_conversation(self):

        human_interaction_count = 0
        while self.active_conversation and self.interact_with_human:
            if self.next_human and human_interaction_count < HUMAN_INTERACTION_LIMIT: # Human's turn
                with self.sr_microphone as source:
                    self.show_speak_popup(True)
                    self.sr_recognizer.adjust_for_ambient_noise(source)
                    print("Listening for human speech... Speak now\n")
                    self.update_right_status("Listening for human speech... Speak now")
                    try:
                        audio = self.sr_recognizer.listen(source, timeout=10, phrase_time_limit=None)
                        # self.transcription = self.sr_recognizer.recognize_whisper(audio, model="base.en", language="English")
                        self.transcription = self.sr_recognizer.recognize_google(audio, language="English")
                        self.chat_history.append("Human: " + self.transcription + "\n")
                        self.display_response(self.transcription)
                        self.update_right_status("Human speech detected. Processing...")
                        self.show_speak_popup(False)
                    except sr.WaitTimeoutError:
                        print("No speech detected within the time limit.")
                        self.show_speak_popup(False)
                        self.interact_with_human = False  # Stop interaction if no speech is detected within the time limit
                        self.next_human = False
                        self.clear_queues()
                        self.moderator_call("The human has stopped interacting with you. Say goodbye to the human and continue the conversation with the AI.")
                        self.update_right_status("No speech detected within the time limit. Resuming AI conversation...")

                    except sr.UnknownValueError:
                        print("Google Web Speech API could not understand the audio.")
                        self.update_right_status("Audio could not be understood. Please try again.")
                        self.show_speak_popup(False)
                    except sr.RequestError as e:
                        print(f"Could not request results from Google Web Speech API; {e}")
                        self.show_speak_popup(False)
                self.next_human = False
                human_interaction_count += 1
                    
            else:
                self.ai_call()
                self.next_human = True
                
            if human_interaction_count >= HUMAN_INTERACTION_LIMIT:
                if not self.next_human:
                    self.ai_call()                    
                self.interact_with_human = False
                self.next_human = False
                self.clear_queues()
                self.moderator_call("The human has stopped interacting with you. Say goodbye to the human and continue the conversation with the AI.")
                self.update_right_status("Human interaction limit reached. Resuming AI conversation...")
                return
                    
    def clear_queues(self):
        
        with self.text_queue.mutex:
            self.text_queue.queue.clear()
        # while not self.speech_synthesis_complete: # Wait for the current AI response to finish
        #     time.sleep(0.1)
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self.stop_audio()
        
    def moderator_call(self, moderator_prompt):
        self.chat_history.append("Silent Moderator: " + moderator_prompt + "\n")
        print("Moderator: " + moderator_prompt + "\n")
    
    def display_response(self, response):
        if self.interact_with_human and not self.next_human:
            print(self.next_speaker + ": " + response + "\n")
            voice = "onyx" if self.next_speaker == 'GPT' else "nova"
            self.speak(response, voice, sender=self.next_speaker.lower())

        elif not self.interact_with_human and not self.next_human:
            print(self.next_speaker + ": " + response + "\n")
            voice = "onyx" if self.next_speaker == 'GPT' else "nova"
            # checking if the queue is actually getting filled
            self.text_queue.put((response, voice, self.next_speaker.lower())) 
            self.next_speaker = 'GPT' if self.next_speaker == 'Gemini' else 'Gemini'
        else:
            print("Human: " + response + "\n")
            self.enqueue_gui_update(response, "human")
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
            initial_prompt = f"As Google Gemini, an opinionated AI personality, you are starting a casual and human-like conversation with OpenAI's GPT about an interesting topic. The topic of the conversation is {self.topic}. Say hi to GPT."
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

    def ai_call(self):
        if not self.detected_wave:
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
                max_tokens=100, 
                temperature=0.2, 
                top_p=0.8, 
                frequency_penalty=0.5, 
                presence_penalty=0.5
            )
            # Adjusting the way to access the text output based on the actual structure of the response
            if completion.choices and completion.choices[0].message:
                text_output = completion.choices[0].message.content  # Adjusted access here
                cleaned_response = re.sub(r'(\*\*)?(Gemini|GPT|Moderator|Human):\s*\2?\s*', '', text_output)  # Remove the speaker label if present
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
                                                                                                                max_output_tokens=100, 
                                                                                                                temperature=0.2,
                                                                                                                top_p=0.8))
            gemini_response_text = response.text           
            cleaned_response = re.sub(r'(\*\*)?(Gemini|GPT|Moderator|Human):\s*\2?\s*', '', gemini_response_text) # Remove the speaker label if present 
            self.chat_history.append("Gemini: " + cleaned_response + "\n")
            return cleaned_response
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return ""

    #################### AUDIO PROCESSING ####################
            
    # def speak_elevenlabs(self, text, voice, sender): 
    #     """Directly convert text to speech and play it."""
    #     try:
    #         self.enqueue_gui_update(text, sender)
    #         voice = GPT_VOICE if voice == "onyx" else GEMINI_VOICE
    #         audio_stream = elevenlabs_client.generate(
    #                         text=text,
    #                         stream=True,
    #                         voice = voice
    #                         )
    #         stream(audio_stream)
    #     except Exception as e:
    #         print(f"Error in speech synthesis: {e}")
    def speak(self, text, voice, sender): 
        """Directly convert text to speech and play it."""
        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1", 
                voice=voice, 
                input=text
            )
            self.enqueue_gui_update(text, sender)
            # Use a temporary file name but manage the file manually to avoid permission issues
            temp_audio_file_path = tempfile.mktemp(suffix='.mp3')
            with open(temp_audio_file_path, 'wb') as temp_audio_file:
                response.stream_to_file(temp_audio_file.name)
            # Ensure the file is closed before attempting to play it
            playsound(temp_audio_file_path)
            os.remove(temp_audio_file_path)
        except Exception:
            return       
        
    def speech_synthesis_worker(self):
        while True:
            if self.interact_with_human or self.detected_wave:
                break
            if not self.text_queue.empty() and self.audio_queue.qsize() < MAX_AUDIO_QUEUE_SIZE: 
                self.speech_synthesis_complete = False
                text, voice, sender = self.text_queue.get()  # Unpack text, voice, and sender
                
                # Generate audio and add to the queue
                audio_path = self.generate_speech_to_file(text, voice)  # Pass voice to the method
                if audio_path is not None:
                    self.audio_queue.put((audio_path, text, sender))  # Adjusted to remove unused unpacking
                self.speech_synthesis_complete = True
            time.sleep(0.1)
            
    # def generate_speech_to_file_elevenlabs(self, text, voice): # Generate speech from text and save to a temporary file
    #     try:
    #         voice = GPT_VOICE if voice == "onyx" else GEMINI_VOICE
    #         audio = elevenlabs_client.generate(
    #                         text=text,
    #                         stream=False,
    #                         voice = voice
    #                         )
            
    #         with NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
    #             audio_file_path = temp_audio_file.name
    #             save(audio, audio_file_path)
    #             return audio_file_path
    #     except Exception as e:
    #         print(f"Warning in text-to-speech conversion: {e}")
    #         return None
        
    def generate_speech_to_file(self, text, voice): # Generate speech from text and save to a temporary file
        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )
            with NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
                audio_file_path = temp_audio_file.name
                response.stream_to_file(audio_file_path)
                return audio_file_path
        except Exception as e:
            print(f"Warning in text-to-speech conversion: {e}")
            return None
        
    def playback_worker(self): # Worker to play audio files from the queue
        while True:
            if self.interact_with_human or self.detected_wave:
                break
            if not self.audio_queue.empty():
                try:
                    audio_path, text, sender = self.audio_queue.get()
                    self.play_audio_file(audio_path, text, sender)
                except Exception as e:
                    print(f"Error playing audio file: {e}")
            time.sleep(0.1)  # Sleep briefly if the queue is empty to reduce CPU usage
            
             
    def start_audio(self, audio_path):
        self.audio_stop_event.clear()  # Reset the event to allow playing
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

    def stop_audio(self):
        self.audio_stop_event.set()  # Signal to stop
        pygame.mixer.music.stop()
        if hasattr(pygame.mixer.music, 'unload'):
            pygame.mixer.music.unload()

    def check_audio_stop(self):
        return self.audio_stop_event.is_set()
    
    def play_audio_file(self, file_path, text, sender):
        voice = "onyx" if sender == 'gpt' else "nova"
        self.enqueue_gui_update(text, sender)
        # self.speak(text, voice, sender)
        
        # play(file_path)
        self.start_audio(file_path)
        while pygame.mixer.music.get_busy():
            if self.check_audio_stop():
                self.stop_audio()  # Stop audio if the event is set
                break
            time.sleep(0.1)
        
def signal_handler(sig, frame):
    print('Exiting...')
    ai_conversation.active_conversation = False
    root.quit()  # This will break the root.mainloop() blocking call
    root.destroy()  # Ensure the GUI is properly closed
    sys.exit(0)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="AI2AI Conversation Loop")
    parser.add_argument("--topic", type=str, default="default", help="Initial topic for the conversation")
    args = parser.parse_args()
    
    root = tk.Tk()
    ai_conversation = AI2AI(root)
    
    threading.Thread(target=ai_conversation.start_conversation, daemon=True).start()
    ai_conversation.process_gui_updates()
    
    signal.signal(signal.SIGINT, signal_handler) # Handle Ctrl+C
    
    root.mainloop() # Start the GUI event loop
    
    
    
    