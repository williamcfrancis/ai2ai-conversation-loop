import os
from PIL import Image
import google.generativeai as genai
import time
import cv2
from PIL import PngImagePlugin
from openai import OpenAI
import base64
import requests
import io
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
vlm_model = genai.GenerativeModel('gemini-pro-vision')
OPENAI_API_KEY = os.getenv("openai_api_key")

def capture_image_from_webcam():
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

def send_image_to_vlm(input_text, img):
    """Send the captured image along with a prompt to Gemini Pro Vision and check for human interaction."""
    try:
        
        gemini_response = vlm_model.generate_content([input_text, img], stream=False)
        return gemini_response
            
    except Exception as e:
        print("Failed to send image to Gemini Pro Vision:", e)
        
def encode_image_to_base64(pil_img):
    """Encode PIL image to base64 string."""
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='JPEG')  # Save PIL image to byte array
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')  # Encode as base64

def send_image_to_openai(base64_image):
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
              "text": "Check if there is a person trying to interact with you in the image. Specifically, if there is a waving gesture, return 'YES', otherwise return 'NO'. If you return 'YES', also include a short description of the person (other than the fact that they are waving) within curly braces."
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

client = OpenAI()

img = capture_image_from_webcam()

start = time.time()        

# response = send_image_to_vlm("Check if there is a person trying to interact with you in the image. Specifically, if there is a waving gesture, return 'YES', otherwise return 'NO'. If you return 'YES', also include a description of the person (other than the fact that they are waving) within curly braces.", img)

base64_image = encode_image_to_base64(img)
response = send_image_to_openai(base64_image)

end = time.time()
print("OpenAI response: ", response['choices'][0]['message']['content'], "\n")
# print("Gemini Pro Vision response: ", response, "\n")
print("Time taken: ", end-start, " seconds\n")