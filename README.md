# AI to AI Conversation Loop

## Overview

An interactive art installation that enables endless dialogues between two leading AI models, GPT and Gemini, with an added dimension of live human interaction. Designed to engage visitors in a unique participatory experience, the installation detects physical gestures, allowing users to seamlessly become part of the AI conversation. This blend of advanced AI conversation capabilities with real-time human input offers an immersive and dynamic interaction, pushing the boundaries of traditional art installations.

## Setup

### Requirements

- Python 3.8+
- SpeechRecognition
- Pygame
- OpenCV
- Pillow
- OpenAI GPT (API key required)
- Google Generative AI (API key required)
- Tkinter for GUI

#### Hardware
- Microphone: For capturing human speech.
- Camera: For detecting human gestures.
- Speakers: For audio output of AI-generated speech.

### Installation

1. Clone the repository:

```
   git clone https://github.com/williamcfrancis/ai2ai-conversation-loop.git
```

2. Install the required Python packages:

```
   pip install -r requirements.txt
```

3. Set up your API keys for OpenAI and Google Generative AI in your environment variables or directly in the script as fallbacks.
   
4. Check your camera port and update the CAMERA_PORT global variable
```
   python check_camera_ports.py
```


### Run

Navigate to the project directory and run:

```
python main.py --topic "default"
```

You can specify a starting conversation topic via the `--topic` argument or let the system choose a random one by default.

## Usage

- **Start the Installation**: Follow the running instructions to start the art installation.
- **Interact with AI**: Visitors can wave their hand in front of the camera to initiate a conversation with the AI. The AI will pause its ongoing conversation and switch to interact with the human.
- **Change Conversation Topics**: The installation autonomously switches topics at random intervals but can also change topics based on visitor interactions.

## Features

- **Endless AI Conversations**: Two AI models engage in continuous dialogue, covering a wide range of topics.
- **Human Interaction Detection**: Utilizes webcam-based gesture recognition to detect when a visitor wishes to interact, seamlessly integrating human input into the AI conversation.
- **Voice Recognition and Speech Synthesis**: Converts human speech to text and generates audible responses, facilitating natural communication with the AI.
- **Dynamic Topic Switching**: The conversation topic changes dynamically, driven by the AI or human interactions, ensuring the dialogue remains engaging and varied.
- **Interactive GUI**: A graphical user interface displays the ongoing conversation, current system status, and provides visual feedback of AI and human contributions.
- **Adaptive Response Timing**: Incorporates a variable playback delay to simulate more natural conversational pauses, making interactions feel more lifelike and less robotic
- **Environmental Awareness**: Adjusts the speech recognition sensitivity dynamically based on the ambient noise level
- **Moderator Intervention**: Features a silent moderator mechanism that can dynamically steer the conversation, initiate topic changes, or facilitate the transition between AI-to-AI and AI-to-human interactions.

## Contributions

We welcome contributions to this project! Whether it's adding new features, improving the code, or suggesting ideas for future development, please feel free to contribute. To get started:

1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes.
4. Push to the branch.
5. Open a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
