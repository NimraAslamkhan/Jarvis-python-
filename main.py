import speech_recognition as sr
import webbrowser
import pyttsx3
import requests
from gtts import gTTS
import pygame
import os
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load BlenderBot tokenizer and model from Hugging Face
tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')

newsapi = "6046489fc5bd4a518644e5c225d11da1"

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function for speaking text using gTTS and pygame
def speak(text):
    tts = gTTS(text)
    tts.save('temp.mp3') 

    pygame.mixer.init()
    pygame.mixer.music.load('temp.mp3')
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.music.unload()
    os.remove('temp.mp3')

# Function to generate a response using Hugging Face's BlenderBot model
def aiProcess(command):
    inputs = tokenizer([command], return_tensors='pt')
    reply_ids = model.generate(**inputs)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return reply

# Function for executing commands
def processCommand(c):
    if "open google" in c.lower():
        webbrowser.open("https://google.com")
    elif "open facebook" in c.lower():
        webbrowser.open("https://facebook.com")
    elif "open youtube" in c.lower():
        webbrowser.open("https://youtube.com")
    elif "open linkedin" in c.lower():
        webbrowser.open("https://linkedin.com")
    elif "news" in c.lower():
        r = requests.get(f"https://newsapi.org/v2/top-headlines?country=in&apiKey={newsapi}")
        if r.status_code == 200:
            data = r.json()
            articles = data.get('articles', [])
            for article in articles:
                speak(article['title'])
    else:
        output = aiProcess(c)
        speak(output)

# Main function to start the Jarvis assistant
if __name__ == "__main__":
    speak("Initializing Jarvis....")

    recognizer = sr.Recognizer()

    while True:
        try:
            with sr.Microphone() as source:
                print("Listening for 'Jarvis'...")

                # Adjust the timeout to give more time for detection
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

                word = recognizer.recognize_google(audio)

                if word.lower() == "jarvis":
                    speak("Yes, how can I help?")
                    
                    # Listen for the next command
                    print("Jarvis Active, listening for command...")
                    with sr.Microphone() as source:
                        audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                        command = recognizer.recognize_google(audio)

                        processCommand(command)
        except sr.WaitTimeoutError:
            print("Listening timed out, retrying...")
        except sr.UnknownValueError:
            print("Could not understand the audio")
        except Exception as e:
            print(f"Error: {e}")
