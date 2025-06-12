import os
import cv2
import time
import numpy as np
import face_recognition
import speech_recognition as sr
import pyttsx3
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from threading import Thread

# ----------- Voice & TTS Setup -----------
def speak(text):
    tts = pyttsx3.init()
    tts.setProperty('voice', tts.getProperty('voices')[1].id)
    tts.setProperty('rate', 150)
    tts.say(text)
    tts.runAndWait()

# ----------- Voice Authentication -----------
def voice_unlock():
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        speak("Say 'open' to unlock.")
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source, timeout=5)
            command = r.recognize_google(audio).lower()
            if "open" in command:
                speak("Voice matched. Proceeding to face recognition.")
                return True
            else:
                speak("Voice not matched. Access denied.")
                return False
        except:
            speak("No voice detected. Exiting.")
            return False

# ----------- Facial Authentication -----------
def face_verification():
    if not os.path.exists("known_face_encoding.npy"):
        cap = cv2.VideoCapture(0)
        speak("Please look at the camera to register your face.")
        time.sleep(2)
        ret, frame = cap.read()
        cap.release()
        cv2.imwrite("known_face.jpg", frame)

        image = face_recognition.load_image_file("known_face.jpg")
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encoding = encodings[0]
            np.save("known_face_encoding.npy", known_encoding)
            speak("Face registered successfully.")
            return True
        else:
            speak("No face found. Exiting.")
            return False
    else:
        known_encoding = np.load("known_face_encoding.npy")
        cap = cv2.VideoCapture(0)
        speak("Please look at the camera for login.")
        time.sleep(2)
        ret, frame = cap.read()
        cap.release()
        cv2.imwrite("live_face.jpg", frame)

        live_image = face_recognition.load_image_file("live_face.jpg")
        live_encodings = face_recognition.face_encodings(live_image)

        if len(live_encodings) > 0:
            live_encoding = live_encodings[0]
            results = face_recognition.compare_faces([known_encoding], live_encoding)
            if results[0]:
                speak("Face matched. Access granted.")
                return True
            else:
                speak("Face not matched. Access denied.")
                return False
        else:
            speak("No face detected. Exiting.")
            return False

class VoiceFaceApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.label = Label(text="Authenticating...", font_size='24sp')
        self.layout.add_widget(self.label)
        Clock.schedule_once(lambda dt: self.start_authentication(), 2)
        return self.layout

    def start_authentication(self):
        Thread(target=self.authenticate_user).start()

    def authenticate_user(self):
        if voice_unlock():
            if face_verification():
                self.update_label("Access granted.")
            else:
                self.update_label("Face auth failed.")
        else:
            self.update_label("Voice auth failed.")

    def update_label(self, text):
        def update(dt):
            self.label.text = text
        Clock.schedule_once(update, 0)

if __name__ == '__main__':
    VoiceFaceApp().run()
