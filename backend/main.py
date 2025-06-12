import tkinter as tk
from tkinter import ttk, font as tkfont
import pyttsx3
import threading
import speech_recognition as sr
import cv2
import numpy as np
import os
import face_recognition
import subprocess
import sys
from PIL import Image, ImageTk

class VisionIRLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision IRL")
        self.root.geometry("900x650")
        self.setup_ui()
        
        # Initialize global variables
        self.VOICE_REGISTRATION_FILE = "voice_open.wav"
        self.KNOWN_FACE_ENCODING_FILE = "known_face_encoding.npy"
        
        # Start authentication flow in a separate thread
        self.root.after(100, lambda: threading.Thread(target=self.run_authentication_flow, daemon=True).start())
        
    def setup_ui(self):
        # Create gradient background canvas
        self.canvas = tk.Canvas(self.root, width=900, height=650, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.draw_gradient("#FF6D00", "#FFC400")  # Orange to yellow gradient
        
        # Main container frame
        self.main_frame = tk.Frame(self.canvas, bg="#FFFFFF", bd=0)
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center", width=800, height=550)
        
        # Load fonts
        self.load_fonts()
        
        # Setup header with logo
        self.setup_header()
        
        # Setup message display
        self.setup_message_display()
        
        # Setup status bar
        self.setup_status_bar()
        
        # Add initial system message
        self.add_message("System initialized...", "system")
        
    def draw_gradient(self, start_color, end_color):
        """Draw vertical gradient from start_color to end_color"""
        r1, g1, b1 = self.hex_to_rgb(start_color)
        r2, g2, b2 = self.hex_to_rgb(end_color)
        
        for i in range(650):
            r = int(r1 + (r2 - r1) * i / 650)
            g = int(g1 + (g2 - g1) * i / 650)
            b = int(b1 + (b2 - b1) * i / 650)
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.canvas.create_line(0, i, 900, i, fill=color)
    
    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def load_fonts(self):
        """Load custom fonts with fallback to system fonts"""
        try:
            self.title_font = tkfont.Font(family="Montserrat", size=28, weight="bold")
            self.subtitle_font = tkfont.Font(family="Montserrat", size=14)
            self.text_font = tkfont.Font(family="Open Sans", size=12)
            self.bold_font = tkfont.Font(family="Open Sans", size=12, weight="bold")
            self.italic_font = tkfont.Font(family="Open Sans", size=12, slant="italic")
        except:
            self.title_font = tkfont.Font(size=28, weight="bold")
            self.subtitle_font = tkfont.Font(size=14)
            self.text_font = tkfont.Font(size=12)
            self.bold_font = tkfont.Font(size=12, weight="bold")
            self.italic_font = tkfont.Font(size=12, slant="italic")
    
    def setup_header(self):
        """Create the header section with logo"""
        header_frame = tk.Frame(self.main_frame, bg="#FFD600")
        header_frame.pack(fill="x", pady=(0, 20))
        
        try:
            # Load and display logo
            logo_img = Image.open("vision_logo.jpg")
            logo_img = logo_img.resize((150, 80), Image.Resampling.LANCZOS)
            self.logo = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(header_frame, image=self.logo, bg="#FFD600")
            logo_label.pack(pady=10)
        except Exception as e:
            print(f"Error loading logo: {e}")
            # Fallback if logo not found
            title_label = tk.Label(header_frame, text="VISION SYSTEM", 
                                 font=self.title_font, bg="#FFD600", fg="#5D4037")
            title_label.pack(pady=15)
        
        # Divider
        tk.Frame(self.main_frame, height=2, bg="#5D4037").pack(fill="x", padx=50, pady=(0, 20))
    
    def setup_message_display(self):
        """Create the message display area with scrollbar"""
        msg_frame = tk.Frame(self.main_frame, bg="#FFF9C4", bd=2, relief="groove")
        msg_frame.pack(fill="both", expand=True, padx=40, pady=(0, 20))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(msg_frame)
        scrollbar.pack(side="right", fill="y")
        
        # Text widget
        self.text_display = tk.Text(msg_frame, wrap="word", 
                                  font=self.text_font,
                                  bg="#FFF9C4",
                                  fg="#5D4037",
                                  insertbackground="#FFA000",
                                  selectbackground="#FFA000",
                                  selectforeground="white",
                                  padx=20,
                                  pady=20,
                                  bd=0,
                                  yscrollcommand=scrollbar.set)
        self.text_display.pack(fill="both", expand=True)
        scrollbar.config(command=self.text_display.yview)
        
        # Configure text tags
        self.text_display.tag_config("system", font=self.bold_font, foreground="#E65100")
        self.text_display.tag_config("instruction", font=self.italic_font, foreground="#5D4037")
    
    def setup_status_bar(self):
        """Create the status bar at the bottom"""
        status_frame = tk.Frame(self.main_frame, bg="#FFD600", height=30)
        status_frame.pack(fill="x", side="bottom", pady=(10, 0))
        
        self.status_label = tk.Label(status_frame, text="Ready", 
                                   font=self.text_font, bg="#FFD600", fg="#5D4037")
        self.status_label.pack(side="right", padx=20)
    
    def speak(self, text):
        """Text-to-speech with GUI output"""
        self.add_message(text, "instruction")  # Instructions in italics that are spoken
        
        tts = pyttsx3.init()
        voices = tts.getProperty('voices')
        female_voice = next((v.id for v in voices if 'female' in v.name.lower() or 'zira' in v.name.lower()), voices[1].id)
        tts.setProperty('voice', female_voice)
        tts.setProperty('rate', 150)
        tts.say(text)
        tts.runAndWait()
    
    def listen_command(self):
        """Listen for voice commands"""
        r = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            r.adjust_for_ambient_noise(source)
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
                return r.recognize_google(audio).lower()
            except Exception:
                return ""
    
    def register_voice(self):
        """Register user's voice"""
        self.add_message("Proceeding to voice registration...", "system")
        
        r = sr.Recognizer()
        mic = sr.Microphone()

        self.speak("No voice registered. Please say the word 'open' to register your voice.")
        with mic as source:
            r.adjust_for_ambient_noise(source)
            try:
                audio = r.listen(source, timeout=7)
                with open(self.VOICE_REGISTRATION_FILE, "wb") as f:
                    f.write(audio.get_wav_data())
                command = r.recognize_google(audio).lower()
                if "open" in command:
                    self.add_message("Voice registration successful.", "system")
                    self.speak("Voice registered successfully.")
                    return True
                else:
                    self.speak("Did not detect the word 'open'. Please try again.")
                    return self.register_voice()
            except Exception:
                self.speak("Failed to register voice. Please try again.")
                return self.register_voice()
    
    def voice_unlock(self):
        """Voice authentication"""
        if not os.path.exists(self.VOICE_REGISTRATION_FILE):
            if not self.register_voice():
                self.add_message("Voice registration failed. Exiting.", "system")
                self.speak("Voice registration failed. Exiting.")
                self.close_app()

        self.add_message("Starting voice authentication...", "system")
        
        r = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            self.speak("Say 'open' to unlock.")
            r.adjust_for_ambient_noise(source)
            try:
                audio = r.listen(source, timeout=7)
                command = r.recognize_google(audio).lower()
                if "exit" in command:
                    self.add_message("Exit command detected.", "system")
                    self.speak("Exit command detected. Closing application.")
                    self.close_app()
                elif "open" in command:
                    self.add_message("Voice authentication successful.", "system")
                    self.speak("Voice matched. Proceeding to face recognition.")
                    return True
                else:
                    self.add_message("Voice authentication failed.", "system")
                    self.speak("Voice not matched. Access denied.")
                    self.close_app()
            except Exception:
                self.add_message("Voice detection failed.", "system")
                self.speak("No voice detected or recognition failed. Exiting.")
                self.close_app()
    
    def face_verification(self):
        """Face authentication"""
        if not os.path.exists(self.KNOWN_FACE_ENCODING_FILE):
            self.add_message("No face registered. Starting face registration...", "system")
            
            cap = cv2.VideoCapture(0)
            self.speak("No face registered. Please look at the camera to register your face.")
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self.add_message("Failed to capture image.", "system")
                self.speak("Failed to capture image. Exiting.")
                self.close_app()
            cv2.imwrite("known_face.jpg", frame)
            image = face_recognition.load_image_file("known_face.jpg")
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encoding = encodings[0]
                np.save(self.KNOWN_FACE_ENCODING_FILE, known_encoding)
                self.add_message("Face registration successful.", "system")
                self.speak("Face registered successfully.")
            else:
                self.add_message("No face detected during registration.", "system")
                self.speak("No face found during registration. Exiting.")
                self.close_app()
        else:
            self.add_message("Starting face authentication...", "system")
            
            known_encoding = np.load(self.KNOWN_FACE_ENCODING_FILE)
            cap = cv2.VideoCapture(0)
            self.speak("Please look at the camera for face login.")
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self.add_message("Failed to capture image.", "system")
                self.speak("Failed to capture image. Exiting.")
                self.close_app()
            cv2.imwrite("live_face.jpg", frame)

            live_image = face_recognition.load_image_file("live_face.jpg")
            live_encodings = face_recognition.face_encodings(live_image)

            if live_encodings:
                live_encoding = live_encodings[0]
                match = face_recognition.compare_faces([known_encoding], live_encoding)[0]
                if match:
                    self.add_message("Face authentication successful.", "system")
                    self.speak("Face matched. Access granted.")
                else:
                    self.add_message("Face authentication failed.", "system")
                    self.speak("Face not matched. Access denied.")
                    self.close_app()
            else:
                self.add_message("No face detected during authentication.", "system")
                self.speak("No face detected during authentication. Exiting.")
                self.close_app()
    
    def ask_to_repeat(self):
        """Ask user if they want to restart or exit"""
        self.add_message("Authentication complete.", "system")
        self.speak("Do you want to start the vision system again, or exit?")
        command = self.listen_command()
        if "start" in command or "yes" in command or "vision" in command:
            self.add_message("Restarting vision system...", "system")
            self.speak("Starting the vision system again.")
            self.launch_vision()
        elif "exit" in command or "no" in command or "stop" in command:
            self.add_message("User requested exit.", "system")
            self.speak("Thank you and goodbye.")
            self.close_app()
        else:
            self.speak("Sorry, I did not understand. Please say start or exit.")
            self.ask_to_repeat()
    
    def launch_vision(self):
        """Launch vision system"""
        self.root.withdraw()  # Hide window
        subprocess.run([sys.executable, "obj.py"])  # Waits till it finishes
        self.root.deiconify()  # Show window again
        self.ask_to_repeat()
    
    def run_authentication_flow(self):
        """Main authentication flow"""
        self.add_message("Starting authentication process...", "system")
        self.voice_unlock()
        self.face_verification()
        self.add_message("Authentication successful.", "system")
        self.speak("Proceeding to Vision System...")
        self.launch_vision()
    
    def close_app(self):
        """Close the application"""
        self.root.destroy()
        sys.exit()
    
    def add_message(self, text, msg_type="system"):
        """Add a message to the display"""
        self.text_display.insert("end", f"{text}\n", msg_type)
        self.text_display.see("end")
        self.root.update()

def main():
    root = tk.Tk()
    app = VisionIRLApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()