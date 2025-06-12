import torch
import cv2
import pyttsx3
from ultralytics import YOLO
import time
import numpy as np
import csv
import os
import speech_recognition as sr
import threading
import queue
import sys
from datetime import datetime
from collections import defaultdict

# ----------- Voice & TTS Setup -----------
def speak(text):
    tts = pyttsx3.init()
    tts.setProperty('voice', tts.getProperty('voices')[1].id)
    tts.setProperty('rate', 150)
    tts.say(text)
    tts.runAndWait()

# ----------- Helper: Listen Command (continuous in thread) -----------
def listen_loop(recognizer, mic, cmd_queue, stop_event):
    while not stop_event.is_set():
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=4)
                command = recognizer.recognize_google(audio).lower()
                print(f"Command received: {command}")
                cmd_queue.put(command)
            except Exception:
                pass

# ----------- Object Detection + Depth + Motion + Voice Commands -----------
def vision_system():
    # Initialize YOLO model with tracking
    model = YOLO("yolov8m.pt")
    
    # Depth estimation parameters
    focal_length = 910
    known_widths = {
        'person': 0.5, 'car': 1.8, 'chair': 0.45, 'bottle': 0.08,
        'cell phone': 0.075, 'cup': 0.09, 'laptop': 0.3, 'potted plant': 0.25,
        'bed': 1.5, 'vase': 0.15, 'book': 0.2, 'keyboard': 0.4,
        'mouse': 0.1, 'remote': 0.15, 'scissors': 0.15, 'teddy bear': 0.3,
        'hair drier': 0.3, 'toothbrush': 0.02, 'tv': 1.0, 'couch': 2.0,
        'dining table': 1.5, 'refrigerator': 1.0, 'microwave': 0.5,
        'oven': 0.6, 'toaster': 0.3, 'sink': 0.5, 'clock': 0.2
    }

    last_spoken = {}
    cooldown = 4
    csv_file = "detections.csv"

    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Object", "Direction", "Distance_m", "Confidence"])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Error: Could not open camera.")
        return

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    cmd_queue = queue.Queue()
    stop_listening_event = threading.Event()
    listener_thread = threading.Thread(target=listen_loop, args=(recognizer, mic, cmd_queue, stop_listening_event))
    listener_thread.daemon = True
    listener_thread.start()

    prev_frame = None
    speak("Vision system activated. Listening for commands.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Motion detection
        motion_detected = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if prev_frame is None:
            prev_frame = gray
        else:
            frame_delta = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = len(cnts) > 0
            prev_frame = gray

        if motion_detected:
            speak("Motion detected around you.")

        # Run YOLO inference with tracking
        results = model.track(frame, persist=True)
        detected_objects = []
        
        # Get all detections
        if results[0].boxes.id is not None:  # Only process if there are detections with tracking
            boxes = results[0].boxes
            for box, track_id, class_id, conf in zip(boxes.xywh.cpu(), 
                                                   boxes.id.int().cpu().tolist(),
                                                   boxes.cls.int().cpu().tolist(),
                                                   boxes.conf.cpu().tolist()):
                class_name = model.names[class_id]
                
                if conf < 0.5:  # Confidence threshold
                    continue
                    
                if class_name not in known_widths:
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] ID:{track_id} {class_name.ljust(12)} No known width for distance calculation")
                    continue
                    
                # Calculate distance - ensure we convert tensor to float
                x, y, w, h = box.tolist()  # Convert tensor to list
                estimated_depth = float((known_widths[class_name] * focal_length) / w)
                
                # Determine direction
                x_center = x
                frame_width = frame.shape[1]
                direction = "left" if x_center < frame_width // 3 else "center" if x_center < 2 * frame_width // 3 else "right"
                
                # Print to terminal
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] ID:{track_id} {class_name.ljust(12)} Distance: {estimated_depth:.2f}m (Confidence: {conf:.2f})")
                
                # Visual display
                display_text = f"{class_name} ({estimated_depth:.2f}m)"
                text_position = (int(x - w/2), int(y - h/2 - 5))
                cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
                cv2.putText(frame, display_text, text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Speak and record if cooldown has passed
                unique_key = f"{class_name}_{direction}"
                now = time.time()
                
                if unique_key not in last_spoken or now - last_spoken[unique_key] > cooldown:
                    message = f"{class_name} on your {direction}. Distance {estimated_depth:.1f} meters."
                    speak(message)
                    last_spoken[unique_key] = now
                    
                    with open(csv_file, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            class_name,
                            direction,
                            round(estimated_depth, 2),
                            round(float(conf), 2)
                        ])
                
                detected_objects.append((class_name, direction, round(estimated_depth, 2), round(float(conf), 2)))

        # Create the information overlay (invisible bar)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (310, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Display detected objects in the overlay
        for idx, (label, direction, distance, conf) in enumerate(detected_objects[:5]):
            text = f"{label} - {direction} - {distance}m ({conf:.1f})"
            cv2.putText(frame, text, (20, 30 + idx * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Object Detection", frame)

        # Process voice commands
        while not cmd_queue.empty():
            command = cmd_queue.get()
            if "exit" in command or "stop" in command or "ruk jao" in command:
                speak("Stopping vision system. Goodbye.")
                stop_listening_event.set()
                cap.release()
                cv2.destroyAllWindows()
                return

            elif "what's around" in command or "mere aas paas" in command or "kya hai" in command:
                if detected_objects:
                    summary = {}
                    for obj, dirc, dist, conf in detected_objects:
                        key = f"{obj} on your {dirc}"
                        summary[key] = summary.get(key, 0) + 1

                    response = "I see "
                    for i, (desc, count) in enumerate(summary.items()):
                        response += f"{count} {desc}s" if count > 1 else desc
                        response += ", " if i < len(summary) - 1 else "."
                    speak(response)
                else:
                    speak("I don't see anything around you right now.")

            elif "save photo" in command or "screenshot" in command:
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                speak(f"Photo saved as {filename}.")

            else:
                speak("Sorry, I didn't understand that command.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak("Stopping vision system. Goodbye.")
            stop_listening_event.set()
            break

    stop_listening_event.set()
    cap.release()
    cv2.destroyAllWindows()
    return

# ----------- Main -----------
if __name__ == "__main__":
    vision_system()