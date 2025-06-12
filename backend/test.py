import torch
import cv2
import pyttsx3
from ultralytics import YOLO
import time
import numpy as np
import csv
import os
import speech_recognition as sr

# ----------- Voice & TTS Setup -----------
def speak(text):
    tts = pyttsx3.init()
    tts.setProperty('voice', tts.getProperty('voices')[1].id)
    tts.setProperty('rate', 150)
    tts.say(text)
    tts.runAndWait()

# ----------- Helper: Listen Command -----------
def listen_command(recognizer, mic):
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)
    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"Command received: {command}")
        return command
    except Exception:
        return ""

# ----------- Object Detection + Depth + Motion + Voice Commands -----------
def vision_system():
    model = YOLO("yolov8m.pt")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device).eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    last_spoken = {}
    cooldown = 4
    scale_factor = 0.05
    csv_file = "detections.csv"

    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Object", "Direction", "Distance_m"])

    cap = cv2.VideoCapture(0)
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    prev_frame = None
    speak("Vision system activated. Listening for commands.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb).to(device)

        with torch.no_grad():
            depth = midas(input_tensor)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

        now = time.time()

        detected_objects = []

        # Motion Detection (Frame Difference)
        motion_detected = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if prev_frame is None:
            prev_frame = gray
            motion_detected = False
        else:
            frame_delta = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = len(cnts) > 0
            prev_frame = gray

        if motion_detected:
            speak("Motion detected around you.")

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = (x1 + x2) // 2
            frame_width = frame.shape[1]

            # Determine object position
            if x_center < frame_width // 3:
                direction = "left"
            elif x_center < 2 * frame_width // 3:
                direction = "center"
            else:
                direction = "right"

            # Depth ROI
            roi_depth = depth[y1:y2, x1:x2]
            if roi_depth.size == 0:
                continue

            avg_depth = np.mean(roi_depth)
            if np.isnan(avg_depth) or avg_depth <= 0:
                continue

            distance_m = int(round(avg_depth * scale_factor))
            unique_key = f"{label}_{direction}"

            # Draw bounding box and label on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({distance_m}m)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if unique_key not in last_spoken or now - last_spoken[unique_key] > cooldown:
                message = f"{label} on your {direction}. Distance {distance_m} meters."
                speak(message)
                last_spoken[unique_key] = now

                with open(csv_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), label, direction, distance_m])

            detected_objects.append((label, direction, distance_m))

        # Show the frame with detections
        cv2.imshow("Object Detection", frame)

        # Listen for voice commands (non-blocking)
        print("Listening for command...")
        command = ""
        try:
            command = listen_command(recognizer, mic)
        except:
            pass

        if command:
            if "what's around" in command or "mere aas paas" in command or "kya hai" in command:
                if detected_objects:
                    summary = {}
                    for obj, dirc, dist in detected_objects:
                        key = f"{obj} on your {dirc}"
                        if key in summary:
                            summary[key] += 1
                        else:
                            summary[key] = 1

                    response = "I see "
                    for i, (desc, count) in enumerate(summary.items()):
                        if count > 1:
                            response += f"{count} {desc}s"
                        else:
                            response += desc
                        if i < len(summary) - 1:
                            response += ", "
                        else:
                            response += "."
                    speak(response)
                else:
                    speak("I don't see anything around you right now.")

            elif "save photo" in command or "screenshot" in command or "take a screenshot" in command:
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                speak(f"Photo saved as {filename}.")

            elif "stop" in command or "ruk jao" in command or "exit" in command:
                speak("Stopping vision system. Goodbye.")
                break

            else:
                speak("Sorry, I didn't understand that command.")

        # Exit on 'q' keypress
        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak("Stopping vision system. Goodbye.")
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------- Main -----------
if __name__ == "__main__":
    vision_system()
