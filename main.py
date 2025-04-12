import os
import streamlit as st
from ultralytics import YOLO
import cv2
import random
import time
from gtts import gTTS
import pygame
import threading
from datetime import datetime, timedelta
import numpy as np

# Initialize pygame mixer
pygame.mixer.quit()  # Ensure the mixer is fully stopped
pygame.mixer.init()

# Load YOLOv8 model
yolo = YOLO("yolov8n.pt")

# Streamlit app layout
st.set_page_config(page_title="AI Vision App", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
        font-family: "Arial", sans-serif;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        justify-content: center;
        align-items: center;
        border-radius: 10px;
        padding: 10px;
    }
    .stCheckbox {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI For Visually Impaired People")
st.write("This application provides real-time object recognition with navigation guidance and audio alerts.")


st.markdown("""
    <style>
    img {
        margin-left: 20px;
        border-radius: 10px;
        border: 2px solid #ccc;
        height: 100%;
        width: 100%;
        object-fit: contain;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Directory to store temp audio files
audio_temp_dir = "audio_temp_files"
if not os.path.exists(audio_temp_dir):
    os.makedirs(audio_temp_dir)

# Placeholder for video frames
colimg1, colimg2 = st.columns(2)
with colimg1:
    welcome_image_path = "hero.jpg"
    if os.path.exists(welcome_image_path):
        st.image(welcome_image_path, use_container_width=True, caption="AI For Visually Impaired People")
    else:
        st.warning("Welcome image not found! Please add 'hero.jpg' in the script directory.")
with colimg2:
    stframe = st.empty()

# User controls
col1, col2 = st.columns(2)
with col1:
    start_detection = st.button("Start Detection")
with col2:
    stop_detection = st.button("Stop Detection")
navigation_activation = st.checkbox("Enable Navigation", value=False)
audio_activation = st.checkbox("Enable Audio Alerts", value=False)

# Categories for audio alerts (hazardous objects or living things)
alert_categories = {"person", "cat", "dog", "knife", "fire", "laptop", "bench", "chair", "bottle", "refrigerator", "box"}

# Dictionary to store the last alert timestamp for each object
last_alert_time = {}
alert_cooldown = timedelta(seconds=10)  # 10-second cooldown for alerts

# Function to draw arrow on the frame
def draw_arrow(frame, direction, color=(0, 255, 0), thickness=2):
    height, width = frame.shape[:2]
    arrow_length = min(width, height) // 6
    start_x = width // 2
    start_y = height // 2
    
    # Calculate end points based on direction
    if direction == "left":
        end_x = start_x - arrow_length
        end_y = start_y
    elif direction == "right":
        end_x = start_x + arrow_length
        end_y = start_y
    else:  # forward or other directions
        end_x = start_x
        end_y = start_y - arrow_length
    
    # Draw the arrow
    cv2.arrowedLine(frame, (start_x, start_y), (end_x, end_y), color, thickness, tipLength=0.3)
    
    # Add text instruction
    text = f"Move {direction.upper()}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = start_x - text_size[0] // 2
    text_y = start_y + arrow_length + 30
    
    # Draw a semi-transparent background for the text
    overlay = frame.copy()
    cv2.rectangle(overlay, (text_x - 10, text_y - 25), (text_x + text_size[0] + 10, text_y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw text
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame

def play_audio_alert(label, position, navigation=None):
    """Generate and play an audio alert with navigation guidance."""
    if navigation and navigation_activation:
        phrases = [
            f"There's a {label} on your {position}. {navigation}",
            f"{label} detected on your {position}. {navigation}",
            f"Alert! A {label} is on your {position}. {navigation}",
        ]
    else:
        phrases = [
            f"Be careful, there's a {label} on your {position}.",
            f"Watch out! {label} detected on your {position}.",
            f"Alert! A {label} is on your {position}.",
        ]
    
    caution_note = random.choice(phrases)
    temp_file_path = os.path.join(audio_temp_dir, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp3")

    tts = gTTS(caution_note)
    tts.save(temp_file_path)

    try:
        pygame.mixer.music.load(temp_file_path)
        pygame.mixer.music.play()

        def cleanup_audio_file():
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.stop()
            try:
                os.remove(temp_file_path)
            except OSError as e:
                print(f"Error deleting file {temp_file_path}: {e}")

        threading.Thread(target=cleanup_audio_file, daemon=True).start()

    except Exception as e:
        print(f"Error playing audio alert: {e}")

def determine_navigation(position, label):
    """Determine navigation guidance based on object position and type."""
    avoid_objects = {"knife", "fire", "refrigerator", "box"}
    
    if label in avoid_objects:
        direction = "right" if position == "left" else "left"
        return f"Be careful! Turn {direction}", direction
    elif position == "left" or position == "right":
        direction = "right" if position == "left" else "left"
        return f"Turn {direction}", direction
    else:
        return "Continue straight", "forward"

def process_frame(frame, audio_mode):
    """Process a single video frame for object detection and navigation."""
    results = yolo(frame)
    result = results[0]

    detected_objects = {}
    navigation_instructions = []
    priority_direction = None
    
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]

        if audio_mode and label not in alert_categories:
            continue

        frame_center_x = frame.shape[1] // 2
        obj_center_x = (x1 + x2) // 2
        position = "left" if obj_center_x < frame_center_x else "right"

        # Store detected object info
        detected_objects[label] = position
        
        # Determine navigation guidance
        nav_text, nav_direction = determine_navigation(position, label)
        navigation_instructions.append((nav_text, nav_direction, label))
        
        # Mark priority direction (avoid dangerous objects first)
        if label in {"knife", "fire"}:
            priority_direction = nav_direction
        elif not priority_direction and label in {"person", "chair", "bench"}:
            priority_direction = nav_direction

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Draw navigation arrow if objects detected
    if navigation_activation:
        if navigation_instructions:
            # Choose the highest priority navigation direction
            if not priority_direction and navigation_instructions:
                priority_direction = navigation_instructions[0][1]
                
            # Extract the base direction (remove 'avoid_' prefix if present)
            display_direction = priority_direction.replace('avoid_', '')
            
            # Draw the arrow with appropriate color (red for avoid, blue for approach)
            arrow_color = (0, 255, 0)  # Green for normal navigation
            if 'avoid' in priority_direction:
                arrow_color = (0, 0, 255)  # Red for "avoid" directions
                
            frame = draw_arrow(frame, display_direction, arrow_color)

    return detected_objects, frame, navigation_instructions

# Main logic
if start_detection:
    st.success("Object detection and navigation started.")
    try:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            st.error("Could not access the webcam. Please check your camera settings.")
        else:
            while not stop_detection:
                ret, frame = video_capture.read()
                if not ret:
                    st.error("Failed to capture video. Please check your camera.")
                    break

                detected_objects, processed_frame, navigation_instructions = process_frame(frame, audio_activation)

                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_container_width=True)

                if audio_activation:
                    current_time = datetime.now()
                    for label, position in detected_objects.items():
                        if (label not in last_alert_time or current_time - last_alert_time[label] > alert_cooldown):
                            # Find navigation instruction for this object
                            nav_text = None
                            for instruction in navigation_instructions:
                                if instruction[2] == label:
                                    nav_text = instruction[0]
                                    break        
                            play_audio_alert(label, position, nav_text)
                            last_alert_time[label] = current_time

                time.sleep(0.1)

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        if 'video_capture' in locals() and video_capture.isOpened():
            video_capture.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()

elif stop_detection:
    pygame.mixer.quit()
    st.warning("Object detection stopped.")