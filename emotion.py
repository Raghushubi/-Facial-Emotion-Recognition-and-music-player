import cv2
import time
import math
from deepface import DeepFace  
# Music player
import pygame
pygame.mixer.init()

# Start webcam 
cap = cv2.VideoCapture(0)

# Load face classifier 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Window seconds for face detection
window_seconds = 2.0
# If >= 65% of the window says it's a face, then it is a face
promote_thresh = 0.65
# If < 15% of the window says it's a face, then it is not a face
delete_thresh = 0.15

# Window seconds for emotion detection
emotion_window_seconds = 15
# If 60% of the window (for emotion detection) says that the dominant emotion is a particular emotion, then it is the stable emotion
emotion_promote_thresh = 0.6

# A dictionary containing the face_id (an index) as key and another dictionary as value
# The inner dictionary contains (for a specific face):smoothed center, latest (x,y,w,h), window list of 0s and 1s:0 if no face detected in
# that particular frame else 1, emotion window with emotions list per frame, stable emotion in the emotion window, previous stable emotion.
tracked_faces = {}
# index like face_id to be incremented
next_face_id = 0

# Current time
start_time = time.time()
# Total frames (keeps updating throughout the main while loop)
frame_count = 0
# Number of frames per the face detection window size, to be calculated at the start of the loop once
window_size = None
# Is the face detection window size calculated yet
cal = False

# Calculate the center of a face
def center(x,y,w,h):
    return (x+w//2,y+h//2)

# Calculate the euclidean distance between two centers
def distance(c1,c2):
    return math.hypot(c1[0]-c2[0],c1[1]-c2[1])

# Play music based on emotion, keep on looping until emotion changes 
def play_music_for(emotion):
    pygame.mixer.music.load(f"{emotion}.mp3")
    pygame.mixer.music.play(-1)

while True:
    # Capture a frame from webcam
    ret, frame = cap.read()
    # If unable to capture
    if not ret:
        break

    frame_count += 1
    if not cal and time.time()-start_time >= window_seconds:
            window_size = frame_count
            # Number of frames calculated for emotion window seconds 
            emotion_window_size = int((window_size/window_seconds)*emotion_window_seconds)           
            cal = True

    # Convert BGR image to gray scale image for face cascade classifier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Frame ids matched to a detection in current frame
    matched_ids = set()

    for (x,y,w,h) in detections:
        c = center(x,y,w,h)
        # Was this detected face already detected before
        matched = False
        # Checking proportions of detected face
        ar = w/h
        if ar<0.75 or ar>1.3:
            continue

        for face_id,data in tracked_faces.items():
            # Previous weight and height of a particular face in tracked_faces
            pw,ph = data["bbox"][2],data["bbox"][3]
            # If the diff in width or height of current detection and a particular face in tracked_faces is greater than 50% of previous
            if abs(w-pw)>0.5*pw or abs(h-ph)>0.5*ph:
                continue
            
            # If the two centers have less distance than of max of width and height divided by 2
            if distance(c,data["center"]) <= max(w,h)/2:
                # calculate smoothed center with average of previous center of matched face and current center of the same face
                cx = (c[0]+data["center"][0])//2
                cy = (c[1]+data["center"][1])//2
                data["center"] = (cx,cy)
                # Update the most recent values for the matched face
                data["bbox"] = (x,y,w,h)

                if cal:
                    # There is a face detected in this frame for this face id
                    data["window"].append(1)
                    # Detected face
                    face_roi = frame[y:y+h,x:x+w]
                    
                    # Default: when no face is detected
                    emotion = "unknown"
                    try:
                        # Calculate emotion every 5 frames so device doesn't slow down
                        if frame_count % 5 == 0:
                            res = DeepFace.analyze(face_roi,actions=["emotion"],enforce_detection=False)
                            emotion = res[0]["dominant_emotion"]
                    except:
                        pass
                    data["emotion_window"].append(emotion)

                matched_ids.add(face_id)
                matched = True
                break

        if not matched:
            # Add new face
            tracked_faces[next_face_id] = {
                "center": c,
                "bbox": (x,y,w,h),
                "window": [1] if cal else [], 
                "emotion_window": [],
                "stable_emotion": None,
                "last_played_emotion": None
            }
            matched_ids.add(next_face_id)
            # Update face ids index
            next_face_id += 1

    if cal:
        # Faces to delete from tracked_faces
        to_delete = []

        for face_id,data in tracked_faces.items():
            # If face is not found in this frame
            if face_id not in matched_ids:
                data["window"].append(0)
                data["emotion_window"].append("unknown")

            # Sliding face window shift
            if len(data["window"])>window_size:
                data["window"].pop(0)

            # Sliding emotion window shift
            if len(data["emotion_window"])>emotion_window_size:
                data["emotion_window"].pop(0)

            # Number of frames containing the face per the total number of frames in the window 
            confidence = sum(data["window"])/len(data["window"])

            if confidence<delete_thresh:
                to_delete.append(face_id)

        for face_id in to_delete:
            del tracked_faces[face_id]

    if cal:
        # Biggest face in the frame
        leader_id = None
        # Area of the face
        max_area = 0
        for face_id,data in tracked_faces.items():
            x,y,w,h = data["bbox"]
            area = w*h
            if area>max_area:
                max_area = area
                leader_id = face_id

        for face_id,data in tracked_faces.items():
            if face_id != leader_id:
                continue
            
            confidence = sum(data["window"])/len(data["window"])
            if confidence>=promote_thresh:
                x,y,w,h = data["bbox"]

                # Count of emotions in the emotion window
                emotion_counts = {}

                for e in data["emotion_window"]:
                    if e == "unknown":
                        continue
                    if e not in emotion_counts:
                        emotion_counts[e] = 1
                    else:
                        emotion_counts[e] += 1

                # Most occuring emotion in the emotion sliding window
                dom_emotion = None
                # Count of most occuring emotion
                max_count = 0
                # Total counts of all emotions
                total = 0

                for e in emotion_counts:
                    total += emotion_counts[e]
                    if emotion_counts[e]>max_count:
                        max_count = emotion_counts[e]
                        dom_emotion = e

                # Number of frames containing the maximum occuring emotion per total count of frames of all emotions
                emotion_confidence = max_count/total if total>0 else 0

                # If atleast 60% of the emotion sliding window has same dominant emotion
                if emotion_confidence>=emotion_promote_thresh:
                    prev_emotion = data["stable_emotion"]
                    data["stable_emotion"] = dom_emotion

                    # If stable emotion changed
                    if data["stable_emotion"] != prev_emotion:
                        # Stop already playing music
                        pygame.mixer.music.stop()
                        # Play music for new stable emotion
                        play_music_for(data["stable_emotion"])
                        # Store the updated most recently played emotion (to not restart the music even if stable emotion remains same)
                        data["last_played_emotion"] = data["stable_emotion"]

                # Draw a green rectangle around the faces which has enough confidence
                cv2.rectangle(frame,(x, y),(x + w, y + h),(0, 255, 0),2)

                # If stable emotion exists, then display that in green just above the face rectangle
                if data["stable_emotion"] is not None:
                    cv2.putText(frame,data["stable_emotion"],(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2)

    # Render the frame
    cv2.imshow("face emotion detection", frame)

    # Listen for keyboard interrupt with character 'q' to break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and destroy all window displays
cap.release()
cv2.destroyAllWindows()
