import cv2
import numpy as np
import mediapipe as mp
import time
import os

# Initialize MediaPipe Hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                     max_num_hands=1,
                     min_detection_confidence=0.7,
                     min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Function to set MacOS volume
def set_volume(volume_percentage):
    volume = int(volume_percentage)
    os.system(f"osascript -e 'set volume output volume {volume}'")

# Variables for FPS calculation
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
        
    # Convert the BGR image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(imgRGB)
    
    # Lists for storing landmark positions
    lmList = []
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw landmarks on the image
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            # Get all landmark positions
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
    
    # If landmarks are detected
    if len(lmList) != 0:
        # Get the positions of thumb and index finger tips
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger
        
        # Draw circles at the tips
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        
        # Draw a line between the tips
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        
        # Calculate the distance between tips
        length = np.hypot(x2 - x1, y2 - y1)
        
        # Convert the range
        # Hand range: 50 - 300
        # Volume range: 0 - 100
        volPer = np.interp(length, [50, 300], [0, 100])
        volBar = np.interp(length, [50, 300], [400, 150])
        
        # Set the volume
        set_volume(volPer)
        
        # Draw volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
    
    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    
    # Display the image
    cv2.imshow("Hand Gesture Volume Control", img)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows() 