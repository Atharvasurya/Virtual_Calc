import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Set up the drawing canvas
canvas = np.zeros((720, 1280, 3), dtype="uint8")

# Hand gesture detection
def detect_gestures(hand_landmarks):
    if not hand_landmarks:
        return None

    landmarks = hand_landmarks[0].landmark

    # Check for different gestures
    if landmarks[8].y < landmarks[6].y and landmarks[4].y > landmarks[3].y:  # Index finger up
        return 'draw'
    if landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y:  # Two fingers up
        return 'navigate'
    if landmarks[4].y < landmarks[2].y:  # Thumb up
        return 'reset'
    if landmarks[20].y < landmarks[18].y:  # Small finger up
        return 'submit'

    return None

def draw_on_canvas(img, canvas, hand_landmarks):
    h, w, _ = img.shape
    if hand_landmarks:
        lm = hand_landmarks[0].landmark
        x, y = int(lm[8].x * w), int(lm[8].y * h)
        cv2.circle(canvas, (x, y), 5, (255, 255, 255), -1)
    return canvas

def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        canvas_resized = cv2.resize(canvas, (w, h))

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gestures(results.multi_hand_landmarks)

                if gesture == 'draw':
                    draw_on_canvas(frame, canvas, results.multi_hand_landmarks)
                elif gesture == 'navigate':
                    cv2.putText(frame, 'Navigating...', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif gesture == 'reset':
                    canvas.fill(0)
                elif gesture == 'submit':
                    # Here you would send the canvas to the AI model
                    # Example: send_to_gemini_model(canvas)
                    cv2.putText(frame, 'Submitting...', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Overlay the canvas on the frame
        frame = cv2.add(frame, canvas_resized)

        cv2.imshow('Virtual Calculator', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
