import cv2
import mediapipe as mp
import time

from prediction import predict 

def draw_border(frame, color, thickness=5):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w-1, h-1), color, thickness)
    return frame

def overlay_image_alpha(img, overlay, x, y, alpha=0.6):

    h, w = overlay.shape[:2]

    roi = img[y:y+h, x:x+w]

    blended = cv2.addWeighted(roi, 1 - alpha, overlay, alpha, 0)

    img[y:y+h, x:x+w] = blended

    return img

def hand_tracking(shared, speech_event):

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)

    text = ""
    text_lenght = 0
    current_letter = ""
    current_letter_idx = 0

    correctStreak = 0

    while shared["running"]:
        correct_letter = False

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # Disegna la mano
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Estrazione keypoints
                keypoints = []

                for lm in hand_landmarks.landmark:
                    keypoints.append([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

                # print(keypoints)

                letter = predict(keypoints)

                cv2.putText(
                    frame,
                    letter,
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3
                )

                if text != "":
                    if letter == text[current_letter_idx].lower():
                        correctStreak += 1
                        correct_letter = True
                        
                        print(f"Lettera corretta: {letter}, streak: {correctStreak}")

                        if correctStreak >= 50:
                            current_letter_idx += 1
                            correctStreak = 0

                            if current_letter_idx >= text_lenght:
                                current_letter_idx = 0
                                text = ""
                                shared["last_speech"] = ""
                                speech_event.set()

                                print("Hai completato la parola!")

                                time.sleep(2)
                    else:
                        correctStreak = 0
                        correct_letter = False

                        print(f"Lettera sbagliata: {letter}, lettera attesa: {text[current_letter_idx].upper()}, streak: {correctStreak}")
                else:
                    frame = draw_border(frame, (255, 255, 255))
        else:
            frame = draw_border(frame, (255, 255, 255))

        # controllo l'ultimo speech riconosciuto e lo mostro sul frame
        if text == "":
            text = shared.get("last_speech", "")
            text_lenght = len(text)

        if text != "":
            current_letter = text[current_letter_idx].upper()
            while current_letter in ["G", "S", "Z"]:
                print("Problema riscontrato con la seguente lettera:", current_letter)
                current_letter_idx += 1
                if current_letter_idx >= text_lenght:
                    current_letter_idx = 0
                    text = ""
                    shared["last_speech"] = ""
                    speech_event.set()
                    time.sleep(2)
                    break
                current_letter = text[current_letter_idx].upper()

            image_path = f"Lis_images\\{current_letter}.jpeg"

            overlay = cv2.imread(image_path)
            overlay = cv2.resize(overlay, (160, 160))

            h, w = frame.shape[:2]

            x = w - 170
            y = 10

            frame = overlay_image_alpha(frame, overlay, x, y, alpha=0.6)

            if correct_letter:
                frame = draw_border(frame, (0, 255, 0))  # VERDE
            else:
                frame = draw_border(frame, (0, 0, 255))  # ROSSO

        cv2.putText(
            frame,
            text,
            (10, 50),  # posizione (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,
            1,         # grandezza
            (0, 255, 0),  # colore (verde)
            2          # spessore
        )

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            shared["running"] = False
            break

        # 🧠 EXIT se chiudi la finestra con X
        if cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
            shared["running"] = False
            break

    cap.release()
    cv2.destroyAllWindows()