import os
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,   # IMPORTANT: per immagini dataset
    max_num_hands=1,
    min_detection_confidence=0.7
)

DATASET_PATH = "LIS-fingerspelling-dataset"  # cartella base

X = []  # keypoints
y = []  # labels

labels_map = {}  # letter -> index
label_index = 0

for label in sorted(os.listdir(DATASET_PATH)):

    print("Processing label:", label)

    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    # assegna id numerico alla classe
    if label not in labels_map:
        labels_map[label] = label_index
        label_index += 1

    for img_name in os.listdir(label_path):

        img_path = os.path.join(label_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                keypoints = []

                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

                X.append(keypoints)
                y.append(labels_map[label])

# converti in numpy
X = np.array(X)
y = np.array(y)

# salva dataset
np.savez("lisy_dataset.npz", X=X, y=y, labels_map=labels_map)

print("Dataset creato!")
print("Samples:", len(X))
print("Features shape:", X.shape)