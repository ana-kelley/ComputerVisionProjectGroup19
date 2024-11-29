import cv2
import dlib
import mediapipe as mp
import os
import csv
import numpy as np

# Initialize MediaPipe FaceMesh for landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Initialize dlib's face detector and shape predictor (for fallback)
detector = dlib.get_frontal_face_detector()

# Download `shape_predictor_68_face_landmarks.dat` from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
path_to_dlib_model = "path/to/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(path_to_dlib_model)

# Define indices for specific facial landmarks (MediaPipe indices)
# These correspond to areas of interest such as lips, eyes, and eyebrows
lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
rightEyeUpper0 = [246, 161, 160, 159, 158, 157, 173]
rightEyeLower0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]
leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]
leftEyeLower0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]
leftEyebrowUpper = [383, 300, 293, 334, 296, 336, 285, 417]
rightEyebrowUpper = [156, 70, 63, 105, 66, 107, 55, 193]

# Combine all landmark indices for comprehensive data extraction
all_landmarks = (
    lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner
    + rightEyeUpper0 + rightEyeLower0
    + leftEyeUpper0 + leftEyeLower0
    + rightEyebrowUpper + leftEyebrowUpper
)

# Specify the root directory containing subfolders named after emotions
root_directory = "path/to/EmotionDatasetImageFolder" 

# Output CSV file to store the processed data
output_csv = "emotion_dataset_face_data.csv"

# Create the header for the CSV file
header = (
    ["Picture"]  # File path of the image
    + [f"Landmark_{i}_X" for i in all_landmarks]  # X-coordinates of landmarks
    + [f"Landmark_{i}_Y" for i in all_landmarks]  # Y-coordinates of landmarks
    + ["Label"]  # Emotion label derived from folder name
)

# Function to align face using eye coordinates as reference
def align_face(image, left_eye, right_eye, desired_left_eye=(0.35, 0.35), face_width=256, face_height=256):
    """
    Align face based on eye positions and scale.
    
    Parameters:
        image (np.ndarray): Input face image.
        left_eye (tuple): Coordinates of the left eye.
        right_eye (tuple): Coordinates of the right eye.
        desired_left_eye (tuple): Desired left eye position in the output image.
        face_width (int): Desired output image width.
        face_height (int): Desired output image height.
        
    Returns:
        np.ndarray: Aligned face image.
    """
    eye_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    desired_right_eye_x = 1.0 - desired_left_eye[0]
    desired_eye_distance = desired_right_eye_x - desired_left_eye[0]
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    scale = desired_eye_distance * face_width / dist

    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    tX = face_width * 0.5
    tY = face_height * desired_left_eye[1]
    M[0, 2] += (tX - eye_center[0])
    M[1, 2] += (tY - eye_center[1])

    aligned_face = cv2.warpAffine(image, M, (face_width, face_height), flags=cv2.INTER_CUBIC)
    return aligned_face

# Process each image in the dataset
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header to the CSV file

    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)

        if not os.path.isdir(folder_path):
            continue  # Skip if not a folder

        emotion_label = folder_name  # Use folder name as the emotion label

        for image_name in os.listdir(folder_path):
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, image_name)

                image = cv2.imread(image_path)
                if image is None:
                    continue  # Skip if the image cannot be read

                # Preprocess the image
                image = cv2.resize(image, (256, 256))
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_image)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Calculate eye landmarks for alignment
                        left_eye = np.mean([[face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y] for idx in [33, 133]], axis=0)
                        right_eye = np.mean([[face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y] for idx in [362, 263]], axis=0)

                        h, w, _ = image.shape
                        left_eye = (int(left_eye[0] * w), int(left_eye[1] * h))
                        right_eye = (int(right_eye[0] * w), int(right_eye[1] * h))

                        # Align face
                        aligned_image = align_face(image, left_eye, right_eye)

                        # Extract landmarks and save data to CSV
                        row = [image_path]
                        for idx in all_landmarks:
                            x = int(face_landmarks.landmark[idx].x * w)
                            y = int(face_landmarks.landmark[idx].y * h)
                            row.extend([x, y])

                        row.append(emotion_label)
                        writer.writerow(row)
                else:
                    # Fallback to dlib for landmark extraction
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)

                    for face in faces:
                        landmarks = predictor(gray, face)
                        left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
                        right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])

                        # Align face
                        aligned_image = align_face(image, left_eye, right_eye)

                        # Write fallback data (landmarks set to zero as placeholder)
                        row = [image_path]
                        for idx in all_landmarks:
                            row.extend([0, 0])  # Placeholder for missing landmarks

                        row.append(emotion_label)
                        writer.writerow(row)

# Release resources
face_mesh.close()
cv2.destroyAllWindows()
