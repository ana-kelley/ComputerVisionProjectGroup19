
# Emotion Recognition using CNN and Facial Landmarks

## Overview
This project implements an emotion recognition system using pre-image processing and a convolutional neural network (CNN). The system leverages keypoint detection to classify emotions based on facial landmarks. It includes:
- Preprocessing and alignment of facial images.
- Training and evaluation of a CNN.
- A baseline random classifier for comparison.

## Repository Contents
1. **`preprocess_landmarks.py`**  
   - Extracts and aligns facial landmarks using MediaPipe and Dlib.  
   - Outputs a CSV file with coordinates for training and testing.  

2. **`random_classifier_baseline.py`**  
   - Implements a random classifier as a baseline for emotion prediction.  
   - Outputs accuracy and visual examples.

3. **`train_and_evaluate_model.py`**  
   - Trains a CNN on preprocessed landmark data.  
   - Evaluates and visualizes performance with metrics and prediction grids.

## Setup Instructions

### 1. Install Required Libraries
Install the required Python packages:
```bash
pip install -r requirements.txt
```


### 2. Resources Needed
1. **Facial Images Dataset**:
   - A directory with subfolders named after emotions (e.g., `happy`, `sad`).
   - Example structure:
     ```
     dataset/
     ├── happy/
     │   ├── img1.jpg
     │   ├── img2.jpg
     ├── sad/
     │   ├── img3.jpg
     │   ├── img4.jpg
     ```

2. **Dlib Shape Predictor Model**:
   - Download `shape_predictor_68_face_landmarks.dat` from [Dlib](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).  
   - Extract and place it in a known location.

3. **Preprocessed Data CSV**:
   - Use `preprocess_landmarks.py` to generate this file.

### 3. Adjusting Placeholder Paths
Replace placeholders in the scripts with appropriate file paths.

#### `preprocess_landmarks.py`
- Replace `path_to_dlib_model` with the path to `shape_predictor_68_face_landmarks.dat`.
- Replace `root_directory` with the path to your dataset.
- Replace `output_csv` with the desired path for the output CSV.

#### `random_classifier_baseline.py`
- Replace `root_directory` with the path to your dataset.

#### `train_and_evaluate_model.py`
- Replace `csv_file` with the path to the preprocessed data CSV.
- Replace `base_dir` with the path to your dataset.

### 4. Running the Scripts

#### Step 1: Preprocess Data
Generate a CSV file of facial landmarks:
```bash
python preprocess_landmarks.py
```
Update the script with:
- Dataset path (`root_directory`).
- Dlib model path (`path_to_dlib_model`).
- Output CSV path (`output_csv`).

#### Step 2: Random Classifier Baseline
Run the baseline random classifier:
```bash
python random_classifier_baseline.py
```

#### Step 3: Train and Evaluate the Model
Train the CNN and evaluate its performance:
```bash
python train_and_evaluate_model.py
```

## Results
- Random Classifier Accuracy: **21.5%**
- CNN Accuracy:
  - Two emotions: **87.98%**
  - Three emotions: **70.66%**
  - Five emotions: **67.22%**

Loss curves and confusion matrices are included to illustrate performance trends.

## Future Work
- Extend emotion detection to real-time webcam footage using OpenCV.
- Test the model on diverse subjects and environments for better generalization.
