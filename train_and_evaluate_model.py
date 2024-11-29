# Import necessary libraries
import os  # For operating system interactions (e.g., file paths)
import pandas as pd  # For handling and processing tabular data
import numpy as np  # For numerical operations
import torch  # PyTorch for deep learning
import torch.nn as nn  # Neural network modules in PyTorch
import torch.optim as optim  # Optimization algorithms in PyTorch
import torch.nn.functional as F  # Functional tools for neural networks
from torch.utils.data import DataLoader, Dataset  # For handling datasets and batching
from sklearn.model_selection import train_test_split  # For splitting datasets into train and test sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For feature scaling and label encoding
from sklearn.metrics import accuracy_score, confusion_matrix  # For evaluating model performance
from sklearn.utils.class_weight import compute_class_weight  # For computing class weights to address imbalance
from tqdm import tqdm  # For displaying progress bars
import cv2  # OpenCV for image processing
import matplotlib.pyplot as plt  # For plotting visualizations
import seaborn as sns  # For heatmaps and advanced visualizations

# Load the combined dataset from a CSV file
base_dir = "path/to/EmotionDatasetImageFolder" # Base directory containing the image folders
csv_file = "path/to/emotion_dataset_face_data.csv" # Path to the preprocessed CSV file containing image paths and labels
combined_df = pd.read_csv(csv_file)

# Separate features (X) and labels (y) from the dataset
X = combined_df.drop(columns=["Picture", "Label"])  # Drop non-feature columns
y = combined_df["Label"]  # Target variable containing labels

# Split the data into training and testing sets (80-20 split), stratified by label distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Encode labels (categorical to numerical) using LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)  # Fit encoder to training labels and transform
y_test = label_encoder.transform(y_test)  # Transform testing labels using the same encoder

# Standardize feature values using StandardScaler (mean=0, variance=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit scaler to training data and transform
X_test = scaler.transform(X_test)  # Transform testing data using the same scaler

# Define a function to augment landmark data by scaling, adding noise, and rotation
def augment_landmarks(X, scale_range=0.1, noise_level=0.02, rotation_range=5):
    """
    Apply data augmentation to landmarks by adding noise, scaling, and rotation.

    Parameters:
        X (numpy.ndarray): Landmark data with shape (n_samples, n_features).
        scale_range (float): Range for random scaling.
        noise_level (float): Standard deviation of random noise.
        rotation_range (int): Maximum rotation angle in degrees.

    Returns:
        numpy.ndarray: Augmented landmark data.
    """
    n_samples, n_features = X.shape
    assert n_features % 2 == 0, "Number of features must be even (x, y pairs)."  # Ensure pairs of coordinates
    n_points = n_features // 2  # Number of landmark points
    X_reshaped = X.reshape(n_samples, n_points, 2)  # Reshape to (n_samples, n_points, 2)

    # Add random noise to landmarks
    noise = np.random.normal(0, noise_level, X_reshaped.shape)

    # Apply random scaling to landmarks
    scale = np.random.uniform(1 - scale_range, 1 + scale_range, size=(n_samples, 1, 1))
    X_scaled = X_reshaped * scale

    # Apply random rotation to landmarks
    angles = np.random.uniform(-rotation_range, rotation_range, size=n_samples)
    rotation_matrices = np.array([
        [[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
         [np.sin(np.radians(angle)),  np.cos(np.radians(angle))]]
        for angle in angles
    ])
    X_rotated = np.einsum('npi,nij->npj', X_scaled, rotation_matrices)  # Perform matrix multiplication for rotation

    # Reshape back to original format and add noise
    X_augmented = X_rotated.reshape(n_samples, n_features) + noise.reshape(n_samples, n_features)
    return X_augmented

# Apply augmentation to the training dataset
X_train = augment_landmarks(X_train)

# Define a PyTorch Dataset class to handle features and labels
class FaceDataset(Dataset):
    def __init__(self, X, y):
        """
        Initialize the dataset with features and labels.

        Parameters:
            X (numpy.ndarray): Input features (landmarks).
            y (numpy.ndarray): Labels (encoded emotions).
        """
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert features to PyTorch tensors
        self.y = torch.tensor(y, dtype=torch.long)  # Convert labels to PyTorch tensors

    def __len__(self):
        return len(self.y)  # Return the total number of samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # Return a single sample at the specified index

# Create PyTorch datasets for training and testing
train_dataset = FaceDataset(X_train, y_train)
test_dataset = FaceDataset(X_test, y_test)

# Create DataLoaders for batching data and enabling shuffling during training
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Shuffle training data
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # No shuffling for testing data

# Define a CNN model for landmark classification
class CNNForLandmarks(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        Initialize the CNN model.

        Parameters:
            input_dim (int): Number of input features (landmarks).
            num_classes (int): Number of output classes (emotions).
        """
        super(CNNForLandmarks, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)  # First convolutional layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Max pooling layer
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)  # Second convolutional layer
        self.fc1 = nn.Linear(32 * (input_dim // 4), 256)  # Fully connected layer
        self.fc2 = nn.Linear(256, num_classes)  # Output layer
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension (1D convolution)
        x = self.pool(F.relu(self.conv1(x)))  # Apply first convolution, ReLU, and pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply second convolution, ReLU, and pooling
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = F.relu(self.fc1(x))  # Apply ReLU to the fully connected layer
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Output layer
        return x

# Initialize the CNN model
input_dim = X_train.shape[1]  # Number of input features
num_classes = len(np.unique(y_train))  # Number of classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

cnn_model = CNNForLandmarks(input_dim, num_classes).to(device)  # Move model to device

# Compute class weights to address class imbalance
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)  # Convert to PyTorch tensor

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss(weight=class_weights)  # Weighted cross-entropy loss
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3, weight_decay=1e-5)  # Adam optimizer
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)  # Reduce LR on plateau

# Define the training function
def train(model, loader, criterion, optimizer, scheduler, epochs=20):
    """
    Train the CNN model.

    Parameters:
        model (nn.Module): The CNN model.
        loader (DataLoader): DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer for updating weights.
        scheduler: Learning rate scheduler.
        epochs (int): Number of epochs.

    Returns:
        list: Training loss over epochs.
    """
    model.train()  # Set model to training mode
    loss_curve = []  # Track loss over epochs
    for epoch in range(epochs):
        running_loss = 0.0  # Track loss for the epoch
        for inputs, labels in tqdm(loader):  # Iterate over batches with a progress bar
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device

            optimizer.zero_grad()  # Zero the gradient buffers
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagate
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients to prevent exploding gradients
            optimizer.step()  # Update weights

            running_loss += loss.item()  # Accumulate loss for the epoch

        scheduler.step(running_loss)  # Adjust learning rate
        epoch_loss = running_loss / len(loader)  # Compute average loss for the epoch
        loss_curve.append(epoch_loss)  # Append to loss curve
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")  # Print epoch loss
    return loss_curve  # Return the loss curve

# Define the evaluation function
def evaluate(model, loader):
    """
    Evaluate the CNN model on the test dataset.

    Parameters:
        model (nn.Module): The CNN model.
        loader (DataLoader): DataLoader for testing data.

    Returns:
        tuple: Accuracy and confusion matrix.
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []  # Store predictions
    all_labels = []  # Store true labels
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
            outputs = model(inputs)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Get predicted class indices
            all_preds.extend(preds.cpu().numpy())  # Collect predictions
            all_labels.extend(labels.cpu().numpy())  # Collect true labels

    accuracy = accuracy_score(all_labels, all_preds)  # Compute accuracy
    cm = confusion_matrix(all_labels, all_preds)  # Compute confusion matrix
    print(f"Accuracy: {accuracy * 100:.2f}%")  # Print accuracy
    print(f"Confusion Matrix:\n{cm}")  # Print confusion matrix
    return accuracy, cm  # Return accuracy and confusion matrix

# Train the model for 30 epochs
loss_curve = train(cnn_model, train_loader, criterion, optimizer, scheduler, epochs=30)

# Evaluate the model on the test dataset
accuracy, cm = evaluate(cnn_model, test_loader)

# Define a function to visualize training loss and confusion matrix
def visualize_loss_and_confusion_matrix(loss_curve, cm, labels, accuracy):
    """
    Visualize the loss curve and confusion matrix.

    Parameters:
        loss_curve (list): Training loss over epochs.
        cm (numpy.ndarray): Confusion matrix.
        labels (list): Class labels.
        accuracy (float): Final accuracy of the model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Create subplots for loss and confusion matrix

    # Plot loss curve
    axes[0].plot(loss_curve, marker='o', color='blue')
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss Curve")

    # Annotate accuracy on the plot
    axes[0].text(
        0.95, 0.95, f"Accuracy: {accuracy * 100:.2f}%",
        horizontalalignment='right', verticalalignment='top',
        transform=axes[0].transAxes,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
    )

    # Plot confusion matrix as a heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Confusion Matrix")

    plt.tight_layout()  # Adjust layout for readability
    plt.show(block=False)  # Display the figure

# Define a function to visualize predictions using images and labels
def visualize_predictions(csv_file, model, label_encoder, base_dir, num_samples=12):
    """
    Visualize model predictions on a subset of images.

    Parameters:
        csv_file (str): Path to the CSV file containing image paths and labels.
        model (nn.Module): Trained model for predictions.
        label_encoder (LabelEncoder): Encoder to decode predictions.
        base_dir (str): Base directory containing the images.
        num_samples (int): Number of samples to visualize.
    """
    df = pd.read_csv(csv_file)  # Load the CSV file
    random_samples = df.sample(num_samples)  # Randomly sample images

    # Determine grid layout for displaying images
    cols = 4
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()  # Flatten axes for easy indexing

    for i, (_, row) in enumerate(random_samples.iterrows()):
        image_name = row['Picture']  # Get image filename
        actual_folder = row['Label']  # Get true label
        image_path = os.path.join(base_dir, actual_folder, image_name)  # Construct full image path

        image = cv2.imread(image_path)  # Read the image
        if image is None:
            print(f"Failed to load image: {image_path}")  # Skip if image is not found
            continue

        landmarks = row.drop(['Picture', 'Label']).values.astype(float)  # Extract landmarks
        landmarks = scaler.transform([landmarks])  # Scale landmarks
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).to(device)  # Convert to PyTorch tensor

        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            output = model(landmarks_tensor)  # Predict using the model
            predicted_label_idx = torch.argmax(output, dim=1).item()  # Get predicted label index
            predicted_label = label_encoder.inverse_transform([predicted_label_idx])[0]  # Decode prediction

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB

        ax = axes[i]  # Get subplot axis
        ax.imshow(image_rgb)  # Display image
        ax.axis('off')  # Remove axis ticks
        ax.set_title(f"Actual: {actual_folder}\nPredicted: {predicted_label}", fontsize=8, loc='center')

    # Turn off unused axes
    for ax in axes[len(random_samples):]:
        ax.axis('off')

    plt.tight_layout()  # Adjust layout
    plt.show()  # Display the figure

# Visualize the loss curve and confusion matrix
visualize_loss_and_confusion_matrix(loss_curve, cm, label_encoder.classes_, accuracy)

# Visualize predictions on a subset of images
visualize_predictions(csv_file, cnn_model, label_encoder, base_dir, num_samples=12)
