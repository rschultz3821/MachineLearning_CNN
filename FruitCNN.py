import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os

# region Dataset and Transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize all images to 64x64
    # Helps the model learn regardless of small orientation changes.
    transforms.RandomRotation(10), # randomly rotates the image by up to ±10 degrees during training,
    transforms.ToTensor(),        # Convert PIL image to Tensor
])

# https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class/data
train_data = datasets.ImageFolder(root='train', transform=transform)
test_data = datasets.ImageFolder(root='test', transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

class_names = train_data.classes
print("Classes:", class_names)
#endregion

# region CNN Model
class FruitCNN(nn.Module):
    def __init__(self):
        super(FruitCNN, self).__init__()
        # self.features: The investigation — gathering all the evidence from the image
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # Conv layer: input channels=3 (RGB), output channels=32
            nn.BatchNorm2d(32),  # Normalize
            nn.ReLU(),  # Rectified Linear Unit: keeps positive values and replaces all negative values with zero

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # Downsampling using stride convolution
            nn.BatchNorm2d(32),  # Normalize
            nn.ReLU(),  # Rectifier

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Increase feature depth to 64
            nn.BatchNorm2d(64),  # Normalize
            nn.ReLU(),  # Rectifier

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Further downsample with stride=2
            nn.BatchNorm2d(64),  # Normalize
            nn.ReLU(),  # Rectifier

            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Increase depth to 128
            nn.BatchNorm2d(128),  # Normalize
            nn.ReLU(),  # Rectifier

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # Final downsampling stage
            nn.BatchNorm2d(128),  # Normalize
            nn.ReLU(),  # Rectifier

            nn.AdaptiveAvgPool2d((1, 1))  # Reduce each channel to a single value
        )

        # self.classifier: The verdict — deciding which fruit it is based on the clues.
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flattens the input tensor from shape [batch, 128, 1, 1] to [batch, 128]
            nn.Linear(128, 64),  # Fully connected layer reducing 128 features to 64
            nn.BatchNorm1d(64),  # Normalize
            nn.ReLU(),  # Rectifier
            nn.Dropout(0.4),  # Randomly zeroes out 40% of the units to help prevent overfitting
            nn.Linear(64, 5)  # Final output layer with 5 units
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
#endregion

# region Training Function
def train_fruit_cnn(epochs=5, lr=0.001):
    # Set the device to CPU
    device = torch.device("cpu")
    # Initialize the CNN model and move it to the selected device
    model = FruitCNN().to(device)
    # Print the total number of trainable parameters in the model
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")

    # Set up the optimizer and the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Loop over the number of training epochs
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0  # Reset the running loss for this epoch
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Loop through the training data in batches
        for images, labels in tqdm(train_loader):
            # Move images and labels to the selected device
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients from the previous step
            optimizer.zero_grad()
            # Forward pass: get the model's predictions
            outputs = model(images)
            # Compute the loss between predictions and true labels
            loss = loss_fn(outputs, labels)
            # Backward pass: compute gradients
            loss.backward()
            # Update model weights using optimizer
            optimizer.step()
            # Accumulate the batch loss
            running_loss += loss.item()
        # Print average loss for the epoch
        print(f"Loss: {running_loss:.4f}")

        # Evaluation on the test set after each epoch
        model.eval()  # Set the model to evaluation mode
        correct = 0  # Counter for correctly predicted labels
        total = 0  # Counter for total number of samples evaluated

        # Disable gradient calculation to speed up inference and reduce memory usage
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                # Forward pass: compute model outputs
                outputs = model(images)
                # Get predicted class indices by selecting the max logit from output
                _, preds = torch.max(outputs, 1)
                # Count how many predictions matched the true labels
                correct += (preds == labels).sum().item()
                # Update total number of samples processed
                total += labels.size(0)
        print(f"Test Accuracy: {correct / total:.4f}")

    # Evaluation on the test set
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # Disable gradient calculation for faster inference
    with torch.no_grad():
        for images, labels in test_loader:
            # Move test data to the selected device
            images, labels = images.to(device), labels.to(device)
            # Get model predictions
            outputs = model(images)
            # Get predicted class labels
            _, preds = torch.max(outputs, 1)
            # Count how many predictions were correct
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=10)
    plt.savefig("FruitCNN_confusion_matrix.png", bbox_inches='tight')  # Save the plot to a file
    plt.title("Confusion Matrix - CNN on Fruit Images")
    plt.show()

    # Print accuracy on the test set
    print(f"Final Test Accuracy: {correct / total:.4f}")
#endregion

# Run Training
train_fruit_cnn(epochs=100)

#Epoch 100/100
#100%|██████████| 72/72 [00:04<00:00, 14.81it/s]
#Loss: 8.2372
#Test Accuracy: 0.9456
#Final Test Accuracy: 0.9340
