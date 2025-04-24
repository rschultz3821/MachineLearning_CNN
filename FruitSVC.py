import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

# region Preprocessing & Load Data
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load dataset
train_dataset = datasets.ImageFolder("train", transform=transform)
test_dataset = datasets.ImageFolder("test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#endregion

# region Feature Extraction using Pretrained Model
device = torch.device("cpu")  # Set the computation device to CPU
resnet = models.resnet18(pretrained=True)  # Load the pre-trained ResNet-18 model from torchvision - downloads and loads weights that were trained on the ImageNet dataset
resnet.fc = torch.nn.Identity()  # Replace the final fully connected layer with an identity function to extract features instead of class scores
resnet = resnet.to(device)  # Move the model to the CPU
resnet.eval()  # Set the model to evaluation mode (disables dropout and batch norm updates)
#endregion

# region Extract features from images using the pretrained ResNet model
def extract_features(dataloader):
    features = []  # Store extracted feature vectors
    labels = []  # Store corresponding labels

    # Disable gradient calculations
    with torch.no_grad():
        # Loop through batches of images and labels in the dataloader
        for images, lbls in dataloader:
            images = images.to(device)  # Move image batch to the CPU

            output = resnet(images).cpu().numpy()  # Pass images through ResNet, move result to CPU, convert to NumPy
            features.extend(output)  # Add feature vectors to the features list
            labels.extend(lbls.numpy())  # Convert labels to NumPy and add to labels list

    # Return features and labels as NumPy arrays
    return np.array(features), np.array(labels)
#endregion

X_train, y_train = extract_features(train_loader)
X_test, y_test = extract_features(test_loader)

# region Train SVC & Evaluate
print("Starting Grid Search for SVC...")
parameters = {
    "C": np.linspace(1, 100, num=5),
    "gamma": np.linspace(0.001, 0.01, num=5)
}
grid = GridSearchCV(SVC(kernel='rbf'), parameters, cv=3)
grid.fit(X_train, y_train)

best_svc = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# Evaluate
y_pred = best_svc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"SVC Test Accuracy: {acc:.4f}")
#endregion

# region Confusion Matrix
cm = confusion_matrix(y_test, y_pred, normalize="true")
disp = ConfusionMatrixDisplay(cm, display_labels=train_dataset.classes)
disp.plot(xticks_rotation=10)
plt.savefig("FruitSVC_confusion_matrix.png", bbox_inches='tight')  # Save the plot to a file
plt.title("Confusion Matrix - SVC on Fruit Images")
plt.show()
#endregion

#Best parameters: {'C': 25.75, 'gamma': 0.001}
#SVC Test Accuracy: 0.8738
