import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
import copy
import matplotlib.pyplot as plt


def evaluate_model_performance(model, loader, device, class_names):
    """
    Evaluates a PyTorch model and displays Precision, Recall, F1, 
    Confusion Matrix, and AUC-ROC curves.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    # Gather Predictions
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Get probabilities using Softmax for AUC-ROC
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    n_classes = len(class_names)

    # Text Report (Precision, Recall, F1)
    print(" CLASSIFICATION REPORT ")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # AUC-ROC Curve
    y_test_bin = label_binarize(all_labels, classes=range(n_classes))
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Return metrics as a dictionary
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

# Configuration
DATASET_DIR = r"C:\Users\vtsar\projects\anveshak\cone_depression\data_main_new"
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset Wrapper
class ApplyTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

# Load Dataset WITHOUT transforms
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=None)
class_names = full_dataset.classes
n = len(full_dataset)

train_n = int(0.7 * n)
val_n = int(0.15 * n)
test_n = n - train_n - val_n

indices = torch.randperm(n)

train_indices = indices[:train_n]
val_indices = indices[train_n:train_n + val_n]
test_indices = indices[train_n + val_n:]

train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)
test_subset = Subset(full_dataset, test_indices)

train_data = ApplyTransform(train_subset, train_transform)
val_data = ApplyTransform(val_subset, eval_transform)
test_data = ApplyTransform(test_subset, eval_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, len(full_dataset.classes))
)
model = model.to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
criterion = nn.CrossEntropyLoss()

# History
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'test_acc': []
}

best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())

# Training Loop
for epoch in range(EPOCHS):

    # Train
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

    epoch_train_loss = running_loss / train_n
    epoch_train_acc = running_corrects.double() / train_n

    # Validation
    model.eval()
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels)

    epoch_val_loss = val_loss / val_n
    epoch_val_acc = val_corrects.double() / val_n

    # Test (per epoch)
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels)
            test_total += labels.size(0)

    epoch_test_acc = (test_correct.double() / test_total).item()

    # Save history
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc.item())
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc.item())
    history['test_acc'].append(epoch_test_acc)

    scheduler.step(epoch_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} "
          f"| Train Acc: {epoch_train_acc:.4f} "
          f"| Val Acc: {epoch_val_acc:.4f} "
          f"| Test Acc: {epoch_test_acc:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())

# Plot Curves
def plot_curves(history):
    epochs = range(len(history['train_loss']))

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.plot(epochs, history['test_acc'], label='Test Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot graphs
plot_curves(history)

# Evaluate model
metrics = evaluate_model_performance(model=model, loader=test_loader, device=DEVICE, class_names=class_names)

# Save model
torch.save(model.state_dict(), r"C:\Users\vtsar\projects\anveshak\cone_depression\model.pth")

# Convert to ONNX

onnx_model_path = r"C:\Users\vtsar\projects\anveshak\cone_depression\model.onnx"

dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11, input_names=['input'], output_names=['output'])

print(f"Done! Saved to {onnx_model_path}")





