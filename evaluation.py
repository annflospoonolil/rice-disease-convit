import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import get_model

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Data Transforms
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================
# Load Test Dataset
# ======================
test_dataset = datasets.ImageFolder(
    root="data/Test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    num_workers = 4,
    pin_memory=torch.cuda.is_available(),
    shuffle=False
)

class_names = test_dataset.classes
num_classes = len(class_names)

# ======================
# Load Model
# ======================
model = get_model(num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

# ======================
# Inference
# ======================
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ======================
# Overall Accuracy
# ======================
overall_acc = accuracy_score(all_labels, all_preds)
print(f"\nOverall Test Accuracy: {overall_acc:.4f}")

# ======================
# Confusion Matrix
# ======================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("Confusion matrix saved as confusion_matrix.png")

# ======================
# Classification Report
# ======================
report = classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    digits=4
)

print("\nClassification Report:\n")
print(report)

with open("classification_report.txt", "w") as f:
    f.write(f"Overall Test Accuracy: {overall_acc:.4f}\n\n")
    f.write(report)

print("Classification report saved as classification_report.txt")

# ======================
# Per-Class Accuracy
# ======================
per_class_acc = cm.diagonal() / cm.sum(axis=1)

print("\nPer-Class Accuracy:")
for i, acc in enumerate(per_class_acc):
    print(f"{class_names[i]}: {acc:.4f}")