import json
from collections import defaultdict
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,hamming_loss
from torch.utils.data import Subset
from torchmetrics.classification import MultilabelF1Score ,MultilabelPrecision, MultilabelRecall, HammingDistance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open('instances_train2017.json', 'r') as file:
    coco_data = json.load(file)

# for top_key in coco_data:
#     first_level_data = coco_data[top_key]
#     if isinstance(first_level_data, dict):
#         child_keys = first_level_data.keys()
#     elif isinstance(first_level_data, list) and len(first_level_data) > 0:
#         child_keys = first_level_data[0].keys()
#     print(f"{top_key}: {list(child_keys)}")

"""
info: ['description', 'url', 'version', 'year', 'contributor', 'date_created']
licenses: ['url', 'id', 'name']
images: ['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id']
{
"license": 3,
"file_name": "000000223648.jpg",
"coco_url": "http://images.cocodataset.org/train2017/000000223648.jpg",
"height": 640,
"width": 490,
"date_captured": "2013-11-14 21:06:15",
"flickr_url": "http://farm6.staticflickr.com/5219/5383892439_6ef5ba6f23_z.jpg",
"id": 223648
}

annotations: ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']
{
"segmentation": [[256.25,27.29,.....,253.3,29.79]],
"area": 393.4210999999999,
"iscrowd": 0,
"image_id": 61909,
"bbox": [251.09,25.81,23.15,21.38],
"category_id": 37,
"id": 300442
}

categories: ['supercategory', 'id', 'name']
{
"supercategory": "indoor",
"id": 89,
"name": "hair drier"
}
"""

"""
image_category: image_id,[category_id]
image_labels:
"""
image_category = defaultdict(list)
for ann in coco_data["annotations"]:
    image_category[ann["image_id"]].append(ann["category_id"])
NUM_CLASSES = 90
image_labels = {}


def get_label_vector(image_id):
    """
    Get the label vector for the given image ID.
    :param image_id: int, the ID of the image
    :return: np.ndarray, a vector indicating the categories for this image
    """

    categories = image_category[image_id]

    label_vector = torch.zeros(NUM_CLASSES, dtype=torch.float32)  # 使用 torch.Tensor 替代 np.zeros

    for cat_id in categories:
        if 1 <= cat_id <= NUM_CLASSES:
            label_vector[cat_id - 1] = 1
    image_labels[image_id] = label_vector
    
    return label_vector

train_dir = "train2017"
val_dir = "val2017"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.CocoDetection(
    root = train_dir,
    annFile = "instances_train2017.json",
    transform = transform
)
num_samples = len(train_data)
num_subset = int(num_samples * 0.05)
subset_indices = list(range(num_subset))  # 前 10% 的索引
train_subset = Subset(train_data, subset_indices)

# get one line for test
# print(train_data[0])

val_data = datasets.CocoDetection(
    root = val_dir,
    annFile = "instances_val2017.json",
    transform = transform
)

num_val_samples = len(val_data)
num_val_subset = int(num_val_samples * 0.1)
val_subset_indices = list(range(num_val_subset))  # 前 10% 的索引
val_subset = Subset(val_data, val_subset_indices)


def collate_fn(batch):
    images = [item[0] for item in batch]
    annotations = [item[1] for item in batch]
    images = torch.stack(images)
    return images, annotations


train_loader = DataLoader(
    train_subset,
    batch_size=4,
    shuffle=True,
    pin_memory=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_subset,
    batch_size=4,
    shuffle=False,
    pin_memory=True,
    collate_fn=collate_fn
)

class MultiLabelResNet(nn.Module):
    def __init__(self, NUM_CLASSES=90):
        super(MultiLabelResNet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),  # Dropout 层
            nn.Linear(2048, NUM_CLASSES)
        )

    def forward(self, x):
        return self.resnet(x)


model = MultiLabelResNet(NUM_CLASSES).to(device)
print(f"Using device: {device}")
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            batch_labels = []

            for t in targets:
                print(t)
                if t is not None and isinstance(t, list) and len(t) > 0 and isinstance(t[0], dict):
                    label = get_label_vector(t[0]["image_id"])
                    # print(label)
                    batch_labels.append(label)
                else:
                    default_label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
                    batch_labels.append(default_label)

            labels = torch.stack(batch_labels).to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    # print(all_labels[0])
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    print(all_preds[0])
    print(all_labels[0])
    # precision = precision_score(all_labels.flatten(), all_preds.flatten(),average='micro', zero_division=0)
    # recall = recall_score(all_labels.flatten(), all_preds.flatten(), average='micro',zero_division=0)
    # f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='micro',zero_division=0)
    # accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
    # hamming_loss1 = hamming_loss(all_labels, all_preds)

    precision_cal= MultilabelPrecision(num_labels= 90, average="micro")
    recall_cal = MultilabelRecall(num_labels= 90, average="micro")
    f1_cal = MultilabelF1Score(num_labels=90, average="micro")
    hamming = HammingDistance(task="multilabel", num_labels=NUM_CLASSES)
    
    precision = precision_cal(all_preds, all_labels)
    recall = recall_cal(all_preds, all_labels)
    hamming_loss1 = hamming(all_preds, all_labels)
    accuracy = 1 - hamming_loss1
    f1 = f1_cal(all_preds, all_labels)
    avg_loss = total_loss / total_samples

    return precision, recall, f1, accuracy, avg_loss, hamming_loss1


precision, recall, f1, subset_accuracy, val_loss,hamming_loss1 = evaluate_model(model, val_loader, device)
print(f"Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Subset Accuracy: {subset_accuracy:.4f}, Val Loss: {val_loss:.4f}, hamming loss:{hamming_loss1:.4f}")
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0

#     for batch_idx, (images, targets) in enumerate(train_loader):
#         batch_labels = []
#         images = images.to(device)
#         for t in targets:
#             if t is not None and isinstance(t, list) and len(t) > 0 and isinstance(t[0], dict):                
#                 label = get_label_vector(t[0]["image_id"])
#                 batch_labels.append(label)
#             else:
#                 default_label = torch.zeros(90, dtype=torch.float32)
#                 batch_labels.append(default_label)

#         labels = torch.stack(batch_labels).to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         print(total_loss)
#         if batch_idx % 5== 0:
#             print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
#     scheduler.step()        

#     precision, recall, f1, subset_accuracy, val_loss,hamming_loss1 = evaluate_model(model, val_loader, device)
#     print(f"Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Subset Accuracy: {subset_accuracy:.4f}, Val Loss: {val_loss:.4f}, hamming loss:{hamming_loss1:.4f}")
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
