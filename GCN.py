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

supercategory_dict = {}
for cat in coco_data["categories"]:
    supercategory = cat["supercategory"]
    name = cat["name"]
    if supercategory not in supercategory_dict:
        supercategory_dict[supercategory] = []
    supercategory_dict[supercategory].append(name)

print(json.dumps(supercategory_dict, indent=4, ensure_ascii=False))
