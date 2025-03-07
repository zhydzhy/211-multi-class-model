from torchmetrics.classification import MultilabelPrecision, MultilabelRecall
import torch
y_pred = torch.tensor([[0, 1, 0], [1, 0, 1]])  # 预测
y_true = torch.tensor([[0, 1, 1], [1, 0, 0]])  # 真实

num_labels = 3  # 3 个标签
precision = MultilabelPrecision(num_labels=num_labels, average="macro")
recall = MultilabelRecall(num_labels=num_labels, average="macro")

print("Multi-label Precision:", precision(y_pred, y_true))
print("Multi-label Recall:", recall(y_pred, y_true))

from torchmetrics.classification import MultilabelF1Score

y_pred = torch.tensor([[0, 1, 0], [1, 0, 1]])  # 预测
y_true = torch.tensor([[0, 1, 1], [1, 0, 0]])  # 真实

num_labels = 3  # 3个标签
f1 = MultilabelF1Score(num_labels=num_labels, average="macro")
print(y_pred.size())
print("Multi-label F1-score:", f1(y_pred, y_true))