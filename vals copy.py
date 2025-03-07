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


all_preds = torch.cat(all_preds, dim=0)
all_labels = torch.cat(all_labels, dim=0)
print(all_labels.size())
# precision = precision_score(all_labels.flatten(), all_preds.flatten(),average='micro', zero_division=0)
# recall = recall_score(all_labels.flatten(), all_preds.flatten(), average='micro',zero_division=0)
# accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
# hamming_loss1 = hamming_loss(all_labels, all_preds)
precision_cal= MultilabelPrecision(num_labels= NUM_CLASSES, average="macro")
recall_cal = MultilabelRecall(num_labels= NUM_CLASSES, average="macro")
f1_cal = MultilabelF1Score(num_labels=NUM_CLASSES, average="macro")
hamming = HammingDistance(task="multilabel", num_labels=NUM_CLASSES)

precision = precision_cal(all_preds, all_labels)
recall = recall_cal(all_preds, all_labels)
hamming_loss1 = hamming(all_preds, all_labels)
accuracy = 1 - hamming_loss1
f1 = f1_cal(all_preds, all_labels)