import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizerFast, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
import json
import os
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CNNClassifier(nn.Module):
    def __init__(self, bert_model, num_labels=2):
        super(CNNClassifier, self).__init__()
        self.bert = bert_model
        self.conv1 = nn.Conv2d(1, 128, (3, 768))
        self.conv2 = nn.Conv2d(1, 128, (4, 768))
        self.conv3 = nn.Conv2d(1, 128, (5, 768))
        self.fc1 = nn.Linear(128 * 3, 128)
        self.fc2 = nn.Linear(128, num_labels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embedded = bert_output.last_hidden_state.unsqueeze(1)  # (batch_size, 1, seq_len, 768)
        x1 = torch.relu(self.conv1(embedded)).squeeze(3)
        x1 = torch.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = torch.relu(self.conv2(embedded)).squeeze(3)
        x2 = torch.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = torch.relu(self.conv3(embedded)).squeeze(3)
        x3 = torch.max_pool1d(x3, x3.size(2)).squeeze(2)
        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def calculate_weights(labels):
    count_0 = labels.count(0)
    count_1 = labels.count(1)
    weight_0 = 1. / count_0
    weight_1 = 1. / count_1
    total = weight_0 + weight_1
    weight_0 /= total
    weight_1 /= total
    return [weight_0, weight_1]


def prepare_data(data, tokenizer):
    comments = [x['comment'] for x in data]
    encodings = tokenizer(comments, truncation=True, padding=True, max_length=512, return_tensors='pt')
    labels = [x['label'] for x in data]
    return encodings, labels


class SATDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train():
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)

    data_path = './datasets/studied_dataset.json'
    with open(data_path, 'r') as f:
        data_json = json.load(f)

    data = [{'comment': item['comment'], 'label': 0 if item['maldonado_dataset_classification'] == 'WITHOUT_CLASSIFICATION' else 1} for item in data_json]

    train_data, test_valid_data = train_test_split(data, test_size=0.4, random_state=42)
    valid_data, test_data = train_test_split(test_valid_data, test_size=0.5, random_state=42)

    train_encodings, train_labels = prepare_data(train_data, tokenizer)
    valid_encodings, valid_labels = prepare_data(valid_data, tokenizer)
    test_encodings, test_labels = prepare_data(test_data, tokenizer)

    train_dataset = SATDDataset(train_encodings, train_labels)
    valid_dataset = SATDDataset(valid_encodings, valid_labels)
    test_dataset = SATDDataset(test_encodings, test_labels)

    weights = calculate_weights(train_labels)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = torch.tensor(weights).to(device)

    model = CNNClassifier(bert_model).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    def train_model(dataloader):
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def eval_model(dataloader):
        model.eval()
        total_loss = 0
        preds, true_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                preds.extend(batch_preds)
                true_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        fp = sum((np.array(preds) == 1) & (np.array(true_labels) == 0))
        fn = sum((np.array(preds) == 0) & (np.array(true_labels) == 1))
        return total_loss / len(dataloader), preds, true_labels, all_probs, fp, fn

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    num_epochs = 20
    best_valid_loss = float('inf')
    early_stopping_patience = 5
    no_improvement_epochs = 0

    for epoch in range(num_epochs):
        train_loss = train_model(train_loader)
        valid_loss, _, _, _, _, _ = eval_model(valid_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_preds, test_labels, test_probs, fp, fn = eval_model(test_loader)
    f1 = f1_score(test_labels, test_preds, average='binary')
    precision = precision_score(test_labels, test_preds)
    recall = recall_score(test_labels, test_preds)
    roc_auc = roc_auc_score(test_labels, test_probs)
    pr_auc = average_precision_score(test_labels, test_probs)

    print('Test Results')
    print(f"Test Loss: {test_loss:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")

    model_path = "replication-package/BERT_CNN/model_name"
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))
    tokenizer.save_pretrained(model_path)


if __name__ == '__main__':
    train()
