import torch
import numpy as np
from transformers import BertTokenizerFast, BertModel
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
import json
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

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


def test():
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)

    model_path = "replication-package/BERT_CNN/pretrained_models/pretrained_bert_cnn_example"
    model = CNNClassifier(bert_model).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location=torch.device('cpu')))

    data_path = './datasets/studied_dataset.json'
    with open(data_path, 'r') as f:
        data_json = json.load(f)

    data = [{'comment': item['comment'], 'label': 0 if item['maldonado_dataset_classification'] == 'WITHOUT_CLASSIFICATION' else 1} for item in data_json]

    _, test_data = train_test_split(data, test_size=0.2, random_state=42)

    test_encodings, test_labels = prepare_data(test_data, tokenizer)
    test_dataset = SATDDataset(test_encodings, test_labels)

    test_loader = DataLoader(test_dataset, batch_size=16)

    def eval_model(dataloader):
        model.eval()
        total_loss = 0
        preds, true_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):  # tqdmを適用して進捗を表示
                input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
                attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
                labels = batch['labels'].to('cuda' if torch.cuda.is_available() else 'cpu')
                outputs = model(input_ids, attention_mask)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                preds.extend(batch_preds)
                true_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        fp = sum((np.array(preds) == 1) & (np.array(true_labels) == 0))
        fn = sum((np.array(preds) == 0) & (np.array(true_labels) == 1))
        return preds, true_labels, all_probs, fp, fn

    test_preds, test_labels, test_probs, fp, fn = eval_model(test_loader)
    f1 = f1_score(test_labels, test_preds, average='binary')
    precision = precision_score(test_labels, test_preds)
    recall = recall_score(test_labels, test_preds)
    roc_auc = roc_auc_score(test_labels, test_probs)
    pr_auc = average_precision_score(test_labels, test_probs)

    print('Test Results')
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")


if __name__ == '__main__':
    test()
