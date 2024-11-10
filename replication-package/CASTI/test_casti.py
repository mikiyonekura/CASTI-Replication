import torch
import numpy as np
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
import json


def prepare_data(data, tokenizer, code_lines_limit=None):
    combined_texts = []
    for x in data:
        comment = x['comment']
        code = x['code']
        if code_lines_limit is not None:
            code_lines = code.split('\n')
            code = '\n'.join(code_lines[:code_lines_limit])
        combined_texts.append(comment + " </s> " + code)
    encodings = tokenizer(combined_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    labels = [x['label'] for x in data]
    return encodings, labels


class SATDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def main():
    model_name = 'replication-package/CASTI/pretrained_models/pretrained_casti_example'
    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = 'datasets/studied_dataset.json'
    with open(data_path, 'r') as f:
        data_json = json.load(f)

    data = [{
        'comment': item['comment'],
        'code': item['relevant_code'],
        'label': 0 if item['maldonado_dataset_classification'] == 'WITHOUT_CLASSIFICATION' else 1
    } for item in data_json]

    _, test_valid_data = train_test_split(data, test_size=0.4, random_state=42)
    _, test_data = train_test_split(test_valid_data, test_size=0.5, random_state=42)

    test_encodings, test_labels = prepare_data(test_data, tokenizer, code_lines_limit=32)
    test_dataset = SATDDataset(test_encodings, test_labels)

    trainer = Trainer(model=model)
    test_results = trainer.predict(test_dataset)
    predictions = np.argmax(test_results.predictions, axis=-1)
    probs = torch.nn.functional.softmax(torch.tensor(test_results.predictions), dim=-1)[:, 1]

    f1 = f1_score(test_labels, predictions, average='binary')
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    roc_auc = roc_auc_score(test_labels, probs)
    precision_vals, recall_vals, _ = precision_recall_curve(test_labels, probs)
    pr_auc = auc(recall_vals, precision_vals)
    cm = confusion_matrix(test_labels, predictions)
    fp = cm[0][1]
    fn = cm[1][0]

    print("Model Test Results")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")


if __name__ == '__main__':
    main()
