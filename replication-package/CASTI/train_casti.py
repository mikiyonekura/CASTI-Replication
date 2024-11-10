import torch
import numpy as np
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, roc_auc_score, auc
import json
import matplotlib.pyplot as plt


def calculate_weights(labels):
    count_0 = labels.count(0)
    count_1 = labels.count(1)
    weight_0 = 1. / count_0
    weight_1 = 1. / count_1
    total = weight_0 + weight_1
    weight_0 /= total
    weight_1 /= total
    return [weight_0, weight_1]


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
    model_name = 'microsoft/codebert-base'
    try:
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    data_path = 'datasets/studied_dataset.json'
    with open(data_path, 'r') as f:
        data_json = json.load(f)

    data = [{
        'comment': item['comment'],
        'code': item['relevant_code'],
        'label': 0 if item['maldonado_dataset_classification'] == 'WITHOUT_CLASSIFICATION' else 1
    } for item in data_json]

    train_data, test_valid_data = train_test_split(data, test_size=0.4, random_state=42)
    valid_data, test_data = train_test_split(test_valid_data, test_size=0.5, random_state=42)

    train_encodings, train_labels = prepare_data(train_data, tokenizer, code_lines_limit=32)
    valid_encodings, valid_labels = prepare_data(valid_data, tokenizer, code_lines_limit=32)
    test_encodings, test_labels = prepare_data(test_data, tokenizer, code_lines_limit=32)

    train_dataset = SATDDataset(train_encodings, train_labels)
    valid_dataset = SATDDataset(valid_encodings, valid_labels)
    test_dataset = SATDDataset(test_encodings, test_labels)

    weights = calculate_weights(train_labels)
    weights = torch.tensor(weights).to('cuda' if torch.cuda.is_available() else 'cpu')

    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.loss_fct = torch.nn.CrossEntropyLoss(weight=weights)

    training_args = TrainingArguments(
        output_dir='./replication-package/CASTI/results',
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.1,
        evaluation_strategy="epoch",
        learning_rate=5e-6,
        save_strategy="epoch",
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=5)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping]
    )

    trainer.train()
    model_path = 'replication-package/CASTI/pretrained_models/test_model'
    trainer.save_model(model_path)

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

    print("Model Evaluation Results")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")


if __name__ == '__main__':
    main()
