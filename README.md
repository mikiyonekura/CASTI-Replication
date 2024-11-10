# CASTI

This repository is the replication package including the source code and the datasets.

## Repository Details
```
.
├── datasets
│   ├── maldonado_dataset.json
│   └── studied_dataset.json
│
├── replication-package
│   ├── BERT_CNN
│   │   ├── test_bert_cnn.py
│   │   └── train_bert_cnn.py
│   │
│   └── CASTI
│       ├── test_casti.py
│       └── train_casti.py
│
└── requirements.txt
```

## require
```
requirements.txt
```

## RUN
Operation confirmed on only Linux

### RQ3. CASTI, BERT+CNN and PILOT prediction
(if you need) run ```replication-package/CASTI/train_casti.py``` to create your model.
Run ```replication-package/CASTI/test_casti.py``` 


(if you need) run ```replication-package/CASTI/train_bert_cnn.py``` to create your model.
Run ```replication-package/BERT_CNN/test_bert_cnn.py```

Note：For the PILOT replication package, see the reference included in the paper “PILOT: Synergy between Text Processing and Neural Networks to Detect Self-Admitted Technical Debt”.