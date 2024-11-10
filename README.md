# CASTI
This repository is the replication package including the source code and the datasets.

## Repository Details
```
├── datasets
│   ├── maldonado_dataset.json
│   └── studied_dataset.json
│
├── replication-package
│   ├── BERT_CNN
│   │   ├── pretrained_models
│   │   │   └── pretrained_bert_cnn_example
│   │   │      
│   │   ├── test_bert_cnn.py
│   │   └── train_bert_cnn.py
│   │
│   └── CASTI
│       ├── pretrained_models
│       │   └── pretrained_casti_example
│       │
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


## CASTI　prediction
Run ```replication-package/CASTI/test_casti.py``` 

(if you need) run ```replication-package/CASTI/train_casti.py``` to create your model.

## BERT＋CNN prediction
Run ```replication-package/BERT_CNN/test_bert_cnn.py```

(if you need) run ```replication-package/CASTI/train_bert_cnn.py``` to create your model.

## PILOT
Note：For the PILOT replication package, see the reference included in the paper “PILOT: Synergy between Text Processing and Neural Networks to Detect Self-Admitted Technical Debt”.