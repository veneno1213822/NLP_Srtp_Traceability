# Introduciton
A text-to-table intelligent traceability model based on extractive multi-fragment reading comprehension model.

Available in both Chinese and English versions. Excluding overly large pre trained models, datasets, etc

Chinese version uses the dataset from the SM competition. English version uses the ToTTo dataset.
# Experimental steps
There are two versions available in Chinese and English.
## Download datasets, pre-trained models, etc
```
1. Datasets:
For Chinese, dataset is already in the \data\ folder.
For English, put the train and dev sets from https://storage.googleapis.com/totto-public/totto_data.zip into ModifiedEnglish\data\ .
2. Pre-trained Models:
Dowload config.json, pytorch_model.bin and vocab.txt from https://huggingface.co/bert-base-chinese/tree/main or https://huggingface.co/bert-base-uncased/tree/main.
Put them into OfficialChinese\models\bert-base-chinese\ or \ModifiedEnglish\models\bert-base-uncased\
```

## Running Code
### Chinse Version
Run data_split.py for data partitioning (training set, validation set)

Run run_data_process.sh to perform data processing. Pay attention to absolute path.

Run run_cail2021_TrainAndEval.py to train and evaluate model.

### Engish Version
Run Preparation.py to perform preparation work from ToTTo dataset.

Run run_data_process.sh to perform data processing. Pay attention to absolute path.

Run run_cail2021_TrainAndEval.py to train and evaluate model.