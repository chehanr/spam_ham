# spam_ham

A simple spam/ ham email classifier.  
Downoad the data set: [https://www.kaggle.com/veleon/ham-and-spam-dataset](https://www.kaggle.com/veleon/ham-and-spam-dataset)

## Usage

Install packages using `poetry install` or `pip install -r requirements.txt`

### Train

```shell
usage: check_spam.py train [-h] [--save_path SAVE_PATH] --ham_path HAM_PATH --spam_path SPAM_PATH

optional arguments:
  -h, --help            show this help message and exit
  --save_path SAVE_PATH
                        model save location
  --ham_path HAM_PATH   ham email files location
  --spam_path SPAM_PATH
                        spam email files location
```

### Predict

```shell
usage: check_spam.py predict [-h] [--model_path MODEL_PATH] [--output_path OUTPUT_PATH] --email_path EMAIL_PATH

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        model location
  --output_path OUTPUT_PATH
                        CSV output location
  --email_path EMAIL_PATH
                        email files location
```
