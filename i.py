import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# File paths (adjusted for your setup)
train_path = r"C:\Users\adnan\Desktop\offensivelang\mal_full_offensive_train (1).csv"
dev_path = r"C:\Users\adnan\Desktop\offensivelang\mal_full_offensive_dev.csv"

# Load CSV files
train_df = pd.read_csv(train_path)
dev_df = pd.read_csv(dev_path)

# Preview
print("Train samples:", len(train_df))
print("Validation samples:", len(dev_df))
train_df.head()

from transformers import DistilBertTokenizerFast

# Load Multilingual DistilBERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-multilingual-cased')

# Example test
example_text = train_df['Text'][0]
print("Original text:", example_text)
print("Tokenized output:", tokenizer(example_text))
label_column = 'Labels'

label_mapping = {
    'Not_offensive': 1,
    'not-malayalam': 2,
    'Offensive_Targeted_Insult_Individual': 0,
    'Offensive_Targeted_Insult_Group': 0,
    'Offensive_Untargetede': 0
}

train_df['label_id'] = train_df[label_column].map(label_mapping)
dev_df['label_id'] = dev_df[label_column].map(label_mapping)

print(train_df[['Text', label_column, 'label_id']].head())
print(train_df['label_id'].value_counts())
