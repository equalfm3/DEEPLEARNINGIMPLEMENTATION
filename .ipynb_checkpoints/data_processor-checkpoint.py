import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import os
import requests
import tarfile
from tqdm import tqdm

def download_file(url, filename):
    """Download a file from a URL to a local file."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def extract_dataset(filename):
    """Extract the dataset from the downloaded tar.gz file."""
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()

def prepare_imdb_data():
    """Download and prepare the IMDB dataset."""
    DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    DATASET_FILENAME = "aclImdb_v1.tar.gz"

    if not os.path.exists(DATASET_FILENAME):
        print("Downloading IMDB dataset...")
        download_file(DATASET_URL, DATASET_FILENAME)
    
    if not os.path.exists("aclImdb"):
        print("Extracting dataset...")
        extract_dataset(DATASET_FILENAME)
    
    # Read positive and negative reviews
    pos_files = os.listdir("aclImdb/train/pos")
    neg_files = os.listdir("aclImdb/train/neg")

    texts, labels = [], []

    for file in pos_files:
        with open(f"aclImdb/train/pos/{file}", "r", encoding="utf-8") as f:
            texts.append(f.read())
            labels.append(1)

    for file in neg_files:
        with open(f"aclImdb/train/neg/{file}", "r", encoding="utf-8") as f:
            texts.append(f.read())
            labels.append(0)

    return texts, labels

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(texts, labels, tokenizer, max_length, batch_size):
    # Set the pad token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = TextDataset(texts, labels, tokenizer, max_length)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

if __name__ == "__main__":
    # Test the data processing
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    
    print("Preparing IMDB dataset...")
    texts, labels = prepare_imdb_data()
    
    print(f"Dataset size: {len(texts)} samples")
    
    data_loader = create_data_loaders(texts, labels, tokenizer, max_length=128, batch_size=16)
    
    for batch in data_loader:
        print("Input shape:", batch['input_ids'].shape)
        print("Attention mask shape:", batch['attention_mask'].shape)
        print("Labels shape:", batch['labels'].shape)
        break

    print("Data processing test complete!")