import torch
from torch.optim import AdamW
from transformers import DistilGPT2ForSequenceClassification, DistilGPT2Tokenizer, get_linear_schedule_with_warmup
from data_processor import prepare_imdb_data, create_data_loaders
from tqdm import tqdm
import numpy as np

def train(model, train_dataloader, val_dataloader, epochs, device):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        
        model.train()
        train_loss = 0
        train_accuracy = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_accuracy += (preds == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_accuracy = train_accuracy / len(train_dataloader.dataset)
        
        print(f"Training loss: {avg_train_loss:.4f}")
        print(f"Training accuracy: {avg_train_accuracy:.4f}")
        
        model.eval()
        val_loss = 0
        val_accuracy = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_accuracy += (preds == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_accuracy = val_accuracy / len(val_dataloader.dataset)
        
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation accuracy: {avg_val_accuracy:.4f}")
    
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = DistilGPT2Tokenizer.from_pretrained('distilgpt2')
    model = DistilGPT2ForSequenceClassification.from_pretrained('distilgpt2', num_labels=2)
    model.to(device)

    texts, labels = prepare_imdb_data()
    
    # Split data into train and validation sets
    split = int(0.9 * len(texts))
    train_texts, val_texts = texts[:split], texts[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    train_dataloader = create_data_loaders(train_texts, train_labels, tokenizer, max_length=128, batch_size=16)
    val_dataloader = create_data_loaders(val_texts, val_labels, tokenizer, max_length=128, batch_size=16)

    epochs = 3
    trained_model = train(model, train_dataloader, val_dataloader, epochs, device)

    # Save the model
    torch.save(trained_model.state_dict(), 'fine_tuned_distilgpt2.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()